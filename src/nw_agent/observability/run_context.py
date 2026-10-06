"""一次运行的观测上下文：阶段耗时、状态快照、token 用量、结局，收进一个可落盘的目录。

**这是 IO 与时间戳的唯一负责人**。``evals/runner.py`` 明文约定「不决定落盘位置、不打
时间戳」，它只把观察到的阶段耗时**交给注入的 recorder**；落哪个目录、写什么文件、
盖什么时间戳，全在本模块。这样 runner 在测试里永远是纯的（默认注入 :class:`NoOpRecorder`，
一行文件都不写）。

**默认值必须是 NoOpRecorder 本身**，而不是「一个会落盘的 RunContext 但恰好根目录被
指向了临时目录」：前者是类型级保证（忘了传 recorder 就一定不落盘），后者只是环境巧合
（一旦 ``NW_HOME`` 没被隔离，测试就往用户目录里写）。
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from nw_agent.observability.ledger import (
    RunRecord,
    StageRecord,
    TokenUsage,
    append,
    load_prices,
)
from nw_agent.observability.store import (
    SNAPSHOTS_DIRNAME,
    SUMMARY_FILENAME,
    ensure_parent,
    ledger_path,
    nw_home,
    run_dir,
)
from nw_agent.observability.store import prices_path as default_prices_path
from nw_agent.observability.tracing import TracingStatus, configure_tracing

# 阶段结局。stage 名由调用方给（P0 是 create/upload/solve/verify；P3 是图节点名）。
_STATUS_OK = "ok"
_STATUS_ERROR = "error"


class Recorder(Protocol):
    """观测接缝：被测流程只依赖这四个方法，不认识文件系统、时间戳与账本。

    刻意全部返回 ``None``：观察者不该有返回值，否则调用方会开始依赖它，
    「关掉观测会改变行为」这种事就离得不远了。
    """

    def stage(self, name: str, seconds: float, *, status: str = _STATUS_OK) -> None:
        """记录一个阶段的耗时与结局。"""
        ...

    def snapshot(self, name: str, state: Mapping[str, object]) -> None:
        """把某阶段的状态快照落盘。``state`` 必须可 JSON 序列化。"""
        ...

    def record_tokens(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """记录一次模型调用的用量。模型名由调用方注入（见产品说明 L240）。"""
        ...

    def finish(self, outcome: str, *, error: str | None = None) -> None:
        """收尾：写运行摘要并把账本行追加进账本。"""
        ...


class NoOpRecorder:
    """什么都不做。**不传 recorder 时的默认值**（理由见模块 docstring）。"""

    def stage(self, name: str, seconds: float, *, status: str = _STATUS_OK) -> None:
        """忽略。"""

    def snapshot(self, name: str, state: Mapping[str, object]) -> None:
        """忽略。"""

    def record_tokens(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """忽略。"""

    def finish(self, outcome: str, *, error: str | None = None) -> None:
        """忽略。"""


@dataclass
class _Accumulator:
    """可变累加器（``RunRecord`` 是 frozen 的，流程中先攒着）。"""

    stages: list[StageRecord]
    tokens: list[TokenUsage]


class RunContext:
    """一次运行的观测上下文。落盘形态见 doc/可观测性.md。

    典型用法（脚本侧）::

        ctx = RunContext.start("eval", repo=repo, prices_path=args.prices)
        report = run_case(case, solver, factory, recorder=ctx)
        ctx.finish(report.outcome, error=report.error)
    """

    def __init__(
        self,
        *,
        kind: str,
        home: Path | None = None,
        run_id: str | None = None,
        repo: str | None = None,
        backend_kind: str | None = None,
        solver: str | None = None,
        case_id: str | None = None,
        prices_path: Path | None = None,
        tracing: TracingStatus | None = None,
    ) -> None:
        """**由 :meth:`start` 调用**；直接构造也可以，但那时要自己解出 tracing 状态。

        Args:
            kind: 运行类别（``eval`` / ``cli``），账本按它过滤。
            home: 运行产物根目录；None 时按 ``NW_HOME`` / ``~/.nightwatch`` 解析。
            run_id: 显式指定运行 ID（测试用）；None 时自动生成。
            repo: 目标仓库标识，账本里据此区分仓库。
            backend_kind / solver / case_id: 评测链路才有的标签，可为 None。
            prices_path: 价格表路径；None 表示用标准位置 ``<home>/prices.json``
                （文件不存在即视为未配置价格，费用恒为 None；**不内置任何价格**）。
            tracing: 本次运行的 tracing 状态，进账本供事后核对。
        """
        self._home = nw_home(home)
        self._kind = kind
        self._repo = repo
        self._backend_kind = backend_kind
        self._solver = solver
        self._case_id = case_id
        self._prices = load_prices(prices_path or default_prices_path(self._home))
        self._tracing = tracing
        self._run_id = run_id or _new_run_id()
        self._started_at = _now()
        self._acc = _Accumulator(stages=[], tokens=[])
        self._snapshot_seq = 0
        self._finished: RunRecord | None = None

    @classmethod
    def start(
        cls,
        kind: str,
        *,
        home: Path | None = None,
        run_id: str | None = None,
        repo: str | None = None,
        backend_kind: str | None = None,
        solver: str | None = None,
        case_id: str | None = None,
        prices_path: Path | None = None,
        trace: bool = False,
        project: str | None = None,
    ) -> RunContext:
        """建上下文，并按 ``trace`` 配置 LangSmith。

        ``trace`` 在这里转成 :class:`TracingStatus` 记进账本——「请求开启」与
        「真的开启了」是两件事，账本要记的是后者。
        """
        return cls(
            kind=kind,
            home=home,
            run_id=run_id,
            repo=repo,
            backend_kind=backend_kind,
            solver=solver,
            case_id=case_id,
            prices_path=prices_path,
            tracing=configure_tracing(trace, project=project),
        )

    # ---------------------------------------------------------------- 只读属性
    @property
    def run_id(self) -> str:
        """本次运行的 ID。"""
        return self._run_id

    @property
    def home(self) -> Path:
        """运行产物根目录。"""
        return self._home

    @property
    def run_dir(self) -> Path:
        """本次运行的产物目录（**未必已存在**——目录只在真写文件时才建）。"""
        return run_dir(self._home, self._run_id)

    @property
    def ledger_path(self) -> Path:
        """账本文件路径。"""
        return ledger_path(self._home)

    @property
    def tracing(self) -> TracingStatus | None:
        """本次运行的 tracing 状态。"""
        return self._tracing

    @property
    def record(self) -> RunRecord | None:
        """收尾后的账本行；未 :meth:`finish` 时为 None。"""
        return self._finished

    @property
    def started_at(self) -> str:
        """开始时刻（ISO8601，带时区）。"""
        return self._started_at

    # ------------------------------------------------------------------ 记录
    def stage(self, name: str, seconds: float, *, status: str = _STATUS_OK) -> None:
        """记录一个阶段。可重复调用（同名阶段出现多次是正常的，例如重试）。"""
        self._acc.stages.append(StageRecord(name=name, seconds=seconds, status=status))

    def snapshot(self, name: str, state: Mapping[str, object]) -> None:
        """把状态快照写成 ``runs/<run_id>/snapshots/<seq>-<name>.json``。

        序号前缀保证目录按发生顺序可读（``ls`` 即时间线）。快照**立即落盘**而不是攒到
        收尾统一写：进程中途崩掉时，已经历过的阶段必须留下痕迹。
        """
        self._snapshot_seq += 1
        path = self.run_dir / SNAPSHOTS_DIRNAME / f"{self._snapshot_seq:02d}-{name}.json"
        ensure_parent(path)
        payload = {"captured_at": _now(), "stage": name, "state": dict(state)}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def record_tokens(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """记录一次模型调用的用量。多次调用累加（主代理 + 各子代理各记一次）。

        P0 没有调用点（没有 LLM 调用），P1 的 ``run_agent`` 节点会接上；本方法由
        单测覆盖，以免成为从未执行过的死代码。
        """
        self._acc.tokens.append(
            TokenUsage(model=model, input_tokens=input_tokens, output_tokens=output_tokens)
        )

    def finish(
        self,
        outcome: str,
        *,
        error: str | None = None,
        escaped: bool | None = None,
    ) -> None:
        """收尾：写 ``summary.json`` 并把一行追加进账本。

        Args:
            outcome: ``success`` / ``fail`` / ``error`` / ``skeleton``。
            error: 失败原因（``error`` 时应当给）。
            escaped: 是否发生越权/注入逃逸（5.7 的硬红线）。**由调用方从评测报告里读出
                再传入**，而不是让被测流程自己上报——这是被观测对象的属性，不是它的自述。

        Raises:
            RuntimeError: 重复收尾。一次运行只该有一行账本，重复收尾必然让 5.7 计数虚高。
            DuplicateRunError: 账本里已有同名 ``run_id``。
        """
        if self._finished is not None:
            raise RuntimeError(f"run {self._run_id} 已收尾，不能重复 finish")
        finished_at = _now()
        tokens = tuple(self._acc.tokens)
        missing = self._prices.missing(tokens)
        costs = [self._prices.cost_usd(usage) for usage in tokens]
        record = RunRecord(
            run_id=self._run_id,
            kind=self._kind,
            outcome=outcome,
            started_at=self._started_at,
            finished_at=finished_at,
            duration_seconds=_span(self._started_at, finished_at),
            stages=tuple(self._acc.stages),
            tokens=tokens,
            # 没有调用就是 None（不是 0）；有调用才给总数。
            total_tokens=(sum(usage.total for usage in tokens) if tokens else None),
            # 只要有一个模型缺单价就整体记 None：部分求和会把「算不全」伪装成「很便宜」。
            cost_usd=(None if missing or not tokens else sum(c for c in costs if c is not None)),
            missing_price_models=missing,
            prices_source=self._prices.source,
            repo=self._repo,
            backend_kind=self._backend_kind,
            solver=self._solver,
            case_id=self._case_id,
            escaped=escaped,
            tracing=self._tracing,
            error=error,
        )
        summary = self.run_dir / SUMMARY_FILENAME
        ensure_parent(summary)
        summary.write_text(
            json.dumps(record.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        append(record, path=self.ledger_path)
        self._finished = record


def _new_run_id() -> str:
    """生成运行 ID：``<UTC 时间戳>-<随机后缀>``。

    时间戳在前是为了让账本按 ``run_id`` 排序即等价于按时间排序；后缀保证同秒内多次
    运行不撞车（撞车会触发账本判重）。
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _now() -> str:
    """带时区的当前时刻（ISO8601，秒级）。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _span(started_at: str, finished_at: str) -> float:
    """两个 ISO8601 时刻之间的秒数；解析不了就返回 0（观测数据不该拖垮运行）。"""
    try:
        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(finished_at)
    except ValueError:
        return 0.0
    return (end - start).total_seconds()