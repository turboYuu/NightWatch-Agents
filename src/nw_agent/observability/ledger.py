"""运行账本：一次运行一行 JSON，供 5.7 度量表按周采集。

为什么是 JSONL 而不是 SQLite：写入量是「每天几十行」，而它要满足的是**可读不可改**
的账本语义——只追加、可 grep/jq、无建表与迁移成本。查询需求（按 repo/kind 过滤、
按周聚合）由 :func:`summarize` 与 ``scripts/report_ledger.py`` 承担，不需要 SQL。

**成本在写入时定格**（``RunRecord.cost_usd`` 由 :class:`~nw_agent.observability.
run_context.RunContext` 按当时的价格表算好）。事后改 ``prices.json`` **不会**回改历史
记录——账本记的是「当时按什么价格算出了多少」，这是账本该有的性质，也让
:func:`summarize` 成为纯聚合、无需再拿到价格表。

**每条记录自带 ``schema_version``**（不放在文件头）：JSONL 逐行独立、可跨版本追加，
文件头那套在这里没有立足点。读到不认识的版本一律跳过并计数，**绝不用当前口径去解释
它**——那会把「口径变更」伪装成「指标回归」。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from nw_agent.observability.store import ensure_parent
from nw_agent.observability.tracing import TracingStatus

LEDGER_SCHEMA_VERSION = 1

# 各字段在报告里的取值。outcome 的 error 语义与评测报告一致：harness 故障，
# 不计入成功率分母（见 evals/runner.py 的同名口径）。
OUTCOMES = ("success", "fail", "error", "skeleton")


class DuplicateRunError(RuntimeError):
    """同一 ``run_id`` 被写入两次。

    这是护栏而非限制：正常重跑会生成新的 ``run_id``，只有「同一次运行被 append 两遍」
    才会命中——那必然让 5.7 的计数虚高。
    """


@dataclass(frozen=True)
class StageRecord:
    """一个阶段（或 P3 的一个节点）的耗时与结局。"""

    name: str
    seconds: float
    status: str = "ok"  # ok | error

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "seconds": round(self.seconds, 4), "status": self.status}


@dataclass(frozen=True)
class TokenUsage:
    """一次模型调用的用量。模型名由调用方注入，**不在这里硬编码任何模型**。"""

    model: str
    input_tokens: int
    output_tokens: int

    @property
    def total(self) -> int:
        """该次调用的总 token。"""
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass(frozen=True)
class RunRecord:
    """一次运行的账本行。

    列表字段用 ``tuple``：frozen dataclass 里的 list 仍可被就地改，等于冻结失败。
    ``tokens`` 为空元组表示「本次运行没有模型调用」，与 ``total_tokens is None`` 同义；
    两者都不是 0——0 会被读成「零成本」。
    """

    run_id: str
    kind: str  # eval | cli
    outcome: str  # OUTCOMES 之一
    started_at: str
    finished_at: str
    duration_seconds: float
    stages: tuple[StageRecord, ...] = ()
    tokens: tuple[TokenUsage, ...] = ()
    total_tokens: int | None = None
    cost_usd: float | None = None
    missing_price_models: tuple[str, ...] = ()
    prices_source: str | None = None
    repo: str | None = None
    backend_kind: str | None = None
    solver: str | None = None
    case_id: str | None = None
    escaped: bool | None = None
    tracing: TracingStatus | None = None
    error: str | None = None
    schema_version: int = LEDGER_SCHEMA_VERSION

    def as_dict(self) -> dict[str, object]:
        """手写展开而非 ``dataclasses.asdict``：后者递归返回 ``dict[str, Any]``，
        在 mypy strict 下比手写更松，也和 ``CaseReport.as_dict`` 的既有范式不一致。"""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "kind": self.kind,
            "outcome": self.outcome,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": round(self.duration_seconds, 4),
            "stages": [stage.as_dict() for stage in self.stages],
            "tokens": [usage.as_dict() for usage in self.tokens],
            "total_tokens": self.total_tokens,
            "cost_usd": None if self.cost_usd is None else round(self.cost_usd, 6),
            "missing_price_models": list(self.missing_price_models),
            "prices_source": self.prices_source,
            "repo": self.repo,
            "backend_kind": self.backend_kind,
            "solver": self.solver,
            "case_id": self.case_id,
            "escaped": self.escaped,
            "tracing": None if self.tracing is None else self.tracing.as_dict(),
            "error": self.error,
        }


@dataclass(frozen=True)
class PriceTable:
    """模型单价表：``{model: {"input": 美元/百万 token, "output": ...}}``。

    **不内置任何价格**：产品说明 L240 要求模型名与单价以配置项注入，写死在代码里的
    价格表会在供应商调价后静默算错。``source is None`` 表示未提供价格表。
    """

    source: str | None = None
    # default_factory 而非直接写 {}：可变默认值在 dataclass 里是所有实例共享同一份，
    # 一旦有人就地修改就会串味。
    per_model: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def cost_usd(self, usage: TokenUsage) -> float | None:
        """算一次调用的费用；**该模型没有单价就返回 ``None``**（不是 0）。"""
        price = self.per_model.get(usage.model)
        if price is None:
            return None
        # 用 in 判定而非真值判定：单价 0 是合法的（免费档位），不能被当成「缺价格」。
        if "input" not in price or "output" not in price:
            return None
        return (usage.input_tokens * price["input"] + usage.output_tokens * price["output"]) / 1e6

    def missing(self, usages: Iterable[TokenUsage]) -> tuple[str, ...]:
        """列出用到了却没有单价的模型（去重且有序，便于报告稳定）。"""
        return tuple(sorted({u.model for u in usages if u.model not in self.per_model}))


def load_prices(path: Path | None) -> PriceTable:
    """读取价格表；``None`` 或文件不存在都返回空表（不是错误）。

    Args:
        path: ``prices.json`` 路径；None 表示未提供。

    Returns:
        价格表；``source`` 记录来源路径，写进账本便于事后追溯用的是什么价格。

    Raises:
        ValueError: 文件存在但内容不是合法价格表。**宁可显式失败**，也不要静默退回
            空表——那会把「配置写错了」伪装成「这次运行没花钱」。
    """
    if path is None or not path.is_file():
        return PriceTable()
    with path.open(encoding="utf-8") as fh:
        data: object = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}：价格表顶层必须是对象（model -> 单价）")
    parsed: dict[str, Mapping[str, float]] = {}
    for model, price in data.items():
        if not isinstance(price, dict):
            raise ValueError(f"{path}：模型 {model!r} 的单价必须是对象")
        entry: dict[str, float] = {}
        for key in ("input", "output"):
            value = cast("dict[str, object]", price).get(key)
            # 显式排除 bool：JSON 里的 true 也是 int 的子类，会被 isinstance 放过去，
            # 于是「手抖写成 true」变成单价 1.0——这种静默错值最难查。
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{path}：模型 {model!r} 缺少数值型的 {key!r}（美元/百万 token）")
            entry[key] = float(value)
        parsed[str(model)] = entry
    return PriceTable(source=str(path), per_model=parsed)


@dataclass(frozen=True)
class Ledger:
    """读账本的结果。坏行与未知版本**不抛异常**——账本是追加写的，一行坏不该让整份读不出来。"""

    path: Path
    records: tuple[RunRecord, ...]
    skipped_lines: int = 0
    skipped_unknown_version: int = 0


def read(path: Path) -> Ledger:
    """读账本。文件不存在视为空账本（首次运行就是这种情况，不是错误）。"""
    if not path.is_file():
        return Ledger(path=path, records=())
    records: list[RunRecord] = []
    skipped_lines = 0
    skipped_version = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            raw: object = json.loads(line)
            version = _field(raw, "schema_version")
            if version != LEDGER_SCHEMA_VERSION:
                skipped_version += 1
                continue
            records.append(_record_from(raw))
        except (ValueError, KeyError, TypeError):
            # 单行损坏不该毁掉整份账本：跳过并计数，由调用方决定要不要吭声。
            skipped_lines += 1
    return Ledger(
        path=path,
        records=tuple(records),
        skipped_lines=skipped_lines,
        skipped_unknown_version=skipped_version,
    )


def append(record: RunRecord, *, path: Path) -> None:
    """追加一行。

    写前扫描已有 ``run_id`` 判重（账本规模在几十到几百行，O(n) 扫描的代价可忽略，
    换来的是不需要索引文件这种额外状态）。

    Raises:
        DuplicateRunError: ``record.run_id`` 已在账本里。
    """
    existing = {item.run_id for item in read(path).records}
    if record.run_id in existing:
        raise DuplicateRunError(f"run_id 已存在，拒绝重复写入：{record.run_id}")
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")


def summarize(records: Sequence[RunRecord]) -> dict[str, object]:
    """把账本行聚合成 5.7 度量表要的四个指标。

    与评测报告同一口径：``error`` 不计入成功率分母（harness 故障不该记在被测对象账上），
    但它单独计数，避免坏数据被悄悄稀释。缺数据的指标一律 ``None`` 而非 0。
    """
    total = len(records)
    counts = {outcome: sum(1 for r in records if r.outcome == outcome) for outcome in OUTCOMES}
    scored = total - counts["error"]
    escapes = sum(1 for r in records if r.escaped)
    durations = [r.duration_seconds for r in records]
    tokens = [r.total_tokens for r in records if r.total_tokens is not None]
    costs = [r.cost_usd for r in records if r.cost_usd is not None]
    missing = sorted({model for r in records for model in r.missing_price_models})
    return {
        "runs": total,
        "outcome_counts": counts,
        "success_rate": (counts["success"] / scored) if scored else None,
        "escape_count": escapes,
        "duration_seconds": {
            "mean": (round(sum(durations) / total, 4) if total else None),
            "max": (round(max(durations), 4) if durations else None),
        },
        "total_tokens": (sum(tokens) if tokens else None),
        # 有 token 数据的运行数：为 0 说明「没有消费」，不是「消费了但没算钱」。
        "runs_with_tokens": len(tokens),
        "total_cost_usd": (round(sum(costs), 6) if costs else None),
        "priced_runs": len(costs),
        "missing_price_models": missing,
        # 已确认开启 tracing 的运行数。P0 里它大于 0 只说明「开关与 key 配好了」，
        # 不代表 LangSmith 上有东西可看——见 doc/可观测性.md。
        "tracing_enabled_runs": sum(
            1 for r in records if r.tracing is not None and r.tracing.enabled
        ),
    }


# ---------------------------------------------------------------------------
# 反序列化（把 Any 关在这里）
# ---------------------------------------------------------------------------
def _field(obj: object, key: str) -> object:
    if not isinstance(obj, dict):
        raise ValueError("账本行必须是 JSON 对象")
    return cast("dict[str, object]", obj)[key]


def _str_or_none(obj: object, key: str) -> str | None:
    value = _field(obj, key)
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"{key} 必须是字符串或 null")


def _require_str(obj: object, key: str) -> str:
    value = _field(obj, key)
    if not isinstance(value, str):
        raise TypeError(f"{key} 必须是字符串")
    return value


def _require_float(obj: object, key: str) -> float:
    value = _field(obj, key)
    if not isinstance(value, int | float):
        raise TypeError(f"{key} 必须是数值")
    return float(value)


def _record_from(raw: object) -> RunRecord:
    """把一行 JSON 还原成 :class:`RunRecord`。字段不全或类型不符一律抛，由 ``read`` 计为坏行。"""
    stages_raw = _field(raw, "stages")
    tokens_raw = _field(raw, "tokens")
    if not isinstance(stages_raw, list) or not isinstance(tokens_raw, list):
        raise TypeError("stages/tokens 必须是数组")
    total_tokens = _field(raw, "total_tokens")
    cost_usd = _field(raw, "cost_usd")
    escaped = _field(raw, "escaped")
    tracing_raw = _field(raw, "tracing")
    if total_tokens is not None and not isinstance(total_tokens, int):
        raise TypeError("total_tokens 必须是整数或 null")
    if cost_usd is not None and not isinstance(cost_usd, int | float):
        raise TypeError("cost_usd 必须是数值或 null")
    if escaped is not None and not isinstance(escaped, bool):
        raise TypeError("escaped 必须是布尔或 null")
    return RunRecord(
        schema_version=LEDGER_SCHEMA_VERSION,
        run_id=_require_str(raw, "run_id"),
        kind=_require_str(raw, "kind"),
        outcome=_require_str(raw, "outcome"),
        started_at=_require_str(raw, "started_at"),
        finished_at=_require_str(raw, "finished_at"),
        duration_seconds=_require_float(raw, "duration_seconds"),
        stages=tuple(_stage_from(item) for item in stages_raw),
        tokens=tuple(_usage_from(item) for item in tokens_raw),
        total_tokens=total_tokens,
        cost_usd=None if cost_usd is None else float(cost_usd),
        missing_price_models=tuple(
            str(model) for model in cast("list[object]", _field(raw, "missing_price_models"))
        ),
        prices_source=_str_or_none(raw, "prices_source"),
        repo=_str_or_none(raw, "repo"),
        backend_kind=_str_or_none(raw, "backend_kind"),
        solver=_str_or_none(raw, "solver"),
        case_id=_str_or_none(raw, "case_id"),
        escaped=escaped,
        tracing=None if tracing_raw is None else TracingStatus.from_dict(tracing_raw),
        error=_str_or_none(raw, "error"),
    )


def _stage_from(raw: object) -> StageRecord:
    return StageRecord(
        name=_require_str(raw, "name"),
        seconds=_require_float(raw, "seconds"),
        status=_require_str(raw, "status"),
    )


def _usage_from(raw: object) -> TokenUsage:
    input_tokens = _field(raw, "input_tokens")
    output_tokens = _field(raw, "output_tokens")
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        raise TypeError("input_tokens/output_tokens 必须是整数")
    return TokenUsage(
        model=_require_str(raw, "model"),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
