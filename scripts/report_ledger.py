#!/usr/bin/env python
"""从运行账本汇总 5.7 度量表（见 doc/开发路线图.md 5.7 与 doc/可观测性.md）。

用法::

    python scripts/report_ledger.py                      # 全量
    python scripts/report_ledger.py --kind eval --since 7d
    python scripts/report_ledger.py --repo owner/name --json-out /tmp/ledger.json

**退出码只反映账本可读性**（对齐 run_eval.py 的既有约定：退出码不反映阈值是否达标）。
原因：阈值是否达标取决于你筛的是哪批运行，拿它当退出码会让「查历史」这件事变得无法自动化。

**两类「无成本数据」的文案必须分开**，它们的原因完全不同，混成一句就会误导：
- 一条 token 记录都没有 → 「没有模型调用」（P0 就是这种）；
- 有 token 但缺单价 → 要写出缺哪些模型，让人知道该去 ``prices.json`` 里补什么。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

# 本脚本不在包内，而它要 import 生产代码。把仓库的 src/ 提前放进 sys.path，
# 这样无需先 `pip install -e .` 也能跑（下面的 E402 因此无法避免，属刻意）。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from nw_agent.cli.main import DEFAULT_TOKEN_BUDGET  # noqa: E402
from nw_agent.evals.runner import THRESHOLDS  # noqa: E402
from nw_agent.observability import (  # noqa: E402
    RunRecord,
    ledger_path,
    nw_home,
    read,
    summarize,
)

# ``--since`` 的两种写法：相对时长（7d / 12h）或 ISO 日期。
_RELATIVE = re.compile(r"^(?P<value>\d+)(?P<unit>[dh])$")


def build_parser() -> argparse.ArgumentParser:
    """构造参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="report_ledger.py",
        description="从运行账本汇总 5.7 度量表。",
    )
    parser.add_argument(
        "--home",
        type=Path,
        default=None,
        help="运行产物与账本的根目录（默认 ~/.nightwatch，可用 NW_HOME 覆盖）",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="只看这个时间之后的运行：相对时长（7d / 12h）或 ISO 日期（2026-10-01）",
    )
    parser.add_argument("--kind", default=None, help="只看某类运行：eval / cli")
    parser.add_argument("--repo", default=None, help="只看某个仓库的维护运行")
    parser.add_argument(
        "--json-out", type=Path, default=None, help="把筛选后的汇总写成 JSON 落盘"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """入口。"""
    args = build_parser().parse_args(argv)
    home = nw_home(args.home)
    path = ledger_path(home)
    ledger = read(path)

    print(f"账本：{path}")
    if not ledger.records:
        # 空账本不是错误：首次运行、或还没开 --trace 就是这种状态。
        print("  还没有任何运行记录（用 --trace 跑一次评测或 CLI 就会产生）。")
        return 0

    try:
        since = parse_since(args.since)
    except ValueError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2
    selected = filter_records(ledger.records, since=since, kind=args.kind, repo=args.repo)
    if not selected:
        # 账本里有记录但筛选后为空：报「筛没了」，而不是打一张全 n/a 的表——
        # 后者很容易被误读成「指标掉了」。
        print(f"  筛选后没有匹配的运行（账本共 {len(ledger.records)} 行）。")
        return 0
    report = summarize(selected)
    report["filters"] = {
        "since": None if since is None else since.isoformat(timespec="seconds"),
        "kind": args.kind,
        "repo": args.repo,
    }
    report["ledger"] = {
        "path": str(path),
        # 坏行与未知版本单列：它们意味着账本受损或口径变更，读数前必须先看到。
        "skipped_lines": ledger.skipped_lines,
        "skipped_unknown_version": ledger.skipped_unknown_version,
    }
    print_summary(report)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n汇总已写入 {args.json_out}")

    # 账本受损（坏行）或遇到不认识的 schema 版本时红灯——读数不可信，先修账本。
    return 1 if (ledger.skipped_lines or ledger.skipped_unknown_version) else 0


def parse_since(text: str | None) -> datetime | None:
    """把 ``--since`` 解析成带时区的时刻；None 表示不限。

    Raises:
        ValueError: 写法不认识（形如 ``7d`` / ``12h`` / ISO 日期）。
    """
    if text is None:
        return None
    relative = _RELATIVE.match(text.strip())
    if relative is not None:
        value = int(relative.group("value"))
        delta = timedelta(days=value) if relative.group("unit") == "d" else timedelta(hours=value)
        return datetime.now().astimezone() - delta
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            f"--since 写法不认识：{text!r}；请用相对时长（7d / 12h）或 ISO 日期（2026-10-01）"
        ) from exc
    # 纯日期没有时区，按本地时区解释，否则与 finished_at 的比较会变成 naive vs aware。
    return parsed if parsed.tzinfo else parsed.astimezone()


def filter_records(
    records: Sequence[RunRecord],
    *,
    since: datetime | None = None,
    kind: str | None = None,
    repo: str | None = None,
) -> list[RunRecord]:
    """按时间 / 类别 / 仓库筛选账本行。"""
    selected: list[RunRecord] = []
    for record in records:
        if kind is not None and record.kind != kind:
            continue
        if repo is not None and record.repo != repo:
            continue
        if since is not None:
            finished = _parse_ts(record.finished_at)
            # 时间戳解析不出来的行，在按时间筛选时只能排除，不能混进来。
            if finished is None or finished < since:
                continue
        selected.append(record)
    return selected


def print_summary(report: dict[str, object]) -> None:
    """打印 5.7 度量表。"""
    runs = int(report["runs"])
    counts = report["outcome_counts"]
    print(
        f"  运行 {runs} 次"
        f"（success {counts['success']} / fail {counts['fail']}"
        f" / error {counts['error']} / skeleton {counts['skeleton']}）"
    )
    skipped = report["ledger"]
    if skipped["skipped_lines"] or skipped["skipped_unknown_version"]:
        print(
            f"  ⚠️ 跳过 {skipped['skipped_lines']} 行坏行、"
            f"{skipped['skipped_unknown_version']} 行未知 schema 版本——读数可能不完整"
        )
    print()
    print("  ── 5.7 度量表 ──")

    rate = report["success_rate"]
    rate_text = "n/a（没有可计分的运行）" if rate is None else f"{rate:.0%}"
    print(
        f"  端到端成功率    {rate_text}"
        f"（要求 >= {THRESHOLDS['end_to_end_success_rate']}，分母排除 error）"
    )
    print(
        f"  越权/注入逃逸   {report['escape_count']} 次"
        f"（硬红线，要求 == {THRESHOLDS['escape_count']}）"
    )
    duration = report["duration_seconds"]
    print(
        f"  端到端耗时      mean {_seconds(duration['mean'])}　max {_seconds(duration['max'])}"
        f"（要求 < {THRESHOLDS['max_duration_seconds']}s）"
    )
    print(f"  单任务 token    {_token_line(report)}")
    print(f"  tracing         已启用 {_tracing_runs(report)}／{runs} 次运行")


def _token_line(report: dict) -> str:
    """token 与成本那一行。两类缺失分开说，不要混成一句。"""
    if not report["runs_with_tokens"]:
        return f"n/a（{report['runs']} 次运行都没有模型调用——P0 常态）"
    tokens = f"{report['total_tokens']} tokens（预算上限 {DEFAULT_TOKEN_BUDGET}）"
    if report["total_cost_usd"] is None:
        missing = "、".join(report["missing_price_models"]) or "未知模型"
        return f"{tokens}，费用 n/a（缺价格表：{missing}；在 prices.json 里补齐）"
    return f"{tokens}，费用 ${report['total_cost_usd']:.4f}"


def _tracing_runs(report: dict) -> int:
    """已确认开启 tracing 的运行数。

    这是「P0 只能验证接线」这条事实的量化体现：次数大于 0 说明开关与 key 都配好了，
    但 **run 数为 0 或 trace 内容为空** 仍属正常——P0 根本没有模型调用。
    """
    return int(report.get("tracing_enabled_runs", 0))


def _parse_ts(text: str) -> datetime | None:
    """解析账本里的 ISO8601 时间戳；解析不了返回 None。"""
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _seconds(value: float | None) -> str:
    """把秒数格式化成人话。"""
    return "n/a" if value is None else f"{value:.2f}s"


if __name__ == "__main__":
    sys.exit(main())
