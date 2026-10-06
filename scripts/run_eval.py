#!/usr/bin/env python
"""跑 seed 评测集，产出对齐 5.7 度量表的基线数字。

对应 doc/开发路线图.md 第 0 阶段的「建 seed 评测集」：用一组固定 golden issue +
固定仓库快照，把「感觉能用」换成可量化的数字，任一指标跌破阈值即视为回归。

用法：

    # 离线（注入 e2b SDK 替身，不联网、不花钱）——日常与 CI 走这条
    python scripts/run_eval.py --backend offline --solver null
    python scripts/run_eval.py --backend offline --solver reference \\
        --json-out evals/reports/reference.json

    # 真链路（联网、计费，需 E2B_API_KEY）——手动跑，验证离线数字在生产链路上仍成立
    python scripts/run_eval.py --backend e2b --solver reference --json-out evals/reports/e2b.json

**两个 solver 的口径别搞混**（见 doc/开发路线图.md 5.7）：
- ``null``：什么都不做。成功率**理应为 0**——这是「用例改前必红」的机器化证据，
  也是 Agent 缺席时的真实基线。
- ``reference``：套用用例自带的 golden 补丁。成功率**理应为 1**——这是 harness
  能识别成功、且不误报逃逸的证据。它不是项目能力指标，别当成成绩。

**退出码只反映 harness 健康度**（有用例记 ``error`` 时非 0），不反映阈值是否达标：
``null`` 基线的成功率本就该是 0，若拿它当退出码，这条基线永远"失败"。阈值判定随报告
输出（``thresholds`` 字段与末尾表格），供 CI 或人判读。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path

# 本脚本不在包内，而它要 import 生产代码与评测库。把 src/ 提前放进 sys.path，
# 这样无需先 `pip install -e .` 也能跑（下面的 E402 因此无法避免，属刻意）。
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (_REPO_ROOT / "src", _REPO_ROOT / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from dotenv import load_dotenv  # noqa: E402

from nw_agent.backends import (  # noqa: E402
    E2BSandboxBackend,
    SandboxBackend,
    SandboxConfig,
    create_backend,
)
from nw_agent.evals import (  # noqa: E402
    BackendFactory,
    NullSolver,
    ReferenceSolver,
    Solver,
    iter_case_dirs,
    load_case,
    run_suite,
)

# 与 bench_sandbox_cold_start.py 一致：允许把 E2B_API_KEY 放在仓库根的 .env 里
# （该文件已被 .gitignore 覆盖）。必须在这里就加载——SDK 是在 create() 时才去读
# os.environ 的，晚于本行就会拿不到。
load_dotenv(override=True)

DEFAULT_CASES_DIR = _REPO_ROOT / "evals" / "cases"
# 预装了 git + conda + pytest 的模板（见 doc/沙箱基准.md）。验收命令用的是
# `python3 -m pytest`，缺它会让每条用例都因为「验收跑不起来」而判失败。
DEFAULT_TEMPLATE = "nightwatch-base"
_SOLVERS: dict[str, type[Solver]] = {
    "null": NullSolver,
    "reference": ReferenceSolver,
}
_BACKENDS = ("offline", "e2b")


def make_e2b_factory() -> BackendFactory:
    """真链路工厂：每次调用建一个真实云端沙箱。"""

    def factory(config: SandboxConfig) -> SandboxBackend:
        return create_backend("e2b", config)

    return factory


def make_offline_factory(root: Path) -> BackendFactory:
    """离链路工厂：注入 e2b SDK 的**测试替身**，不联网。

    ⚠️ 这不是第二个后端实现（产品说明 3.3 已删掉「本地假沙箱」，`create_backend`
    造不出它）。替身只存在于 ``tests/``，这里通过 ``E2BSandboxBackend._attach``
    这个既有注入点接上，走的是**生产后端同一条代码路径**，只是把远端换成了宿主机上的
    临时目录。代价是它不翻译沙箱绝对路径，故每条用例的 ``repo_path`` /
    ``acceptance_path`` 都要落在临时目录里——这正是 runner 从 config 取路径、
    而 case 数据里不许出现 ``/workspace`` 的原因。

    每条用例一个独立根目录：用例之间不该共享任何文件状态。
    """

    def factory(config: SandboxConfig) -> SandboxBackend:
        from e2b_double import SandboxDouble

        case_root = root / str(config.metadata.get("eval_case", "case"))
        case_root.mkdir(parents=True, exist_ok=True)
        workspace = case_root / "workspace"
        local = replace(
            config,
            repo_path=str(workspace / "repo"),
            acceptance_path=str(workspace / "acceptance"),
        )
        return E2BSandboxBackend._attach(local, SandboxDouble(case_root))

    return factory


def build_parser() -> argparse.ArgumentParser:
    """构造参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="run_eval.py",
        description="跑 seed 评测集，产出 5.7 度量表的基线数字。",
    )
    parser.add_argument(
        "--cases", type=Path, default=DEFAULT_CASES_DIR, help="用例根目录（默认 evals/cases）"
    )
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="只跑指定用例 id，可重复；默认跑全部",
    )
    parser.add_argument(
        "--solver", choices=sorted(_SOLVERS), default="null", help="求解器（默认 null）"
    )
    parser.add_argument(
        "--backend",
        choices=list(_BACKENDS),
        default="offline",
        help="offline 走 SDK 替身不花钱（默认）；e2b 建真实云端沙箱并计费",
    )
    parser.add_argument(
        "--json-out", type=Path, default=None, help="把报告写成 JSON 落盘，供回归对比"
    )
    parser.add_argument(
        "--template",
        default=DEFAULT_TEMPLATE,
        help=f"E2B 模板名（仅 --backend e2b 用；默认 {DEFAULT_TEMPLATE}，"
        "它是唯一预装了 pytest 的模板，用 E2B 默认模板会让验收因缺 pytest 而全红）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """入口。"""
    args = build_parser().parse_args(argv)
    # mypy 看不到 argparse 的 choices 约束，故这里显式断言一次类型。
    solver_factory = _SOLVERS[args.solver]

    cases_dir = Path(args.cases)
    if not cases_dir.is_dir():
        print(f"❌ 用例目录不存在：{cases_dir}", file=sys.stderr)
        return 2
    cases = [load_case(path) for path in iter_case_dirs(cases_dir)]
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case.case_id in wanted]
        missing = wanted - {case.case_id for case in cases}
        if missing:
            print(f"❌ 找不到用例：{sorted(missing)}", file=sys.stderr)
            return 2
    if not cases:
        print(f"❌ {cases_dir} 下没有可用例", file=sys.stderr)
        return 2

    sandbox_count = len(cases) if args.backend == "e2b" else 0
    print(f"求解器：{args.solver}　后端：{args.backend}　用例：{len(cases)} 条")
    if sandbox_count:
        print(f"⚠️ 将创建 {sandbox_count} 个真实云端沙箱并计费。")
    else:
        print("离线模式：注入 SDK 替身，不联网、不计费。")
    print()

    with tempfile.TemporaryDirectory(prefix="nw-eval-") as tmp:
        factory = make_e2b_factory() if args.backend == "e2b" else make_offline_factory(Path(tmp))
        report = run_suite(
            cases,
            solver_factory(),
            factory,
            backend_kind=args.backend,
            # 只有真链路关心模板；离线替身不认它，传 None 免得误导。
            config=SandboxConfig(template=args.template) if args.backend == "e2b" else None,
        )

    report["generated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print_report(report)
    if args.json_out is not None:
        print(f"\n报告已写入 {args.json_out}")

    summary = report["summary"]
    assert isinstance(summary, dict)  # 自家 run_suite 的形状，断言即是文档
    errors = summary["error"]
    if errors:
        print(f"\n❌ {errors} 条用例记 error（harness 故障，非 solver 失败），见报告。")
        return 1
    return 0


def print_report(report: dict[str, object]) -> None:
    """打印逐条明细与汇总表。"""
    cases = report["cases"]
    summary = report["summary"]
    assert isinstance(cases, list) and isinstance(summary, dict)

    print("  ── 逐条明细 ──")
    header = f"  {'用例':<20}{'结果':<9}{'证据':<6}{'L3':<5}{'逃逸':<6}{'改动':<8}{'耗时(s)':>9}"
    print(header)
    for case in cases:
        assert isinstance(case, dict)
        mark = {"success": "✅", "fail": "❌", "error": "💥"}.get(str(case["outcome"]), "?")
        print(
            f"  {case['case_id']:<20}{mark + str(case['outcome']):<9}"
            f"{case['evidence_level']:<6}{_mark(bool(case['l3_passed'])):<5}"
            f"{_mark(not case['escaped']):<6}{len(case['changed_paths']):<8}"
            f"{case['duration_seconds']:>9.2f}"
        )
        if case["violations"] or case["secret_paths"]:
            print(f"      越界：{case['violations']}　凭据形态：{case['secret_paths']}")
        if case["rejected_uploads"]:
            print(f"      上传被拒（未进沙箱）：{case['rejected_uploads']}")
        if case["error"]:
            print(f"      error：{case['error']}")

    print()
    print("  ── 汇总（对照 5.7 阈值）──")
    rate = summary["end_to_end_success_rate"]
    rate_text = "n/a（全部 error）" if rate is None else f"{rate:.0%}"
    print(
        f"  成功率        {summary['success']}/{int(summary['total']) - int(summary['error'])}"
        f"　= {rate_text}"
    )
    print(f"  逃逸次数      {summary['escape_count']}（硬红线，须恒为 0）")
    print(
        f"  端到端耗时    mean {_seconds(summary['mean_duration_seconds'])}"
        f"　max {_seconds(summary['max_duration_seconds'])}"
    )
    print(f"  token 成本    {summary['total_tokens']}（P0 无模型调用，恒为 null）")
    if not summary["harness_ok"]:
        print(f"  ⚠️ harness 不健康：{summary['error']} 条用例记 error")
    print()
    thresholds = report["thresholds"]
    assert isinstance(thresholds, dict)
    for name, verdict in thresholds.items():
        assert isinstance(verdict, dict)
        print(
            f"  {_mark(bool(verdict['passed']))} {name:<26} = {verdict['value']}"
            f"（要求 {verdict['required']}）"
        )


def _mark(ok: bool) -> str:
    """✅ / ❌。"""
    return "✅" if ok else "❌"


def _seconds(value: float | None) -> str:
    """把秒数格式化成可读文本。"""
    return "n/a" if value is None else f"{value:.2f}s"


if __name__ == "__main__":
    sys.exit(main())
