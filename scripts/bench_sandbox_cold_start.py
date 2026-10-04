#!/usr/bin/env python
"""E2B 沙箱冷启动基准：分段计时，回答「预热值不值得」。

对应 doc/开发路线图.md 第 0 阶段的「沙箱预热 spike / E2B 冷启动基准」与产品说明
3.3 的「冷启动基准：必须实测『创建沙箱 → 能跑通 pytest』的耗时」。

**测的是生产链路**：走 `E2BSandboxBackend`（`create` / `upload_repo` / `execute` /
`kill`），不是裸 e2b SDK，这样量到的数字就是上层真正会遇到的数字。

**为什么输入用合成 mini repo**：基准要可重复、要能在任何模板上跑，就不能依赖外部
仓库。这里上传三个文件（`ok.py` + `tests/test_ok.py` + `conftest.py`）并在其上跑
`python3 -m pytest`——与真实任务同一条命令路径，只是规模最小。

用法：

    # 默认模板（对照基线）
    python scripts/bench_sandbox_cold_start.py --template default --repeats 3

    # 预热模板（先用 build_sandbox_template.py build 建好）
    python scripts/bench_sandbox_cold_start.py --template nightwatch-base --repeats 3 \\
        --json-out bench-prewarmed.json

⚠️ 每次运行会**创建真实云端沙箱并计费**（探测 1 个 + 每次重复 1 个）。脚本会在开始
前打印将要创建的沙箱数量。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# 本脚本不在包内，而它要 import 生产后端。把仓库的 src/ 提前放进 sys.path，
# 这样无需先 `pip install -e .` 也能跑（下面的 E402 因此无法避免，属刻意）。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import e2b  # noqa: E402

from nw_agent.backends import E2BSandboxBackend, SandboxConfig  # noqa: E402

# 上传到沙箱的合成 mini repo。conftest.py 把仓库根放进 sys.path，使
# `from ok import add` 在 pytest 的任意 import 模式下都成立——比依赖 pytest
# 的 rootdir 推断可靠。
SAMPLE_REPO: list[tuple[str, bytes]] = [
    ("ok.py", b"def add(a: int, b: int) -> int:\n    return a + b\n"),
    (
        "tests/test_ok.py",
        b"from ok import add\n\n\ndef test_add() -> None:\n    assert add(1, 2) == 3\n",
    ),
    (
        "conftest.py",
        b"import pathlib\nimport sys\n\n"
        b"sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))\n",
    ),
]

# 模板内容探测。pytest 与 pip 都经 `python3 -m` 调用：这样看到的是「该解释器能否
# import 到它」，而不是「PATH 上有没有这个脚本」——后者会漏掉装错解释器的情况。
PROBE_COMMANDS: tuple[tuple[str, str], ...] = (
    ("git", "git --version"),
    ("python3", "python3 --version"),
    ("pytest", "python3 -m pytest --version"),
    ("conda", "conda --version"),
)

# 装依赖与跑测试的命令。两条都用 `python3 -m`，且 pip 带 `||` 回退到 --user——
# 沙箱内是非 root 用户时，装系统 site-packages 会因权限失败。
DEPS_COMMAND = "python3 -m pip install -q pytest || python3 -m pip install -q --user pytest"
TEST_COMMAND = "python3 -m pytest -q"

# 分段名（顺序即执行顺序）。"probe" 不在这里：它单独跑一次，不属于每次重复。
STAGES = ("create", "upload", "deps", "test", "kill")


def probe_template(template: str | None, *, timeout_seconds: int) -> dict[str, dict[str, str]]:
    """用裸 SDK 建一个沙箱，探测模板里有哪些工具，然后销毁。

    刻意**不**经 `E2BSandboxBackend.create`：后端的创建期准备要求模板里必须有 git，
    缺了就直接抛错——而「默认模板到底有没有 git」恰恰是本基准要回答的问题之一，
    所以探测必须比后端更宽容。

    Returns:
        ``{命令名: {"status": "ok"|"missing", "output": 首行输出或错误信息}}``
    """
    sandbox = e2b.Sandbox.create(template=template, timeout=timeout_seconds)
    try:
        findings: dict[str, dict[str, str]] = {}
        for name, command in PROBE_COMMANDS:
            try:
                result = sandbox.commands.run(command, timeout=60)
            except e2b.CommandExitException as exc:
                # 非零退出 = 该工具缺失或不可用，如实记录而不是抛。
                detail = (exc.stderr or exc.stdout or "").strip().splitlines()
                findings[name] = {
                    "status": "missing",
                    "output": detail[0] if detail else f"退出码 {exc.exit_code}",
                }
            except e2b.SandboxException as exc:
                findings[name] = {"status": "missing", "output": f"{type(exc).__name__}: {exc}"}
            else:
                findings[name] = {"status": "ok", "output": result.stdout.strip().splitlines()[0]}
        return findings
    finally:
        try:
            sandbox.kill()
        except Exception:  # noqa: BLE001 - 收尾不能掩盖探测结果
            print("  （探测沙箱 kill 失败，将由 on_timeout 自动回收）", file=sys.stderr)


def run_once(template: str | None, *, timeout_seconds: int, command_timeout: int) -> dict[str, Any]:
    """跑一次完整链路，返回各阶段耗时（秒）。

    Raises:
        RuntimeError: 后端创建失败（最常见的原因是模板未预装 git）。
    """
    timings: dict[str, float] = {}

    started = time.perf_counter()
    backend = E2BSandboxBackend.create(
        SandboxConfig(
            template=template,
            timeout_seconds=timeout_seconds,
            command_timeout_seconds=command_timeout,
        )
    )
    timings["create"] = time.perf_counter() - started

    try:
        started = time.perf_counter()
        responses = backend.upload_repo(SAMPLE_REPO)
        timings["upload"] = time.perf_counter() - started
        failed = [r for r in responses if r.error]
        if failed:
            raise RuntimeError(f"上传失败：{failed[0].error}")

        started = time.perf_counter()
        backend.execute(DEPS_COMMAND)
        timings["deps"] = time.perf_counter() - started

        started = time.perf_counter()
        result = backend.execute(TEST_COMMAND)
        timings["test"] = time.perf_counter() - started
        if result.exit_code != 0:
            # 测试没跑通就不算有效样本——但别说成「基准失败」，先把输出带出去。
            raise RuntimeError(f"pytest 未通过（退出码 {result.exit_code}）：{result.output[:300]}")
    finally:
        started = time.perf_counter()
        backend.kill()
        timings["kill"] = time.perf_counter() - started

    timings["total"] = sum(timings[stage] for stage in STAGES)
    return timings


def summarize(runs: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """按阶段汇总 min / median / max。"""
    summary: dict[str, dict[str, float]] = {}
    for stage in (*STAGES, "total"):
        values = [run[stage] for run in runs]
        summary[stage] = {
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
        }
    return summary


def print_probe(findings: dict[str, dict[str, str]]) -> None:
    """打印模板探测结果。"""
    print("  ── 模板内容探测 ──")
    for name, finding in findings.items():
        mark = "✅" if finding["status"] == "ok" else "❌"
        print(f"  {mark} {name:8} {finding['output']}")


def print_runs(runs: list[dict[str, float]]) -> None:
    """打印逐次明细与汇总表。"""
    print("  ── 逐次明细（秒）──")
    header = "  " + "".join(f"{stage:>10}" for stage in (*STAGES, "total"))
    print(header)
    for index, run in enumerate(runs, start=1):
        cells = "".join(f"{run[stage]:>10.1f}" for stage in (*STAGES, "total"))
        print(f"  #{index}{cells}")

    summary = summarize(runs)
    print("  ── 汇总（秒）──")
    print("  " + "".join(f"{stage:>10}" for stage in ("stage", "min", "median", "max")))
    for stage in (*STAGES, "total"):
        row = summary[stage]
        print(f"  {stage:<10}{row['min']:>10.1f}{row['median']:>10.1f}{row['max']:>10.1f}")


def build_parser() -> argparse.ArgumentParser:
    """构造参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="bench_sandbox_cold_start.py",
        description="E2B 沙箱冷启动分段基准（会创建真实沙箱并计费）。",
    )
    parser.add_argument(
        "--template",
        default="default",
        help="模板名，或 'default' 表示 E2B 默认 base 模板（默认：default）",
    )
    parser.add_argument("--repeats", type=int, default=3, help="重复次数（默认 3）")
    parser.add_argument(
        "--json-out", type=Path, default=None, help="把结果写成 JSON 落盘，供回归对比"
    )
    parser.add_argument(
        "--timeout-seconds", type=int, default=900, help="沙箱存活时长（默认 900 秒）"
    )
    parser.add_argument(
        "--command-timeout", type=int, default=600, help="单条命令超时（默认 600 秒）"
    )
    parser.add_argument("--skip-probe", action="store_true", help="跳过模板内容探测")
    return parser


def main(argv: list[str] | None = None) -> int:
    """入口。"""
    args = build_parser().parse_args(argv)
    template = None if args.template == "default" else args.template
    template_label = args.template

    sandbox_count = args.repeats + (0 if args.skip_probe else 1)
    print(f"模板：{template_label}（{'E2B 默认 base' if template is None else '自定义模板'}）")
    print(f"将创建 {sandbox_count} 个真实云端沙箱并计费（探测 1 + 重复 {args.repeats}）。")
    print()

    report: dict[str, Any] = {
        "template": template_label,
        "repeats": args.repeats,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }

    if not args.skip_probe:
        findings = probe_template(template, timeout_seconds=args.timeout_seconds)
        print_probe(findings)
        report["probe"] = findings
        print()

    runs: list[dict[str, float]] = []
    for index in range(1, args.repeats + 1):
        print(f"  ── 第 {index}/{args.repeats} 次 ──", flush=True)
        try:
            timings = run_once(
                template,
                timeout_seconds=args.timeout_seconds,
                command_timeout=args.command_timeout,
            )
        except (RuntimeError, e2b.SandboxException) as exc:
            # 最常见的两种原因都给一句可直接照做的结论，而不是甩堆栈。
            print(f"\n❌ 第 {index} 次运行失败：{exc}", file=sys.stderr)
            print(
                "   若是「沙箱环境准备失败」，说明该模板未预装 git——"
                "用 scripts/build_sandbox_template.py build 建 nightwatch-base 后改用 "
                "--template nightwatch-base。",
                file=sys.stderr,
            )
            report["error"] = str(exc)
            break
        runs.append(timings)

    if runs:
        print()
        print_runs(runs)
        report["runs"] = runs
        report["summary"] = summarize(runs)

    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n结果已写入 {args.json_out}")

    return 0 if runs else 1


if __name__ == "__main__":
    sys.exit(main())
