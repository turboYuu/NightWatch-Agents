"""``nw-agent`` 命令行入口。

P0 阶段只定义**参数契约**（与 doc/NightWatch产品说明.md 第 5 章 P0 一致），
真正的工作流在后续阶段接入。当前运行只做参数解析与回显，
目的是先把 CLI 骨架和 `nw-agent --help` 跑通。

参数契约（与文档严格对齐）：
    --repo          仓库 owner/name 或本地路径（必填）
    --issue         GitHub Issue 编号（与 --task 二选一）
    --task          自然语言任务描述（与 --issue 二选一）
    --dry-run       只跑非沙箱流程：不建沙箱、不调远端
    --review        人工审核策略：always 强制人审（默认）/ auto 仅白名单任务免审
                    / never 全免（仅本地调试）
    --trace         开启 tracing / 结构化日志
    --budget-tokens 单任务 token 预算上限（预算熔断用）
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

# 默认的 token 预算上限。夜间批量跑时，单任务失控烧钱是最贵的错误，
# 因此给一个偏保守的默认值，可通过 --budget-tokens 覆盖。
DEFAULT_TOKEN_BUDGET = 200_000


@dataclass(frozen=True)
class CliOptions:
    """一次运行的全部输入。

    字段设计为与后续 ``MaintenanceState`` 的初始值对应，
    这样从 CLI 到状态图的映射是直接的、无需二次转换。
    """

    repo: str | None        # owner/name 或本地路径
    issue: int | None       # GitHub Issue 编号
    task: str | None        # 自然语言任务描述
    dry_run: bool           # 是否只跑非沙箱流程
    review: str             # 人工审核策略：always / auto / never
    trace: bool             # 是否开启 tracing
    budget_tokens: int      # 单任务 token 预算上限


def build_parser() -> argparse.ArgumentParser:
    """构造参数解析器。

    单独抽成函数，便于测试直接复用、以及未来生成补全脚本。
    """
    parser = argparse.ArgumentParser(
        prog="nw-agent",
        description="夜间维护者：把技术债 Issue 自动变成待人工确认的 PR 草案。",
    )

    # --- 任务来源：--repo 必填；--issue 与 --task 二选一 ---
    parser.add_argument("--repo", required=True, help="仓库 owner/name 或本地路径")
    parser.add_argument("--issue", type=int, help="GitHub Issue 编号")
    parser.add_argument("--task", help='自然语言任务描述，如 "给 foo() 补 docstring"')

    # --- 运行模式开关 ---
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只跑非沙箱流程（生成方案、展示 diff 预览），不建沙箱、不调远端",
    )
    parser.add_argument(
        "--review",
        choices=["always", "auto", "never"],
        default="always",
        help="人工审核策略：always 强制人审（默认）/ auto 白名单免审 / never 全免",
    )
    parser.add_argument("--trace", action="store_true", help="开启 tracing 或结构化 JSON 日志")
    parser.add_argument(
        "--budget-tokens",
        type=int,
        default=DEFAULT_TOKEN_BUDGET,
        help=f"单任务 token 预算上限，超出即熔断（默认 {DEFAULT_TOKEN_BUDGET}）",
    )

    return parser


def parse_args(argv: list[str] | None = None) -> CliOptions:
    """解析并校验命令行参数，返回强类型的 :class:`CliOptions`。

    Args:
        argv: 参数列表；为 None 时读取 ``sys.argv``（正常 CLI 场景）。

    Raises:
        SystemExit: 参数非法时由 argparse 触发（打印错误并以非零码退出）。
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # 必须给出任务来源，否则没有可执行的目标。
    if args.issue is None and not args.task:
        parser.error("必须提供 --issue 或 --task 之一")

    # 二者互斥：同时给出会让“以谁为准”变得含糊，直接拒绝。
    if args.issue is not None and args.task:
        parser.error("--issue 与 --task 只能二选一")

    return CliOptions(
        repo=args.repo,
        issue=args.issue,
        task=args.task,
        dry_run=args.dry_run,
        review=args.review,
        trace=args.trace,
        budget_tokens=args.budget_tokens,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI 主入口。

    P0 阶段仅回显解析结果；工作流留待后续阶段接入（届时在这里构造
    ``MaintenanceState`` 并驱动 LangGraph 状态图）。

    Returns:
        进程退出码；0 表示成功。
    """
    options = parse_args(argv)

    # TODO(P1+): 用 options 初始化 MaintenanceState，编译并 invoke LangGraph 状态图。
    print("nw-agent（P0 骨架）已解析参数：")
    print(f"  repo          = {options.repo}")
    print(f"  issue         = {options.issue}")
    print(f"  task          = {options.task}")
    print(f"  dry_run       = {options.dry_run}")
    print(f"  review        = {options.review}")
    print(f"  trace         = {options.trace}")
    print(f"  budget_tokens = {options.budget_tokens}")
    return 0


if __name__ == "__main__":
    sys.exit(main())