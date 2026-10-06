"""``nw-agent`` 命令行入口。

P1 第一周起，``--task`` 会**真跑**：读本地仓库 → 建 E2B 沙箱 → 上传文件子集 →
装配主代理（`create_deep_agent` + 一个 ``fix`` 子代理）→ 跑一次任务 → 打印宿主导出的补丁。

参数契约（与 doc/NightWatch产品说明.md 第 5 章一致）：
    --repo          仓库 owner/name 或本地路径（必填；**当前只支持本地目录**）
    --issue         GitHub Issue 编号（与 --task 二选一；**待 P5 才能用**）
    --task          自然语言任务描述（与 --issue 二选一）
    --dry-run       只做校验与预演：不建沙箱、不调远端、**不调模型**
    --review        人工审核策略：always 强制人审（默认）/ auto 仅白名单任务免审
                    / never 全免（仅本地调试）——**门控本身待 P4**
    --trace         开启 LangSmith tracing 并落运行产物与账本
    --budget-tokens 单任务 token 预算上限（预算熔断用）

⚠️ **本轮没有工具权限收紧（已知风险，不是疏忽）**：模型手里的 ``edit_file`` /
``write_file`` 的路径由它自己给，能写沙箱内任意位置；``execute`` 能跑任意 shell 命令。
缓解只有三条：沙箱一次性、凭据不进沙箱、宿主与沙箱之间只流出 diff。
写白名单与命令白名单是**下一周的第一优先级**，在此之前不要把本命令当作已安全的工具。

需要哪些凭据（放仓库根的 ``.env`` 或环境变量里，``.env`` 已被 gitignore）：

    E2B_API_KEY        建远端沙箱用（``--dry-run`` 不需要）
    NW_MODEL_MAIN      主代理模型，``provider:model`` 形式
    NW_MODEL_FIX       fix 子代理模型，同上
    <provider> 的 key  如 ``ANTHROPIC_API_KEY``——由 ``init_chat_model`` 自己读

``--dry-run`` **不校验 key 是否有效**：实测缺 key 时 ``init_chat_model`` 在构造期不报错，
报错推迟到首次调用。所以 dry-run 能挡住的是「配置串写错 / 仓库不可读 / 文件太多」，
挡不住「key 是假的」。

``--help`` 的输出是怎么来的（实现说明）：
    本模块**没有任何一行代码负责打印帮助页**。`nw-agent --help` 的整页内容
    全部由 argparse 依据本文件的声明自动拼装，对应关系如下：

    - 用法行开头的 ``nw-agent`` 与顶部那句描述 → :func:`build_parser` 里
      ``ArgumentParser(prog=..., description=...)`` 的两个参数；
    - 用法行里的每个片段（如 ``[--review {always,auto,never}]``）→ 每个
      ``add_argument`` 的声明：``choices`` 决定花括号里的枚举值，
      ``required=True`` 决定该项**不被**方括号包裹；
    - 说明里的 ``REPO`` / ``BUDGET_TOKENS`` 等大写占位符 → argparse 由选项名
      推导（去 ``--``、横线转下划线、转大写），无需手写 ``metavar``；
    - 每项右侧的说明文字 → 该 ``add_argument`` 的 ``help=`` 参数；
    - ``-h/--help`` 本身 → argparse 默认 ``add_help=True`` 自动注入，无需声明。

    所以改帮助页 = 改这里各 ``add_argument`` 的 ``help`` / ``choices``，
    而不是去改什么输出代码。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from nw_agent.agents import (
    AgentConfigError,
    AgentModels,
    BudgetExceeded,
    build_main_agent,
    resolve_agent_models,
    run_task,
)
from nw_agent.backends import SandboxConfig, create_backend, read_local_repo
from nw_agent.observability import RunContext, TracingStatus, configure_tracing

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
    # 帮助页的「外壳」由这两个参数决定：prog 是用法行开头的命令名（写死为 nw-agent，
    # 这样即便经 `python -m nw_agent` 调用，用法行也显示同一个名字）；
    # description 是用法行下方那段总述。其余内容由下面每个 add_argument 拼出。
    parser = argparse.ArgumentParser(
        prog="nw-agent",
        description="夜间维护者：把技术债 Issue 自动变成待人工确认的 PR 草案。",
    )

    # --- 任务来源：--repo 必填；--issue 与 --task 二选一 ---
    # 帮助页里的 REPO / ISSUE / TASK 是 argparse 从选项名推导的占位符，不必手写 metavar。
    # required=True 使得 --repo 在用法行中不带方括号，一眼可辨是必填项。
    parser.add_argument("--repo", required=True, help="仓库 owner/name 或本地路径")
    parser.add_argument("--issue", type=int, help="GitHub Issue 编号")
    parser.add_argument("--task", help='自然语言任务描述，如 "给 foo() 补 docstring"')

    # --- 运行模式开关 ---
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验配置与将要上传的文件：不建沙箱、不调远端、不调模型",
    )
    parser.add_argument(
        "--review",
        choices=["always", "auto", "never"],
        default="always",
        help="人工审核策略：always 强制人审（默认）/ auto 白名单免审 / never 全免",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="开启 LangSmith tracing，并把运行产物与账本落盘",
    )
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

    # 这一行同时负责 --help 与参数校验失败两条退出路径，二者都以抛 SystemExit 终止进程：
    #   --help     → argparse 打印帮助页后 sys.exit(0)，属正常退出；因此不会走到本函数
    #                或 main() 的后续任何一行；
    #   参数非法   → parser.error() 往 stderr 打印用法与错误信息后 sys.exit(2)。
    # 测试正是据此用 pytest.raises(SystemExit) 断言，无需捕获输出流。
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

    ``--task`` 走真链路（读本地仓库 → 建沙箱 → 装配主代理 → 跑一次 → 打印补丁）；
    ``--dry-run`` 只做校验与预演。``--issue`` / 非本地 ``--repo`` 显式拒绝——静默回显会让人
    把「什么都没发生」误读成「跑过了没问题」，账本里也留不下痕迹。

    Returns:
        进程退出码：0 表示跑出了补丁（或 dry-run 通过）；1 表示跑完但没有改动或运行失败；
        2 表示用法/配置错误（与 argparse 的错误码一致）。
    """
    # 顺序契约（见 doc/可观测性.md）：load_dotenv → configure_tracing → 任何被 trace 的调用。
    # 为什么 CLI 也开始读 .env：真跑一次要用到 E2B_API_KEY 与模型 provider 的 key，而
    # `src/` 此前刻意零 dotenv（凭据由调用方注入）。CLI 自己就是那个调用方了，故在此加载，
    # 与 scripts/*.py 的 load_dotenv(override=True) 范式保持一致。
    load_dotenv(override=True)

    options = parse_args(argv)

    # 抢在任何可能被 trace 的调用之前配置：langsmith 的 get_env_var 带 lru_cache，
    # 晚设的环境变量可能被缓存屏蔽（见 observability/tracing.py 的顺序契约）。
    tracing = configure_tracing(options.trace)

    print("nw-agent：")
    print(f"  repo          = {options.repo}")
    print(f"  task          = {options.task}")
    print(f"  dry_run       = {options.dry_run}")
    print(f"  review        = {options.review}（人工审核门控待 P4）")
    print(f"  trace         = {options.trace}（tracing {tracing.describe()}）")
    print(f"  budget_tokens = {options.budget_tokens}")
    print("  注意：本轮尚无 write/execute 白名单，模型可写沙箱内任意路径。")
    print()

    if options.issue is not None:
        print(
            "❌ 未实现：--issue 需要 P5 的 GitHub 集成；当前请用 --task 直接给任务描述。",
            file=sys.stderr,
        )
        return 2

    # ``--repo`` 在 argparse 里是 required=True，运行时不会是 None；类型上的可空来自
    # CliOptions 与 MaintenanceState 对齐的设计，故这里显式收窄而不是断言。
    repo_raw = options.repo
    repo_path = Path(repo_raw) if repo_raw is not None else None
    if repo_path is None or not repo_path.is_dir():
        print(
            f"❌ --repo 目前只支持本地目录（GitHub owner/name 待 P5）。收到：{repo_raw!r}",
            file=sys.stderr,
        )
        return 2

    try:
        models = resolve_agent_models()
    except AgentConfigError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2

    if options.dry_run:
        return _dry_run(repo_path, models)
    return _run(repo_path, options, models, tracing)


def _dry_run(repo_path: Path, models: AgentModels) -> int:
    """预演：校验配置、算出将上传的文件，**不建沙箱、不调远端、不调模型**（产品说明 3.3）。

    它的价值是在花钱之前把「配置写错、仓库不可读、文件超上限」这类错误全暴露出来。
    ⚠️ 但它**挡不住假 key**：缺 key 时 ``init_chat_model`` 在构造期不报错（实测），
    报错推迟到首次调用，所以这里如实说明、不假装校验过。
    """
    local = read_local_repo(repo_path)
    main_model, fix_model = models.specs()
    print(f"  主模型        = {main_model}")
    print(f"  fix 子代理    = {fix_model}")
    print(f"  {local.summary()}")
    if local.rejected:
        print(f"  刻意不上传（凭据类）：{_preview(local.rejected)}")
    if local.skipped:
        print(f"  ⚠️ 超上限未上传（任务可能因缺文件而失败）：{_preview(local.skipped)}")
    print()
    print("dry-run：未建沙箱、未调远端、未调模型。")
    print("  注意：dry-run 不校验 API key 是否有效——缺 key 只会在首次调用时才报错。")
    return 0


def _run(
    repo_path: Path,
    options: CliOptions,
    models: AgentModels,
    tracing: TracingStatus,
) -> int:
    """真链路：读仓库 → 建沙箱 → 上传 → 装配 → 跑一次 → 打印补丁。"""
    if not (os.environ.get("E2B_API_KEY") or "").strip():
        print(
            "❌ 缺少 E2B_API_KEY（放仓库根的 .env 或环境变量里）；"
            "只想预演请加 --dry-run。",
            file=sys.stderr,
        )
        return 2

    local = read_local_repo(repo_path)
    print(f"  {local.summary()}")
    if local.skipped:
        # 静默截断会让人以为上传是完整的，任务失败时又会去怀疑模型——必须说出来。
        print(f"  ⚠️ 超上限未上传（任务可能因缺文件而失败）：{_preview(local.skipped)}")

    main_model, fix_model = models.specs()
    context = RunContext(
        kind="cli",
        repo=str(repo_path),
        backend_kind="e2b",
        tracing=tracing,
        # 没给 --home 时落在 ~/.nightwatch；NW_HOME 可覆盖。
    )
    context.snapshot(
        "run-request",
        {
            "repo": str(repo_path),
            "task": options.task,
            "models": {"main": main_model, "fix": fix_model},
            "budget_tokens": options.budget_tokens,
            "review": options.review,
            "uploaded_files": len(local.files),
            "rejected_upload_paths": local.rejected,
            "skipped_upload_paths": local.skipped,
        },
    )

    with create_backend("e2b", SandboxConfig()) as backend:
        try:
            backend.upload_repo(local.files)
            agent = build_main_agent(backend, models)
            result = run_task(
                agent,
                backend,
                options.task or "",
                model_label=main_model,
                recorder=context,
                # 预算 0 或负数视为「不限」，与 e2b 后端对 timeout<=0 的处理同构。
                budget_tokens=options.budget_tokens if options.budget_tokens > 0 else None,
            )
        except BudgetExceeded as exc:
            context.finish("error", error=str(exc))
            print(f"\n❌ {exc}", file=sys.stderr)
            print(f"   运行产物：{context.run_dir}")
            return 1
        except Exception as exc:  # noqa: BLE001 - 真链路的失败要记进账本再退出
            context.finish("error", error=f"{type(exc).__name__}: {exc}")
            print(f"\n❌ 运行失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            print(f"   运行产物：{context.run_dir}")
            return 1

    # 有改动才算跑成：空补丁意味着任务没被完成（不是错误，但也不能报 success）。
    outcome = "success" if result.patch else "fail"
    context.finish(outcome)

    print()
    if result.patch:
        print("── 补丁（宿主导出）──")
        print(result.patch)
    else:
        print("⚠️ 模型没有产生任何文件改动（补丁为空）。")
    if result.final_text:
        print("── 主代理说明 ──")
        print(result.final_text)
    usage = f"token {result.total_tokens}（{result.n_llm_calls} 次模型调用）"
    if result.n_calls_without_usage:
        usage += f"；另有 {result.n_calls_without_usage} 次未拿到用量，统计不完整"
    print(f"\n{usage}")
    print(f"运行产物：{context.run_dir}")
    print(f"账本：{context.ledger_path}")
    return 0 if result.patch else 1


def _preview(paths: list[str], limit: int = 8) -> str:
    """把路径列表截成一行可读的预览；超出部分如实标注数量。"""
    shown = "、".join(paths[:limit])
    if len(paths) > limit:
        shown += f" …（共 {len(paths)} 条）"
    return shown


if __name__ == "__main__":
    # 直接 `python src/nw_agent/cli/main.py` 时走这条。正式入口是安装后的
    # `nw-agent` 命令或 `python -m nw_agent`，二者最终也汇到同一个 main()。
    sys.exit(main())