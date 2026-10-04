#!/usr/bin/env python
"""定义并构建 NightWatch 的 E2B 基础模板（`nightwatch-base`）。

对应 doc/NightWatch产品说明.md 3.3「沙箱预热 · 基础镜像」：预构建一个带
`git` + `conda` + `pytest` 的模板镜像，后续每次跑任务用
`Sandbox.create(template=...)` 复用，避免每次冷装。

**为什么是脚本而不是包内模块**：`src/nw_agent/` 严格按产品说明第 2 章的六层划分
（cli/graph/agents/tools/backends/memory），构建模板不属于其中任何一层；上层要用
这个模板只需在 `SandboxConfig(template="nightwatch-base")` 里写个名字，不需要 import
任何代码。故放顶层 `scripts/`。

用法：

    # 只打印模板会生成的 Dockerfile——不联网、不需要 API Key，供 review 与离线测试
    python scripts/build_sandbox_template.py --print-dockerfile

    # 真正构建（服务端构建，约几分钟，需要 E2B_API_KEY）
    python scripts/build_sandbox_template.py build
    python scripts/build_sandbox_template.py build --name nightwatch-base-min   # 造变体做对比

构建完成后，跑基准脚本时用 `--template nightwatch-base` 指定它。
"""

from __future__ import annotations

import argparse
import sys

from e2b import Template

# 构建出来的模板标识；也是 `SandboxConfig(template=...)` 里要写的值。
DEFAULT_TEMPLATE_NAME = "nightwatch-base"

# 沙箱内的仓库根。与 nw_agent.backends.interface.WORKDIR 保持一致
# （脚本不 import 包，故此处重复一次字面量，改一处要同步）。
WORKDIR = "/workspace"

# Miniconda 安装器。E2B 沙箱是 linux-64（x86_64），故取该架构的官方安装包。
# 装它是因为产品说明 3.3「环境准备」要求按目标仓库自己的依赖清单增量安装，而
# environment.yml 只能用 conda 装。
MINICONDA_INSTALLER_URL = (
    "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
)
MINICONDA_PREFIX = "/opt/miniconda3"


def build_template(*, with_conda: bool = True, with_pytest: bool = True) -> Template:
    """返回（尚未构建的）`nightwatch-base` 模板定义。

    基底选 E2B 官方 `e2bdev/base` 而非裸 ubuntu：该镜像已带 envd 等运行时前提，
    自己从裸镜像起步要复刻这些才能被 SDK 正常驱动。

    刻意**不**调用 `set_start_cmd`：模板的启动/就绪命令若为 None，会继承基底镜像自带
    的那套（envd 的启动逻辑）。这里重写只会把它顶掉——对一个「拿来做通用开发沙箱」的
    模板来说没有需要自启动的服务，继承才是对的。

    Args:
        with_conda: 是否预装 Miniconda。关掉可造「轻量变体」做对比实验。
        with_pytest: 是否预装 pytest。关掉则回到「默认模板」的状态（需现装）。

    Returns:
        `TemplateBuilder`/`TemplateFinal`，可直接交给 `Template.build` 或
        `Template.to_dockerfile`。
    """
    template = Template().from_base_image()

    # bzip2 不是可选项：Miniconda 安装器解包时依赖它，缺了会在安装中途失败。
    template = template.apt_install(["git", "ca-certificates", "curl", "bzip2"])

    if with_conda:
        # 传列表给 run_cmd，SDK 会用 `&&` 合成一条 RUN（不是各给一条 RUN）——任一步失败
        # 整条失败，构建日志里看到的是一条命令。
        template = template.run_cmd(
            [
                f"curl -fsSL {MINICONDA_INSTALLER_URL} -o /tmp/miniconda.sh",
                # -b 静默批处理模式，无需交互同意许可。
                f"bash /tmp/miniconda.sh -b -p {MINICONDA_PREFIX}",
                "rm -f /tmp/miniconda.sh",
                # 只把 conda 本体软链到 /usr/local/bin，**不**链 python/pip：
                # 链了会把系统 python 顶掉，而 pytest 装在系统 python 上，两者混用
                # 会让「到底哪个解释器在跑」变得不可预测。conda 留给 P1 建独立环境用。
                f"ln -sf {MINICONDA_PREFIX}/bin/conda /usr/local/bin/conda",
            ]
        )

    if with_pytest:
        # 装在系统 python 上：e2b_backend 的 execute 默认 cwd 在仓库根，跑的就是
        # PATH 上的 python/pytest，与这里一致。
        template = template.pip_install("pytest")

    return template.set_workdir(WORKDIR)


def _print_build_log(entry: object) -> None:
    """`on_build_logs` 回调：把每条构建日志打到 stdout。"""
    # LogEntry.__str__ 已含时间戳与级别，直接打印即可（注意它会打印到 stdout）。
    print(entry, file=sys.stdout, flush=True)


def _cmd_print_dockerfile(args: argparse.Namespace) -> int:
    """打印模板对应的 Dockerfile。不联网、不需要 API Key。"""
    template = build_template(with_conda=not args.no_conda, with_pytest=not args.no_pytest)
    print(Template.to_dockerfile(template))
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    """在 E2B 服务端构建模板。需要 E2B_API_KEY，且会计费（构建时长）。"""
    template = build_template(with_conda=not args.no_conda, with_pytest=not args.no_pytest)
    print(f"开始构建模板 {args.name!r}（服务端构建，约几分钟）……", flush=True)
    info = Template.build(
        template,
        name=args.name,
        skip_cache=args.skip_cache,
        on_build_logs=_print_build_log,
    )
    print(f"\n构建完成：name={info.name} template_id={info.template_id} build_id={info.build_id}")
    print(f"后续用 SandboxConfig(template={args.name!r}) 复用它。")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构造参数解析器。默认动作是构建；`--print-dockerfile` 切到离线模式。"""
    parser = argparse.ArgumentParser(
        prog="build_sandbox_template.py",
        description="定义并构建 NightWatch 的 E2B 基础模板（nightwatch-base）。",
    )
    parser.add_argument(
        "--print-dockerfile",
        action="store_true",
        help="只打印模板的 Dockerfile 后退出（离线、无需 Key），不构建",
    )
    # 构建相关开关。--print-dockerfile 模式下 --name 无意义，但保留以便两种模式共用一份参数。
    parser.add_argument("--name", default=DEFAULT_TEMPLATE_NAME, help="构建出的模板名")
    parser.add_argument(
        "--skip-cache", action="store_true", help="忽略构建缓存，全量重建（排查缓存问题时用）"
    )
    # 变体开关：用于 spike 隔离变量，回答「conda 增大镜像是否拖慢启动」。
    parser.add_argument("--no-conda", action="store_true", help="不预装 Miniconda")
    parser.add_argument("--no-pytest", action="store_true", help="不预装 pytest")
    return parser


def main(argv: list[str] | None = None) -> int:
    """入口。`--print-dockerfile` 走离线路径，否则构建。"""
    args = build_parser().parse_args(argv)
    if args.print_dockerfile:
        return _cmd_print_dockerfile(args)
    return _cmd_build(args)


if __name__ == "__main__":
    sys.exit(main())
