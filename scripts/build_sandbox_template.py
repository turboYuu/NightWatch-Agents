#!/usr/bin/env python
"""定义并构建 NightWatch 的 E2B 基础模板（`nightwatch-base`）。

对应 doc/NightWatch产品说明.md 3.3「沙箱预热 · 基础镜像」：预构建一个带
`git` + `conda` + `pytest` 的模板镜像，后续每次跑任务用
`Sandbox.create(template=...)` 复用，避免每次冷装。conda 发行版默认取 Miniforge
（理由见 `MINIFORGE_INSTALLER_URL` 上方注释），要 Miniconda 加 `--miniconda`。

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

from dotenv import load_dotenv
from e2b import Template

load_dotenv(override=True)

# 构建出来的模板标识；也是 `SandboxConfig(template=...)` 里要写的值。
DEFAULT_TEMPLATE_NAME = "nightwatch-base"

# 沙箱内的仓库根。与 nw_agent.backends.interface.WORKDIR 保持一致
# （脚本不 import 包，故此处重复一次字面量，改一处要同步）。
WORKDIR = "/workspace"

# conda 发行版。E2B 沙箱是 linux-64（x86_64），故取该架构的安装包。
# 装 conda 是因为产品说明 3.3「环境准备」要求按目标仓库自己的依赖清单增量安装，
# 而 environment.yml 只能用 conda 装。
#
# **默认 Miniforge 而非 Miniconda**：
#   1. 走 conda-forge，不含 Anaconda 商业频道，不必处理附加条款；
#   2. 安装包小约 37%（124MB vs 198MB）——镜像大小正是本 spike 要评估的成本项；
#   3. 自带 mamba，P1 按 environment.yml 建环境更快。
# 两者都是 constructor 装出来的 conda，`-b -p` 用法完全一致，可随时用 --miniconda 换回。
MINIFORGE_INSTALLER_URL = (
    "https://github.com/conda-forge/miniforge/releases/latest/download/"
    "Miniforge3-Linux-x86_64.sh"
)
MINICONDA_INSTALLER_URL = (
    "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
)
CONDA_PREFIX = "/opt/conda"

# conda 安装目录里**沙箱用户需要写**的子目录（见 `_add_conda_steps` 末段的理由）。
CONDA_USER_WRITABLE_SUBDIRS = ("envs", "pkgs")

# 沙箱里的默认用户。`e2bdev/base` 声明了 `DEFAULT USER user`(uid 1000)，构建期
# 的 run_cmd 也以它执行——这既是不加 `user="root"` 就会 Permission denied 的原因，
# 也是下面要把 conda 的可写子目录与 WORKDIR 交给它的原因。
SANDBOX_USER = "user"


def build_template(
    *,
    with_conda: bool = True,
    with_pytest: bool = True,
    use_miniconda: bool = False,
) -> Template:
    """返回（尚未构建的）`nightwatch-base` 模板定义。

    基底选 E2B 官方 `e2bdev/base` 而非裸 ubuntu：该镜像已带 envd 等运行时前提，
    自己从裸镜像起步要复刻这些才能被 SDK 正常驱动。

    刻意**不**调用 `set_start_cmd`：模板的启动/就绪命令若为 None，会继承基底镜像自带
    的那套（envd 的启动逻辑）。这里重写只会把它顶掉——对一个「拿来做通用开发沙箱」的
    模板来说没有需要自启动的服务，继承才是对的。

    Args:
        with_conda: 是否预装 conda。关掉可造「轻量变体」做对比实验。
        with_pytest: 是否预装 pytest。关掉则回到「默认模板」的状态（需现装）。
        use_miniconda: 用 Miniconda 而非默认的 Miniforge。

    Returns:
        `TemplateBuilder`/`TemplateFinal`，可直接交给 `Template.build` 或
        `Template.to_dockerfile`。
    """
    template = Template().from_base_image()

    # bzip2 不是可选项：conda 安装器解包时依赖它，缺了会在安装中途失败。
    template = template.apt_install(["git", "ca-certificates", "curl", "bzip2"])

    if with_conda:
        installer_url = MINICONDA_INSTALLER_URL if use_miniconda else MINIFORGE_INSTALLER_URL
        template = _add_conda_steps(template, installer_url)

    if with_pytest:
        # 装在系统 python 上：e2b_backend 的 execute 默认 cwd 在仓库根，跑的就是
        # PATH 上的 python/pytest，与这里一致。`pip_install` 默认 `g=True`，
        # 内部即以 `user="root"` 执行——非 root 写不进系统 site-packages。
        template = template.pip_install("pytest")

    # 建 WORKDIR 并交给沙箱用户。Docker 的 WORKDIR 会以 root 建出目录，那样
    # 沙箱用户往里写不了——上游后端因此得靠 `sudo -n install -d` 兜底。模板是
    # 「预热」的，这步就该在这里做掉，让热路径不必提权。
    template = template.run_cmd(
        f"mkdir -p {WORKDIR} && chown {SANDBOX_USER}:{SANDBOX_USER} {WORKDIR}",
        user="root",
    )

    return template.set_workdir(WORKDIR)


def _add_conda_steps(template: object, installer_url: str) -> object:
    """装上 conda：下载 → 校验 → 安装 → 自证，外加软链与可写目录。

    **为什么下载与安装必须在同一条 RUN 里**：E2B 的构建层缓存不保留 ``/tmp``。
    拆成多条 RUN 时，一旦下载那步命中缓存（内容不进快照），后面那条 RUN 就会
    ``bash: /tmp/conda-installer.sh: No such file or directory``（exit 127）——
    实测踩过。放进同一条 RUN 后安装器不跨步骤，缓存命中与否都成立。

    代价是「哪一步炸了」不再由 RUN 边界体现，故用 ``echo`` 打编号标记补回来：
    RUN 的 stdout 会进构建日志，失败时看最后一条标记就知道断在哪。

    以 ``user="root"`` 执行：基底的 ``DEFAULT USER`` 是 ``user``(uid 1000)，
    而 ``/opt`` 与 ``/`` 对非 root 不可写——安装到 ``/opt/conda``、写 ``/usr/local/bin``
    都只有 root 做得到。E2B 自己的 ``apt_install`` / ``pip_install(g=True)`` 同此约定。

    Args:
        template: 当前 `TemplateBuilder`。
        installer_url: conda 发行版安装器地址。

    Returns:
        追加了各步骤的 `TemplateBuilder`。
    """
    template = template.run_cmd(
        [
            "echo '[1/4] 下载安装器'",
            # --retry：构建环境的网络偶发失败不该让整次构建白跑。
            f"curl -fsSL --retry 3 --retry-delay 2 {installer_url} -o /tmp/conda-installer.sh",
            "echo '[2/4] 校验下载物是脚本'",
            # 错误页或被截断的响应同样是 HTTP 200，直接丢给 bash 只会得到一句含糊的
            # 语法错误；先确认开头是 shebang。
            "head -c 2 /tmp/conda-installer.sh | grep -q '#!' "
            "|| { echo '安装器不是可执行脚本，下载可能被拦截或截断'; exit 1; }",
            f"echo '[3/4] 安装到 {CONDA_PREFIX}'",
            # -b 静默批处理模式，无需交互确认。
            f"bash /tmp/conda-installer.sh -b -p {CONDA_PREFIX}",
            "rm -f /tmp/conda-installer.sh",
            "echo '[4/4] 自证 conda 可用'",
            # 装完立刻自证：这一步过了才说明 conda 真的可用，而不是「安装器返回 0 但没装上」。
            f"{CONDA_PREFIX}/bin/conda --version",
        ],
        user="root",
    )
    # 只把 conda 本体软链到 /usr/local/bin，**不**链 python/pip：
    # 链了会把系统 python 顶掉，而 pytest 装在系统 python 上，两者混用
    # 会让「到底哪个解释器在跑」变得不可预测。conda 留给 P1 建独立环境用。
    # mkdir -p 是保险：基底镜像不保证有 /usr/local/bin，缺了会让 ln 直接失败。
    template = template.run_cmd(
        f"mkdir -p /usr/local/bin && ln -sf {CONDA_PREFIX}/bin/conda /usr/local/bin/conda",
        user="root",
    )
    # conda 装成 root 所有，只把**它需要写的两个空目录**交给沙箱用户：
    # P1 的 `conda create -n xxx` 落 envs、包缓存落 pkgs，两者不可写的话
    # conda 装了等于没装。刻意不 `chown -R /opt/conda`——那会让整个 conda 安装
    # （百余 MB）在新层里复制一份，正好抵消掉换 Miniforge 省下的镜像体积。
    writable = " ".join(f"{CONDA_PREFIX}/{sub}" for sub in CONDA_USER_WRITABLE_SUBDIRS)
    template = template.run_cmd(
        f"mkdir -p {writable} && chown -R {SANDBOX_USER}:{SANDBOX_USER} {writable}",
        user="root",
    )
    return template


def _print_build_log(entry: object) -> None:
    """`on_build_logs` 回调：把每条构建日志打到 stdout。"""
    # LogEntry.__str__ 已含时间戳与级别，直接打印即可（注意它会打印到 stdout）。
    print(entry, file=sys.stdout, flush=True)


def _cmd_print_dockerfile(args: argparse.Namespace) -> int:
    """打印模板对应的 Dockerfile。不联网、不需要 API Key。"""
    template = build_template(
        with_conda=not args.no_conda,
        with_pytest=not args.no_pytest,
        use_miniconda=args.miniconda,
    )
    print(Template.to_dockerfile(template))
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    """在 E2B 服务端构建模板。需要 E2B_API_KEY，且会计费（构建时长）。"""
    template = build_template(
        with_conda=not args.no_conda,
        with_pytest=not args.no_pytest,
        use_miniconda=args.miniconda,
    )
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
    """构造参数解析器。动作必须显式给出（`build` 或 `--print-dockerfile`）。"""
    parser = argparse.ArgumentParser(
        prog="build_sandbox_template.py",
        description="定义并构建 NightWatch 的 E2B 基础模板（nightwatch-base）。",
    )
    # 位置参数，且**故意不给默认值**：构建要花钱、要几分钟，不该因为「什么都没传」
    # 就自动开建。什么都不给时打印帮助并以退出码 2 结束（见 main）。
    parser.add_argument(
        "action",
        nargs="?",
        choices=["build"],
        default=None,
        help="要执行的动作；当前只有 build（在服务端构建模板）",
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
    parser.add_argument("--no-conda", action="store_true", help="不预装 conda")
    parser.add_argument("--no-pytest", action="store_true", help="不预装 pytest")
    parser.add_argument(
        "--miniconda",
        action="store_true",
        help="用 Miniconda 而非默认的 Miniforge（装了 Anaconda 商业频道时用）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """入口。

    `--print-dockerfile` 走离线路径；`build` 走构建；两者都没给时打印帮助并以 2 退出
    ——构建有成本，不设置「默认动作」这回事。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.print_dockerfile:
        return _cmd_print_dockerfile(args)
    if args.action == "build":
        return _cmd_build(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
