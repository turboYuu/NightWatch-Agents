"""``scripts/build_sandbox_template.py`` 的离线回归。

模板只在**构建时**（联网、计费、几分钟）才真正被验证，改错了要等下次构建才发现。
这里用 ``Template.to_dockerfile()`` 把模板定义转成 Dockerfile 文本做断言——不联网、
不需要 Key，把「这个基础镜像里到底装了什么」钉成一条能在 CI 跑的回归。

不测 ``bench_sandbox_cold_start.py``：它是测量工具，验收方式就是真跑一次。
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("e2b", reason="模板由 e2b SDK 定义")

# 构建脚本在 scripts/ 而非包内，导入前需把它加进 sys.path。
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import build_sandbox_template  # noqa: E402
from e2b import Template  # noqa: E402
from e2b.template.types import InstructionType  # noqa: E402

from nw_agent.backends import WORKDIR  # noqa: E402


def dockerfile(**overrides: object) -> str:
    """把模板定义渲染成 Dockerfile 文本。"""
    return Template.to_dockerfile(build_sandbox_template.build_template(**overrides))


def run_instructions(**overrides: object) -> list[tuple[str, str | None]]:
    """返回 ``(命令, 执行用户)`` 列表。

    有些断言在 ``to_dockerfile()`` 的输出上做不到——它**会丢掉 RUN 的执行用户**，
    而「这一步必须以 root 跑」恰是踩过坑的地方（基底 ``DEFAULT USER user``，
    非 root 写不进 ``/opt`` 与 ``/usr/local/bin``）。故这里改读构建器的指令对象。
    """
    template = build_sandbox_template.build_template(**overrides)
    return [
        (instruction["args"][0], instruction["args"][1])
        if len(instruction["args"]) > 1
        else (instruction["args"][0], None)
        # 下划线属性：SDK 没有公开的等价物，而执行用户只在这一层拿得到。
        for instruction in template._template._instructions
        if instruction["type"] == InstructionType.RUN
    ]


def test_default_template_installs_git_conda_and_pytest() -> None:
    """默认模板必须含三件事：git（基线 diff 的前提）、conda、pytest。"""
    rendered = dockerfile()
    assert "install -y git" in rendered
    assert build_sandbox_template.CONDA_PREFIX in rendered
    assert "pip install pytest" in rendered


def test_default_conda_distribution_is_miniforge() -> None:
    """默认取 Miniforge：conda-forge、无附加条款、包更小（见常量上方注释）。"""
    assert "Miniforge3" in dockerfile()
    assert "Miniconda3" not in dockerfile()


def test_miniconda_flag_switches_distribution() -> None:
    """``--miniconda`` 应换回 Anaconda 的 Miniconda 安装器。"""
    rendered = dockerfile(use_miniconda=True)
    assert "Miniconda3" in rendered
    assert "Miniforge3" not in rendered


def test_conda_download_and_install_share_one_run() -> None:
    """下载与安装必须在**同一条 RUN**里，不能拆开。

    E2B 的构建层缓存不保留 `/tmp`：拆开时若下载那步命中缓存（内容不进快照），
    紧随其后的安装步骤就会 `bash: /tmp/conda-installer.sh: No such file or
    directory`（exit 127）。实测踩过这个坑——曾按「多条 RUN 便于定位失败」
    的理由拆开，结果模板根本建不出来。

    代价是 RUN 边界不再标示断点，改用 `echo '[n/4] ...'` 编号标记补回来，
    故一并断言标记存在。
    """
    rendered = dockerfile()
    runs = [line for line in rendered.splitlines() if line.startswith("RUN ")]
    install_runs = [run for run in runs if "conda-installer.sh" in run]
    assert len(install_runs) == 1, "下载与安装必须同处一条 RUN"
    assert "curl" in install_runs[0] and "--version" in install_runs[0]
    assert "[4/4]" in install_runs[0], "缺少编号标记，失败时定位不到断点"


def test_root_only_steps_run_as_root() -> None:
    """写系统路径的步骤必须以 root 执行。

    基底 `e2bdev/base` 声明了 `DEFAULT USER user`(uid 1000)，而 `/opt`、`/` 与
    `/usr/local/bin` 对非 root 不可写。裸 `run_cmd` 会以 `user` 身份跑，实测
    在 `bash installer -p /opt/conda` 上 Permission denied。
    """
    instructions = run_instructions()
    needles = (build_sandbox_template.CONDA_PREFIX, "/usr/local/bin/conda", "pip install pytest")
    for needle in needles:
        matching = [(cmd, user) for cmd, user in instructions if needle in cmd]
        assert matching, f"未找到含 {needle!r} 的 RUN"
        assert all(user == "root" for _, user in matching), f"{needle!r} 所在的 RUN 未以 root 执行"


def test_template_workdir_matches_backend_workdir() -> None:
    """脚本里的 WORKDIR 必须与后端常量一致。

    脚本刻意不 import 这个包（避免 scripts 依赖 src），因此字面量被写了两遍——
    这条断言就是防两处漂移的。
    """
    assert build_sandbox_template.WORKDIR == WORKDIR


def test_no_conda_variant_drops_conda() -> None:
    """``--no-conda`` 变体不应含 conda 安装步骤（用于隔离变量做对比）。"""
    rendered = dockerfile(with_conda=False)
    assert "Miniforge3" not in rendered
    assert build_sandbox_template.CONDA_PREFIX not in rendered
    assert "install -y git" in rendered  # git 不受该开关影响


def test_no_pytest_variant_drops_pytest() -> None:
    """``--no-pytest`` 变体不应含 pytest 安装步骤。"""
    rendered = dockerfile(with_pytest=False)
    assert "pip install pytest" not in rendered
    assert "Miniforge3" in rendered  # conda 不受该开关影响


def test_print_dockerfile_cli_works_offline(capsys: pytest.CaptureFixture[str]) -> None:
    """``--print-dockerfile`` 走通并打印 Dockerfile（这是无 Key 时的验收路径）。"""
    assert build_sandbox_template.main(["--print-dockerfile"]) == 0
    assert "FROM " in capsys.readouterr().out


# ---------------------------------------------------------------- 参数表面
# 这一组守的是「文档里写的调用方式真能跑」。此前没有任何测试覆盖参数解析，
# 于是文档写着 `... build`、argparse 里却没有这个位置参数，直到人手跑才发现。
def test_build_action_is_accepted() -> None:
    """``build`` 必须是可解析的位置参数（文档与基准流程都按这个写法调用）。"""
    args = build_sandbox_template.build_parser().parse_args(["build"])
    assert args.action == "build"
    assert args.print_dockerfile is False


def test_no_action_prints_help_instead_of_building(capsys: pytest.CaptureFixture[str]) -> None:
    """什么都不传时打印帮助并以 2 退出——**不**默认开建（构建有成本）。"""
    assert build_sandbox_template.main([]) == 2
    assert "--print-dockerfile" in capsys.readouterr().out


def test_unknown_action_is_rejected() -> None:
    """拼错的动作名必须被拒绝，而不是被当作合法输入。"""
    with pytest.raises(SystemExit):
        build_sandbox_template.build_parser().parse_args(["buidl"])
