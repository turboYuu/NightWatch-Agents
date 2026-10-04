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

from nw_agent.backends import WORKDIR  # noqa: E402


def dockerfile(**overrides: bool) -> str:
    """把模板定义渲染成 Dockerfile 文本。"""
    return Template.to_dockerfile(build_sandbox_template.build_template(**overrides))


def test_default_template_installs_git_conda_and_pytest() -> None:
    """默认模板必须含三件事：git（基线 diff 的前提）、conda、pytest。"""
    rendered = dockerfile()
    assert "install -y git" in rendered
    assert "Miniconda3" in rendered
    assert "pip install pytest" in rendered


def test_template_workdir_matches_backend_workdir() -> None:
    """脚本里的 WORKDIR 必须与后端常量一致。

    脚本刻意不 import 这个包（避免 scripts 依赖 src），因此字面量被写了两遍——
    这条断言就是防两处漂移的。
    """
    assert build_sandbox_template.WORKDIR == WORKDIR


def test_no_conda_variant_drops_miniconda() -> None:
    """``--no-conda`` 变体不应含 conda 安装步骤（用于隔离变量做对比）。"""
    rendered = dockerfile(with_conda=False)
    assert "Miniconda3" not in rendered
    assert "install -y git" in rendered  # git 不受该开关影响


def test_no_pytest_variant_drops_pytest() -> None:
    """``--no-pytest`` 变体不应含 pytest 安装步骤。"""
    rendered = dockerfile(with_pytest=False)
    assert "pip install pytest" not in rendered
    assert "Miniconda3" in rendered  # conda 不受该开关影响


def test_print_dockerfile_cli_works_offline(capsys: pytest.CaptureFixture[str]) -> None:
    """``--print-dockerfile`` 子命令走通并打印 Dockerfile（这是无 Key 时的验收路径）。"""
    assert build_sandbox_template.main(["--print-dockerfile"]) == 0
    assert "FROM " in capsys.readouterr().out
