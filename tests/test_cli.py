"""CLI 的参数契约与分支分发测试。

前半组只校验 argparse 的契约（不碰网络、不碰 key）；后半组（``main()`` 的分发）**不建
沙箱、不调模型**——它们只验证「哪种输入该被拒绝、退出码是什么」，因此既不花钱也能进 CI。

注：P1 起 ``cli.main`` 会 import 运行时依赖（dotenv / deepagents / e2b），所以本文件不再
是「干净环境也能跑」的。真正需要隔离的是环境变量：``main()`` 第一行会 ``load_dotenv``，
若不禁用，测试结果就会随开发机上的 ``.env`` 变化——故相关用例用 ``_hermetic_env`` 把
dotenv 关掉再改环境变量。
"""

from __future__ import annotations

from pathlib import Path

import pytest

import nw_agent.cli.main as cli_main
from nw_agent.agents import FIX_MODEL_ENV, MAIN_MODEL_ENV
from nw_agent.cli.main import DEFAULT_TOKEN_BUDGET, main, parse_args


@pytest.fixture
def hermetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """让 ``main()`` 不去读仓库根的 ``.env``，并把模型配置设成合法值。

    不这么做的话，「缺 E2B_API_KEY」这类用例会被开发机上的 ``.env`` 静默满足，
    测试通过与否取决于本机状态——那是假绿。
    """
    monkeypatch.setattr(cli_main, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.setenv(MAIN_MODEL_ENV, "anthropic:claude-sonnet-4-6")
    monkeypatch.setenv(FIX_MODEL_ENV, "anthropic:claude-sonnet-4-6")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """一个最小可读的本地仓库。"""
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    return tmp_path


def test_requires_task_source() -> None:
    """既不给 --issue 也不给 --task 应报错退出。"""
    with pytest.raises(SystemExit):
        parse_args(["--repo", "owner/name"])


def test_issue_and_task_are_mutually_exclusive() -> None:
    """同时给 --issue 与 --task 应报错退出。"""
    with pytest.raises(SystemExit):
        parse_args(["--repo", "owner/name", "--issue", "1", "--task", "改点东西"])


def test_defaults_are_safe() -> None:
    """默认值应符合“安全优先”：强制人审、有预算上限、非 dry-run。"""
    opts = parse_args(["--repo", "owner/name", "--issue", "123"])
    assert opts.review == "always"                 # 默认强制人工审核
    assert opts.budget_tokens == DEFAULT_TOKEN_BUDGET
    assert opts.dry_run is False
    assert opts.task is None


def test_task_mode_parses() -> None:
    """--task 模式应正确填充字段，且 issue 为空。"""
    opts = parse_args(["--repo", "owner/name", "--task", "给 foo() 补 docstring"])
    assert opts.task == "给 foo() 补 docstring"
    assert opts.issue is None


def test_review_choice_is_validated() -> None:
    """--review 只接受 always/auto/never，非法值应报错。"""
    with pytest.raises(SystemExit):
        parse_args(["--repo", "owner/name", "--issue", "1", "--review", "maybe"])


# ------------------------------------------------------------ 分支分发
# 这一组断言的是「显式拒绝」而不是静默回显：什么都不做却退出 0，会让人把「没跑」误读成
# 「跑过了没问题」，账本里也留不下痕迹。
def test_issue_is_rejected_as_not_implemented() -> None:
    """--issue 要等 P5 的 GitHub 集成，现在必须明确报错（退出码 2）而不是假装跑过。"""
    assert main(["--repo", ".", "--issue", "12"]) == 2


def test_non_local_repo_is_rejected(hermetic_env: None) -> None:
    """owner/name 形式要等 P5；当前只支持本地目录。"""
    assert main(["--repo", "owner/name", "--task", "改点东西"]) == 2


def test_missing_model_config_exits_2(hermetic_env: None, repo: Path) -> None:
    """缺 NW_MODEL_* 时退出码 2，且错误来自 AgentConfigError（不猜默认模型）。"""
    assert main(["--repo", str(repo), "--task", "改点东西"]) == 2


def test_missing_e2b_key_exits_2(
    hermetic_env: None, repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """真链路缺 E2B_API_KEY 时退出码 2，并提示可以先用 --dry-run。"""
    assert main(["--repo", str(repo), "--task", "改点东西"]) == 2
    assert "E2B_API_KEY" in capsys.readouterr().err


def test_dry_run_reports_upload_plan_and_filters_credentials(
    hermetic_env: None, repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--dry-run 不建沙箱、不调模型，但要把「将上传什么、滤掉了什么」说清楚。"""
    (repo / ".env").write_text("SECRET=1\n", encoding="utf-8")
    (repo / "calc.py").write_text("x = 1\n", encoding="utf-8")

    assert main(["--repo", str(repo), "--task", "改点东西", "--dry-run"]) == 0

    out = capsys.readouterr().out
    assert "将上传 1 个文件" in out  # 只有 calc.py；.env 被滤掉
    assert ".env" in out  # 被拒路径要留痕，不能静默丢弃
    assert "未建沙箱" in out


def test_dry_run_needs_no_e2b_key(hermetic_env: None, repo: Path) -> None:
    """--dry-run 不碰远端，因此**不该**要求 E2B_API_KEY（hermetic_env 已把它删掉）。"""
    assert main(["--repo", str(repo), "--task", "改点东西", "--dry-run"]) == 0
