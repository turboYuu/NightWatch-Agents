"""CLI 参数契约的冒烟测试。

这些测试不依赖任何第三方运行时依赖（cli.main 只用到标准库），
因此即使在还没装 langgraph / deepagents 的干净环境里也能跑，
用于守住 P0 定义的参数契约不被后续改动破坏。
"""

import pytest

from nw_agent.cli.main import DEFAULT_TOKEN_BUDGET, parse_args


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
