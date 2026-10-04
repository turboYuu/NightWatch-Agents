"""``SandboxBackend`` 与 deepagents 协议的契约测试。

这些断言守的是「必须真继承 ``BaseSandbox``」这个设计决策：deepagents 判定后端
能力用的是**类属性身份比对**而非 ``isinstance``/鸭子类型。一旦有人把继承换成
自带 Protocol、或覆写了下面那几个旧名字，这里会立刻变红——比写在注释里可靠。

全部是**类级**断言，不实例化后端、不联网；行为回归见 ``tests/test_backends_e2b.py``。
"""

import pytest

pytest.importorskip("deepagents", reason="SandboxBackend 依赖 deepagents；见 pyproject.toml")

from deepagents.backends.protocol import (  # noqa: E402
    BackendProtocol,
    execute_accepts_timeout,
)
from deepagents.backends.sandbox import BaseSandbox  # noqa: E402

from nw_agent.backends import E2BSandboxBackend, SandboxBackend  # noqa: E402

# 这些名字一旦被子类覆写，deepagents 会认为该后端「实现的是旧 API」并走废弃分支。
_LEGACY_API_NAMES = ("ls_info", "grep_raw", "glob_info")


def test_backend_is_a_base_sandbox_subclass() -> None:
    """后端必须是 ``BaseSandbox`` 的子类——鸭子类型通不过 deepagents 的能力探测。"""
    assert issubclass(E2BSandboxBackend, BaseSandbox)
    assert issubclass(E2BSandboxBackend, SandboxBackend)


@pytest.mark.parametrize("name", _LEGACY_API_NAMES)
def test_legacy_api_names_are_not_overridden(name: str) -> None:
    """不得覆写 ``ls_info`` / ``grep_raw`` / ``glob_info``。"""
    assert getattr(E2BSandboxBackend, name) is getattr(BackendProtocol, name)


def test_execute_accepts_timeout_keyword() -> None:
    """deepagents 会探测 ``execute`` 是否接受 timeout；不接受就走降级路径。"""
    assert execute_accepts_timeout(E2BSandboxBackend) is True


def test_kill_clears_abstract_requirement() -> None:
    """具体后端必须已实现全部抽象成员，否则根本无法实例化。"""
    assert E2BSandboxBackend.__abstractmethods__ == frozenset()
