"""全测试级隔离：任何用例都不许往真实用户目录写运行产物。

**为什么是 autouse 的 fixture 而不是「记得传 tmp_path」**：后者靠纪律，一次疏忽就往
``~/.nightwatch`` 里写了东西，而且写进去之后没人会发现（它不影响测试结果）。这里把
``NW_HOME`` 强制指到本用例的 ``tmp_path``，于是「默认路径」在测试里天然是临时的——
哪怕有人写了一个完全没考虑隔离的用例，也污染不到真实目录。

``monkeypatch`` 而非直接改 ``os.environ``：用例异常退出时它仍会恢复环境，
不会把 ``NW_HOME`` 泄漏给同会话里的后续用例。
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_nw_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """把运行产物根目录钉在本用例的临时目录下。"""
    monkeypatch.setenv("NW_HOME", str(tmp_path / "nw-home"))
