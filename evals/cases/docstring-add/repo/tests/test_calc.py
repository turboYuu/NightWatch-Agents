"""初始快照自带的测试：**初始即通过**，只覆盖行为、不覆盖 docstring。

它的作用有两个：一是让 verify 命令里的 `tests` 目录真实存在；二是提供回归信号
——solver 若为了凑 docstring 改坏了函数体，这里会红。
"""

from calc import add, sub


def test_add_returns_sum() -> None:
    assert add(2, 3) == 5


def test_sub_returns_difference() -> None:
    assert sub(2, 3) == -1
