"""初始快照自带的测试：**初始即通过**，只覆盖行为。"""

from calc import mul


def test_mul_returns_product() -> None:
    assert mul(2, 3) == 6
