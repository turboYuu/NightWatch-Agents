"""bugfix-off-by-one 的隐藏验收：初始快照下**必然失败**，改对后全绿。

断言用的是**绝对索引**（而不是「窗口宽度」这类相对量），因为相对量在错位的实现下
同样成立——那正是仓库自带测试漏掉这个 bug 的原因。
"""

from pagination import page_bounds


def test_first_page_starts_at_zero() -> None:
    assert page_bounds(1, 10) == (0, 10)


def test_second_page() -> None:
    assert page_bounds(2, 10) == (10, 20)


def test_non_uniform_page_size() -> None:
    assert page_bounds(3, 25) == (50, 75)


def test_single_element_pages() -> None:
    assert page_bounds(4, 1) == (3, 4)


def test_sequence_covers_everything_without_gaps() -> None:
    """把区间拼起来，必须恰好覆盖 0..12 且不重叠——错位一页会整体移位。"""
    covered = [index for page in range(1, 5) for index in range(*page_bounds(page, 3))]
    assert covered == list(range(12))
