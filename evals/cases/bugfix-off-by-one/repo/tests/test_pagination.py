"""初始快照自带的测试：**带 bug 也通过**。

两条断言在 ``page * per_page`` 与 ``(page - 1) * per_page`` 下都成立，因此它们
提供的是回归信号（防止改坏窗口宽度或页间衔接），而不是正确性判据。真正能区分
对错的是隐藏验收里的绝对索引断言——这刻意复现了「仓库自有测试覆盖不到 bug」的
真实场景。
"""

from pagination import page_bounds


def test_window_size_is_per_page() -> None:
    start, end = page_bounds(2, 10)
    assert end - start == 10


def test_adjacent_pages_are_contiguous() -> None:
    _, first_end = page_bounds(1, 10)
    second_start, _ = page_bounds(2, 10)
    assert second_start == first_end
