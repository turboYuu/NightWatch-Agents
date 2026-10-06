"""极小示例模块：评测用例 bugfix-off-by-one 的初始快照。

``page_bounds`` 的 docstring 描述的是**期望行为**，代码却是错的——这是真实仓库里
最常见的形态（文档对、实现错），也让「照 docstring 改」成为一条合法解法。
"""


def page_bounds(page: int, per_page: int) -> tuple[int, int]:
    """返回第 page 页（从 1 开始计）在完整序列中的切片区间 ``[start, end)``。"""
    start = page * per_page
    end = start + per_page
    return start, end
