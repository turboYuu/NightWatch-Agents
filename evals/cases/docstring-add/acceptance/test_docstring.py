"""docstring-add 的隐藏验收：初始快照下**必然失败**，改对后全绿。

为什么四条断言缺一不可——单独一条「docstring 非空」会被下面几种假解骗过：

- 只有一句凑数的话（例如 ``TODO``）：由 ``test_docstring_is_relevant`` 拦下
- 把函数签名抄一遍当 docstring：同上（关键词 + 最小长度）
- 给 sub() 也加上（改错对象 / 越界改动）：由 ``test_other_function_untouched`` 与 L3 拦下
- 删掉函数体只留 docstring：``getdoc`` 非空但功能没了，由 ``test_add_behavior_unchanged`` 拦下
"""

import inspect

from calc import add, sub

_MIN_DOC_CHARS = 8
_RELEVANT_KEYWORDS = ("sum", "add", "相加", "求和", "返回", "和")


def test_add_has_docstring() -> None:
    assert inspect.getdoc(add), "add() 缺 docstring"


def test_docstring_is_relevant() -> None:
    doc = (inspect.getdoc(add) or "").lower()
    assert len(doc) >= _MIN_DOC_CHARS, f"docstring 过短，疑似凑数：{doc!r}"
    assert any(word in doc for word in _RELEVANT_KEYWORDS), f"docstring 与功能无关：{doc!r}"


def test_add_behavior_unchanged() -> None:
    assert add(2, 3) == 5
    assert add(-1, 1) == 0


def test_other_function_untouched() -> None:
    """sub() 不在 target_files 里，是本次任务的反向哨兵。"""
    assert inspect.getdoc(sub) is None
    assert sub(2, 3) == -1