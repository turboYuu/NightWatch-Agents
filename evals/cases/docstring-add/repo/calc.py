"""极小示例模块：评测用例 docstring-add 的初始快照。

刻意保持极小（十几行、KB 级）：用例要能秒级上传与导出 diff，否则会拖垮
5.7 度量表里「端到端耗时 < 5 分钟」这条。
"""


def add(a: int, b: int) -> int:
    return a + b


def sub(a: int, b: int) -> int:
    return a - b
