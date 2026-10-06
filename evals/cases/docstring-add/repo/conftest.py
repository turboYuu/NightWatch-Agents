"""把仓库根放进 sys.path。

没有它，`from calc import add` 只在 pytest 恰好把仓库根插进 sys.path 时成立。
显式插一行，行为与 import 模式、rootdir 推断都无关。
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
