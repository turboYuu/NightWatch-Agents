"""把仓库根放进 sys.path（理由同 docstring-add 用例的同名文件）。"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
