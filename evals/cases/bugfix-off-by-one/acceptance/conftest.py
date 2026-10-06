"""把被测仓库的根放进 sys.path（理由同 docstring-add 用例的同名文件）。"""

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1] / "repo"
sys.path.insert(0, str(_REPO_ROOT))
