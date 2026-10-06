"""把被测仓库的根放进 sys.path。

用 ``Path(__file__).resolve().parents[1] / "repo"`` 相对自身定位，而不是写死
``/workspace/repo``——验收测试离线（注入 SDK 替身的临时目录）与 E2B（/workspace）
两条链路都要能跑，而替身不翻译沙箱绝对路径。
"""

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1] / "repo"
sys.path.insert(0, str(_REPO_ROOT))
