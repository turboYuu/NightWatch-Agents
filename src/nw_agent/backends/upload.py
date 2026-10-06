"""上传选文件：把「凭据不进沙箱」从注释变成运行期过滤。

职责（见 doc/NightWatch产品说明.md 3.3「只上传仓库的必要文件子集」）：
- 定义 ``DEFAULT_DENY_PATTERNS``——凭据类文件名/路径的黑名单
- 提供 :func:`select_upload_files`，把一组待上传文件滤成「可上传 / 被拒」两拨

**为什么是黑名单而不是白名单**：白名单（「只允许写 ``target_files``」）是 P1
``write_file`` 的职责，且它针对的是**写**、依赖任务上下文；本模块针对的是**上传**，
在不了解目标仓库结构的前提下只能采取「先堵住已知的凭据形态」这一保守策略。
两者不可互换，故不合并实现。

**与 ``SandboxConfig.__post_init__`` 的分工**：那里的 ``_FORBIDDEN_ENV_MARKERS``
管的是**环境变量键**；本模块管的是**文件路径**。二者是同一条安全原则
（凭据只存宿主机、绝不进沙箱）的两处落地，规则表刻意不复用——键与路径的匹配
语义不同，强行共用只会让两边的规则互相牵制。
"""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence
from pathlib import PurePosixPath

# 凭据类文件的默认黑名单。语义是「文件名或相对路径命中任一模式即拒绝」，
# 匹配大小写不敏感（macOS/Windows 上 `.ENV` 与 `.env` 是同一类东西，不该只在
# Linux 上被拦住）。
#
# 取舍：`.env.*` 会连带拒掉 `.env.example` / `.env.template` 这类无凭据的模板文件。
# 宁可少传一个文件（任务可能因此失败，可见），也不要多传一个凭据（静默泄漏）。
# 确有需要的仓库可在调用处传自定义 ``deny_patterns`` 放开。
DEFAULT_DENY_PATTERNS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa",
    "id_rsa.*",
    "id_ed25519",
    "id_ed25519.*",
    ".ssh",
    ".ssh/*",
    ".aws",
    ".aws/*",
    ".netrc",
    "credentials",
    "credentials.*",
    "*credentials*.json",
)


def is_denied(path: str, patterns: Sequence[str] = DEFAULT_DENY_PATTERNS) -> bool:
    """判断单个路径是否命中凭据黑名单。

    对**整条相对路径**与**basename** 分别匹配：前者让 ``.ssh/*`` 这类目录规则生效，
    后者让 ``../../.env``、``config/id_rsa`` 这类换了目录但文件名明确的路径也拦得住。

    Args:
        path: 相对路径（正反斜杠均可）。
        patterns: 黑名单模式，``fnmatch`` 语义。

    Returns:
        命中任一模式即 True。
    """
    normalized = PurePosixPath(path.replace("\\", "/")).as_posix()
    candidates = (normalized, PurePosixPath(normalized).name)
    return any(
        fnmatch.fnmatch(candidate.lower(), pattern.lower())
        for candidate in candidates
        for pattern in patterns
    )


def select_upload_files(
    files: Sequence[tuple[str, bytes]],
    *,
    deny_patterns: Sequence[str] = DEFAULT_DENY_PATTERNS,
) -> tuple[list[tuple[str, bytes]], list[str]]:
    """把待上传文件滤成「可上传」与「被拒路径」两拨。

    保持输入顺序（上传顺序影响不了结果，但有序输出让报告与日志可比）。

    Args:
        files: ``(相对路径, 内容)`` 序列。
        deny_patterns: 凭据黑名单，见 :data:`DEFAULT_DENY_PATTERNS`。

    Returns:
        ``(允许上传的文件, 被拒的路径列表)``。被拒路径返回给调用方是为了
        **留痕**——静默丢弃会让人误以为文件本来就不在，从而查不出任务为何缺文件。
    """
    allowed: list[tuple[str, bytes]] = []
    rejected: list[str] = []
    for path, content in files:
        if is_denied(path, deny_patterns):
            rejected.append(path)
        else:
            allowed.append((path, content))
    return allowed, rejected
