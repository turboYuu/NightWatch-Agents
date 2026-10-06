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
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

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


# ---------------------------------------------------------------------------
# 本地仓库 → 上传子集
# ---------------------------------------------------------------------------
# 上传要跳过的目录名。判断依据是「与任务无关且体积大」：版本库基线由沙箱侧重建、
# 依赖与缓存目录动辄上万文件（上传耗时是整条链路的主要成本项）、IDE 配置与代码理解无关。
SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".idea",
        ".vscode",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".ipynb_checkpoints",
    }
)

# 默认上限。**必须有上限**：一个失控的大仓库会让上传卡到超时，而失败现象是
# 「任务超时」而不是「文件太多」，不设限就查不出原因。
DEFAULT_MAX_FILES = 2000
DEFAULT_MAX_BYTES = 32 * 1024 * 1024  # 32 MiB


@dataclass(frozen=True)
class LocalRepoFiles:
    """本地仓库的读取结果。"""

    files: list[tuple[str, bytes]]
    """已过滤、可直接喂给 ``upload_repo`` 的 ``(仓库内相对路径, 内容)``。"""

    rejected: list[str]
    """命中凭据黑名单、**刻意不上传**的路径。"""

    skipped: list[str]
    """因超出文件数/字节上限而被丢弃的路径。**必须报出来**——静默截断会让人以为
    上传是完整的，任务失败时又会怀疑是模型改错而不是「文件压根没送进去」。"""

    truncated: bool
    """是否发生了截断（等价于 ``bool(skipped)``，单列是为了让调用点读起来更直白）。"""

    def summary(self) -> str:
        """一行摘要，供 CLI 打印。"""
        total_bytes = sum(len(content) for _, content in self.files)
        text = f"将上传 {len(self.files)} 个文件（{total_bytes / 1024:.0f} KiB）"
        if self.rejected:
            text += f"，滤掉 {len(self.rejected)} 个凭据类文件"
        if self.skipped:
            text += f"，⚠️ 因超上限丢弃 {len(self.skipped)} 个文件"
        return text


def read_local_repo(
    path: Path,
    *,
    deny_patterns: Sequence[str] = DEFAULT_DENY_PATTERNS,
    max_files: int = DEFAULT_MAX_FILES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> LocalRepoFiles:
    """把本地仓库读成「可上传的文件子集」。

    顺序：先按目录黑名单与凭据黑名单过滤，再按**累计字节数**与文件数截断。截断发生在
    排序之后，因此结果对同一仓库是确定的（可重复），不会这次漏这几个、下次漏那几个。

    Args:
        path: 本地仓库根目录。
        deny_patterns: 凭据黑名单，见 :data:`DEFAULT_DENY_PATTERNS`。
        max_files: 文件数上限。
        max_bytes: 总字节上限（按**压缩前**的原始大小算）。

    Returns:
        :class:`LocalRepoFiles`。

    Raises:
        NotADirectoryError: ``path`` 不是目录。
    """
    if not path.is_dir():
        raise NotADirectoryError(f"不是目录：{path}")

    kept: list[tuple[str, bytes]] = []
    rejected: list[str] = []
    skipped: list[str] = []
    total_bytes = 0

    for file_path in sorted(path.rglob("*")):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(path)
        if SKIP_DIR_NAMES & set(relative.parts):
            continue
        posix = relative.as_posix()
        if is_denied(posix, deny_patterns):
            rejected.append(posix)
            continue
        content = file_path.read_bytes()
        if len(kept) >= max_files or total_bytes + len(content) > max_bytes:
            skipped.append(posix)
            continue
        kept.append((posix, content))
        total_bytes += len(content)

    return LocalRepoFiles(
        files=kept,
        rejected=rejected,
        skipped=skipped,
        truncated=bool(skipped),
    )
