"""沙箱后端抽象与实现。

职责（见 doc/NightWatch产品说明.md 3.3）：
- 定义统一的 ``SandboxBackend`` 接口：创建 / 上传代码 / 执行命令 / 导出 diff / 销毁
- ``E2BSandboxBackend``：唯一实现，云端一次性沙箱（注意代码外发风险）

上层只依赖接口，替换后端无需改动编排与工具代码——构造入口只有 :func:`create_backend`。

**导入边界（刻意设计）**：
    ``e2b`` 是必需依赖，因此本包直接向 :mod:`~nw_agent.backends.e2b_backend` 取
    ``E2BSandboxBackend``，import 本包即 import e2b。

    但 ``deepagents`` 同样是**运行时**依赖：接口本身继承它的 ``BaseSandbox``
    （deepagents 的能力探测是类属性身份比对，鸭子类型不成立，理由见 ``interface``
    模块的 docstring）。
"""

from __future__ import annotations

from typing import Literal

from nw_agent.backends.e2b_backend import E2BSandboxBackend
from nw_agent.backends.errors import (
    ERR_FILE_NOT_FOUND,
    ERR_INVALID_PATH,
    ERR_IS_DIRECTORY,
    ERR_PERMISSION_DENIED,
    SandboxClosedError,
)
from nw_agent.backends.interface import (
    ACCEPTANCE_PATH,
    REPO_PATH,
    WORKDIR,
    SandboxBackend,
    SandboxConfig,
)
from nw_agent.backends.upload import (
    DEFAULT_DENY_PATTERNS,
    is_denied,
    select_upload_files,
)

# 后端种类。加新后端时这里与 create_backend 一并扩展。
BackendKind = Literal["e2b"]

__all__ = [
    "DEFAULT_DENY_PATTERNS",
    "ERR_FILE_NOT_FOUND",
    "ERR_INVALID_PATH",
    "ERR_IS_DIRECTORY",
    "ERR_PERMISSION_DENIED",
    "ACCEPTANCE_PATH",
    "REPO_PATH",
    "WORKDIR",
    "BackendKind",
    "E2BSandboxBackend",
    "SandboxBackend",
    "SandboxClosedError",
    "SandboxConfig",
    "create_backend",
    "is_denied",
    "select_upload_files",
]


def create_backend(
    kind: BackendKind = "e2b",
    config: SandboxConfig | None = None,
) -> SandboxBackend:
    """按名字创建并返回一个沙箱后端。

    保留 ``kind`` 参数（而非直接暴露 ``E2BSandboxBackend.create``）是为了守住
    「换隔离方案只改 ``backend=`` 一处」这条 seam；当前值域只有 ``"e2b"``。

    Args:
        kind: ``"e2b"``（默认且当前唯一的生产后端）。
        config: 创建参数；None 时用 ``SandboxConfig()`` 的默认值。

    Returns:
        已创建、可直接使用的后端。**销毁由调用方负责**——推荐用 ``with``：

            with create_backend("e2b") as backend:
                ...

    Raises:
        ValueError: ``kind`` 不是已知的后端类型。
        RuntimeError: 创建失败（远端错误已包装为可读中文）。
    """
    backend_config = config if config is not None else SandboxConfig()
    if kind == "e2b":
        return E2BSandboxBackend.create(backend_config)
    raise ValueError(f"未知的后端类型：{kind!r}；当前可选 'e2b'。")
