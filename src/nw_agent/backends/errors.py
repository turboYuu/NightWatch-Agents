"""backends 包的共用异常与错误字面量。

单独成模块（而不是塞进 ``interface.py``）是为了让上层的 ``tools/`` 在 P1 能
只 import 异常类型，而不牵扯抽象基类与 deepagents 依赖。

本模块刻意不 import 任何第三方：异常与字面量是最底层的公共约定。
"""

from __future__ import annotations


class SandboxClosedError(RuntimeError):
    """沙箱已销毁后仍被调用动作。

    显式失败而不是把远端的「沙箱不存在」往上透传——后者在上层看来与
    网络故障难以区分，而这里的语义是**调用方误用**，应当立刻暴露。
    """


# ---------------------------------------------------------------------------
# deepagents 的 FileOperationError 字面量
# ---------------------------------------------------------------------------
# 这些字符串要与 deepagents / langchain_e2b 的协议约定一致，上层（P1 的 tools/、
# 测试结果解析）才能直接按字面量判定，不必关心后端实现。
ERR_INVALID_PATH = "invalid_path"
ERR_IS_DIRECTORY = "is_directory"
ERR_FILE_NOT_FOUND = "file_not_found"
ERR_PERMISSION_DENIED = "permission_denied"

# 命令超时对应的退出码。与 langchain_e2b.TIMEOUT_EXIT_CODE 对齐，
# 上层据此识别「超时」，不必关心后端如何发现超时。
TIMEOUT_EXIT_CODE = 124
