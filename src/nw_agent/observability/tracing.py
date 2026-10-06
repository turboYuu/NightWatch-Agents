"""LangSmith tracing 的接线：**只设环境变量，不 import langsmith**。

为什么这样切：LangChain 家族（含 deepagents）是否上报 trace，取决于**环境变量**，
我们不需要（也不该）直接依赖 langsmith 的 Python API。于是生产代码不 import 它，
`pyproject.toml` 与锁文件都不用动——它已经作为 `deepagents` 的连带依赖存在
（见 doc/依赖管理.md 的「已知事项」）。

以下三条是从已安装的 `langsmith 0.14.4` 源码里**核对过**的事实，改动前请重新核对：

1. **开关判定是 ``== "true"`` 的精确比较**（``utils.py:141-142``）：值必须是**小写
   字符串** ``true``；``True`` / ``1`` / ``yes`` 都不会开启。别想当然改成 ``1``。
2. **``get_env_var`` 带 ``lru_cache``**（``utils.py:420``，且全库没有 ``cache_clear``
   调用）：**晚设的环境变量可能被缓存屏蔽**。因此有一条顺序契约——
   ``load_dotenv`` → :func:`configure_tracing` → 再触发任何被 trace 的调用。
   CLI 的 ``main()`` 第一行就调本函数，正是为了抢在它前面。
3. **变量前缀优先 ``LANGSMITH`` 后 ``LANGCHAIN``**（``utils.py:425``）：新名优先，
   旧名仍生效，故两个都写能挡住「某个第三方只读旧名」这种静默丢数据的失败模式。

⚠️ **外发风险**：开启后，Issue 正文、代码片段、diff 与提示词都会离开本地上传到
LangSmith（第三方 SaaS）。与 3.3 对云端沙箱的立场一致：私有仓库、含密钥的仓库默认
不建议开启。详见 doc/可观测性.md。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import cast

# 开关变量（新名在前，优先）。值是精确的字符串 "true"，不是布尔。
ENV_TRACING = ("LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2")
# API Key 变量。key 单独设置是**无效**的：必须同时有开关变量为 true。
ENV_API_KEY = ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")
# 项目名变量。缺省时 SDK 落到 "default"。
ENV_PROJECT = ("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT")
TRACING_ON = "true"


@dataclass(frozen=True)
class TracingStatus:
    """本次运行 tracing 的实际状态。

    **必须被打印并写进账本**——「显式」的意义就在这里：缺 key 时若不吭声，
    trace 会静默丢失，而人以为一切正常。
    """

    enabled: bool
    project: str | None = None
    reason: str | None = None
    key_env: str | None = None
    """提供 key 的那个环境变量的**名字**（不是 key 本身）。新旧名混用时能一眼看出。"""

    def describe(self) -> str:
        """一句话人话，供控制台与报告使用。"""
        if self.enabled:
            return f"已启用（project={self.project or 'default'}，key 来自 {self.key_env}）"
        return f"未启用：{self.reason or '原因未知'}"

    def as_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "project": self.project,
            "reason": self.reason,
            "key_env": self.key_env,
        }

    @classmethod
    def from_dict(cls, raw: object) -> TracingStatus:
        """从账本行还原。字段不符时抛，由 ``ledger.read`` 计为坏行。"""
        if not isinstance(raw, dict):
            raise TypeError("tracing 必须是对象")
        data = cast("dict[str, object]", raw)
        enabled = data["enabled"]
        if not isinstance(enabled, bool):
            raise TypeError("tracing.enabled 必须是布尔")
        return cls(
            enabled=enabled,
            project=_opt_str(data.get("project")),
            reason=_opt_str(data.get("reason")),
            key_env=_opt_str(data.get("key_env")),
        )


def configure_tracing(enabled: bool, *, project: str | None = None) -> TracingStatus:
    """按需开启 LangSmith tracing，返回实际状态。

    Args:
        enabled: 是否请求开启（CLI 的 ``--trace``）。**False 时一个变量都不改**。
        project: LangSmith 项目名；None 时沿用环境里的设置（SDK 缺省落 "default"）。

    Returns:
        实际状态。缺 API Key 时返回 ``enabled=False`` 并给出原因——因为 key 单独设置
        或开关单独设置都收不到数据，宁可显式报告「没开」，也不假装成功。

    两条行为约定：

    - **``enabled=False`` 时不碰任何变量。** 用户 shell 里可能自有采集（CI 就在这么干），
      我们没有资格把它关掉；而且「是否落盘我们的运行产物」与「LangSmith 是否收数据」
      是两件事，把前者缺席等同于关闭后者是范畴错误。
    - **只写 ``true``，永不写 ``false``。** 两个开关名都写，且值一致；给其中任一写
      ``false`` 都可能意外关掉用户的全局 tracing。
    """
    if not enabled:
        return TracingStatus(
            enabled=False,
            reason=(
                "未请求（环境变量里已自行开启）"
                if tracing_env_enabled()
                else "未请求（--trace 未开启）"
            ),
        )

    key_env = _first_non_empty(ENV_API_KEY)
    if key_env is None:
        return TracingStatus(
            enabled=False,
            reason=(
                f"缺少 {ENV_API_KEY[0]}（或旧名 {ENV_API_KEY[1]}）——"
                "缺 key 时 trace 会静默丢失，故不假装开启"
            ),
        )

    for name in ENV_TRACING:
        os.environ[name] = TRACING_ON
    if project:
        # 只写新名：它读在前，足以覆盖旧名；旧名若由用户设过，保持原样不去改。
        os.environ[ENV_PROJECT[0]] = project
    return TracingStatus(
        enabled=True,
        project=project or _first_non_empty(ENV_PROJECT) or "default",
        key_env=key_env,
    )


def tracing_env_enabled() -> bool:
    """环境变量当前是否已开启 tracing（复刻 SDK 的 ``== "true"`` 判定）。"""
    return any(os.environ.get(name) == TRACING_ON for name in ENV_TRACING)


def _first_non_empty(names: tuple[str, ...]) -> str | None:
    """按优先级取第一个非空环境变量**名**（返回名字而非值，避免把 key 带进日志/账本）。"""
    for name in names:
        if (os.environ.get(name) or "").strip():
            return name
    return None


def _opt_str(value: object) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise TypeError("应为字符串或 null")
