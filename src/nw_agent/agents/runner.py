"""跑一次任务：invoke 主代理 → 导出补丁 → 记 token。

这一层是 P1 交付物 `nw-agent --repo <local> --task "..."` 的核心，也是
``RunContext.record_tokens`` 的第一个真实调用点——5.7 度量表里「单任务 token 成本」
由此从 ``null`` 变成真数字。

三条边界：

1. **补丁由宿主侧导出**（``backend.export_diff()``，产品说明 3.3），**不信任模型返回的
   diff 文本**。模型说的「我改了 X」只是叙述，补丁是事实。
2. **不负责销毁沙箱**。生命周期归调用方（CLI 用 ``with create_backend(...)``），
   在这里 kill 会让「谁负责清理」变模糊——虽然 ``kill()`` 幂等，但职责要单一。
3. **不吞异常**。超预算抛 :class:`~nw_agent.agents._token_sink.BudgetExceeded`、
   模型鉴权失败抛 provider 的异常，都由调用方决定怎么记账与退出。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from nw_agent.agents._token_sink import TokenSink
from nw_agent.backends import SandboxBackend
from nw_agent.observability import Recorder

# 单次运行的步数上限，覆盖 deepagents 自带的 9999。
#
# 默认 9999 是**没有任何保护的**：一个打转的循环能一路烧到天亮（产品说明 3.1 的预算熔断
# 就是这个用途的一半）。100 步对一个「委派一次 fix、改几个文件」的任务绰绰有余。
DEFAULT_RECURSION_LIMIT = 100


@dataclass(frozen=True)
class RunTaskResult:
    """一次任务运行的结果。"""

    final_text: str
    """主代理给用户的最终文本（末条 AI 消息的正文）。"""

    patch: str
    """宿主侧导出的 unified diff；**空串表示没有改动**，不是错误。"""

    models: tuple[str, ...]
    """本次使用的模型标识（主、fix 两个配置串），进账本便于事后对账。"""

    total_tokens: int
    """累计 token（主代理 + 各子代理）。拿不到用量时可能为 0，见 ``calls_without_usage``。"""

    n_llm_calls: int
    """成功记到用量的模型调用次数。"""

    n_calls_without_usage: int
    """**有**调用但没拿到用量元数据的次数。非 0 说明 token 统计不完整——
    如实报出来，而不是把它当 0 记进成本。"""

    def as_dict(self) -> dict[str, object]:
        """转成可直接进 JSON/快照的字典。"""
        return {
            "final_text": self.final_text,
            "patch": self.patch,
            "models": list(self.models),
            "total_tokens": self.total_tokens,
            "n_llm_calls": self.n_llm_calls,
            "n_calls_without_usage": self.n_calls_without_usage,
        }


def run_task(
    agent: CompiledStateGraph[Any, Any, Any, Any],
    backend: SandboxBackend,
    task: str,
    *,
    model_label: str,
    recorder: Recorder | None = None,
    budget_tokens: int | None = None,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> RunTaskResult:
    """跑一次任务并返回结果。

    Args:
        agent: :func:`~nw_agent.agents.factory.build_main_agent` 装配出的主代理。
        backend: 该 agent 所绑的沙箱后端（补丁从它导出）。
        task: 自然语言任务描述（CLI 的 ``--task``）。
        model_label: 拿不到真实模型名时的回落标签，通常传 ``NW_MODEL_MAIN`` 的值。
        recorder: 用量与阶段的落点；None 表示只计数不落盘。
        budget_tokens: token 预算上限；超出即熔断（抛 ``BudgetExceeded``）。
        recursion_limit: 步数上限，默认 :data:`DEFAULT_RECURSION_LIMIT`。

    Returns:
        :class:`RunTaskResult`。

    Raises:
        BudgetExceeded: 累计 token 超预算。
        Exception: 模型鉴权失败、provider 报错等，原样上抛（调用方负责记账与退出码）。
    """
    sink = TokenSink(recorder, model_label=model_label, budget_tokens=budget_tokens)
    raw: object = agent.invoke(
        {"messages": [{"role": "user", "content": task}]},
        # recursion_limit 必须在这里覆盖：deepagents 把 9999 写死进了自己的 config。
        config={"callbacks": [sink], "recursion_limit": recursion_limit},
    )
    return RunTaskResult(
        final_text=_final_text(raw),
        patch=backend.export_diff(),
        models=(model_label,),
        total_tokens=sink.total_tokens,
        n_llm_calls=sink.calls,
        n_calls_without_usage=sink.calls_without_usage,
    )


def _final_text(raw: object) -> str:
    """从 ``invoke`` 的返回值里取末条消息的正文。

    这是本模块**唯一**处理 ``Any`` 的地方（``agent.invoke`` 的返回类型是
    ``dict[str, Any] | Any``）。逐层收窄，取不到就返回空串——调用方拿不到文本不该
    让整个运行失败（补丁才是交付物）。
    """
    if not isinstance(raw, dict):
        return ""
    messages = raw.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    return _content_text(getattr(messages[-1], "content", None))


def _content_text(content: object) -> str:
    """把消息正文归一成字符串。

    ``content`` 可能是 str，也可能是多模态的 block 列表（``[{"type": "text", "text": ...}]``）。
    只取文本部分，其余类型直接忽略。
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(str(block["text"]))
    return "\n".join(parts)
