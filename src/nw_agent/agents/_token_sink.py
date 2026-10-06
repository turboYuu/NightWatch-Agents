"""把模型用量接进运行账本，并在超预算时熔断。

**为什么必须走 callback 而不是读 ``invoke`` 的返回值**：deepagents 的子代理在**独立的
上下文窗口**里跑（产品说明 3.2 的 isolated 模式），顶层 ``result["messages"]`` 只留下
主代理自己的消息——实测一次「主代理委派 → 子代理改文件 → 主代理收敛」的运行，顶层只有
2 条 AIMessage，而挂在 ``config={"callbacks": [...]}`` 上的 handler 收到了 **4 次**
``on_llm_end``。子代理显式继承父 config 的 callbacks（``deepagents/middleware/subagents.py``
里有明文注释），所以 callback 是唯一能拿全用量的地方。

⚠️ **回调里抛异常默认会被吞掉**：``langchain_core`` 的 ``handle_event`` 捕获 handler 的
任何异常后**只打一条 warning**，除非 handler 的 ``raise_error`` 为真
（``callbacks/manager.py:344-346``：``if handler.raise_error: raise``）。因此本类的
``raise_error = True`` 是**预算熔断能生效的前提**——不设它，超预算异常会被静默丢弃，
账面上「有熔断」，实际上一次都不会触发。

代价是：本类里的任何异常都会中断整个运行。这正是想要的——计数逻辑出错应当**响亮地**
失败，而不是悄悄把 token 记少。
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult

from nw_agent.observability import NoOpRecorder, Recorder

# 从 provider 的原始返回里找模型名时依序尝试的键。各 provider 命名不统一，故是多候选。
_MODEL_NAME_KEYS = ("model_name", "model", "model_id")


class BudgetExceeded(RuntimeError):
    """本次运行的 token 用量超出预算上限，已中断。

    由 :class:`TokenSink` 在 ``on_llm_end`` 里抛出、经 ``raise_error = True`` 穿透
    callback 层，最终从 ``agent.invoke()`` 冒出来。
    """

    def __init__(self, used: int, budget: int) -> None:
        super().__init__(f"token 预算熔断：已用 {used}，上限 {budget}")
        self.used = used
        self.budget = budget


class TokenSink(BaseCallbackHandler):
    """累计每次模型调用的用量，写进 ``recorder``，必要时熔断。

    ``raise_error = True`` 的理由见模块 docstring：**不设它，熔断就是空话**。
    """

    # 见模块 docstring。langchain-core 只有在这个类属性为真时才会把 handler 的异常
    # 往上抛，否则一律吞成 warning。
    raise_error = True

    def __init__(
        self,
        recorder: Recorder | None = None,
        *,
        model_label: str,
        budget_tokens: int | None = None,
    ) -> None:
        """Args:
        recorder: 用量落点；None 表示不记（只是计数与熔断）。
        model_label: 拿不到真实模型名时使用的标签（通常是 ``NW_MODEL_MAIN`` 的串）。
        budget_tokens: 预算上限；None 表示不熔断（仅计数）。
        """
        super().__init__()
        self._recorder = recorder if recorder is not None else NoOpRecorder()
        self._model_label = model_label
        self._budget = budget_tokens
        self._seen: set[UUID] = set()
        self.total_tokens = 0
        self.calls = 0
        self.calls_without_usage = 0

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """累计本次调用的用量；超预算即抛 :class:`BudgetExceeded`。

        ``run_id`` 去重是防御性的：当前版本里聊天模型只经由 ``on_llm_end`` 上报一次，
        但若将来 langchain-core 同时派发 ``on_chat_model_end``，去重能保证不会双计。
        """
        if run_id in self._seen:
            return
        self._seen.add(run_id)

        for generation_list in response.generations:
            for generation in generation_list:
                self._consume(generation, response)
        if self._budget is not None and self.total_tokens > self._budget:
            raise BudgetExceeded(self.total_tokens, self._budget)

    def _consume(self, generation: Generation, response: LLMResult) -> None:
        """处理单条 generation：取用量、记一笔、累加。"""
        message = generation.message if isinstance(generation, ChatGeneration) else None
        usage = _usage_of(message)
        if usage is None:
            # 拿不到用量就**不记**（而不是记 0）：本项目的口径是「缺数据为 None，
            # 写成 0 会被读成零成本」。这里至少要让人知道有几笔漏了。
            self.calls_without_usage += 1
            return
        input_tokens, output_tokens = usage
        self._recorder.record_tokens(
            _model_name(message, response, fallback=self._model_label),
            input_tokens,
            output_tokens,
        )
        self.total_tokens += input_tokens + output_tokens
        self.calls += 1


def _usage_of(message: BaseMessage | None) -> tuple[int, int] | None:
    """从消息里取 ``(input_tokens, output_tokens)``；没有用量元数据返回 None。

    用 ``usage_metadata`` 而不是 ``response_metadata``：后者是 provider 原始返回、字段名
    不统一（实测常见为空字典），前者是 langchain-core 归一化后的标准字段
    （``AIMessage.usage_metadata``）。
    """
    if message is None:
        return None
    usage = getattr(message, "usage_metadata", None)
    if not isinstance(usage, dict):
        return None
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None
    return input_tokens, output_tokens


def _model_name(message: BaseMessage | None, response: LLMResult, *, fallback: str) -> str:
    """尽力取真实模型名；取不到就回落到调用方注入的标签。

    **不猜**：候选键都试过仍拿不到，就用 ``fallback``（``NW_MODEL_MAIN`` 的值），
    至少账本里能定位到「这次运行用的是哪个配置」。回落到配置串而不是留空，是因为
    ``TokenUsage.model`` 是必填字段，空串会让账本难以解读。
    """
    sources: list[dict[str, Any]] = []
    if message is not None and isinstance(message.response_metadata, dict):
        sources.append(message.response_metadata)
    if isinstance(response.llm_output, dict):
        sources.append(response.llm_output)
    for source in sources:
        for key in _MODEL_NAME_KEYS:
            value = source.get(key)
            if isinstance(value, str) and value:
                return value
    return fallback
