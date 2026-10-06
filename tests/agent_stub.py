"""测试用的 stub 模型：不发任何网络请求，按「当前能调哪些工具」决定下一步动作。

定位与 ``e2b_double.py`` 相同——**只存在于 ``tests/``，产品代码引用不到**。它不是
「离线模式」：产品路径永远走真模型（``NW_MODEL_MAIN`` / ``NW_MODEL_FIX``），所以
不提供 CLI 开关来选它。

**为什么按工具名决策而不是按响应顺序**：deepagents 是**嵌套 agent 循环**（主代理 model
节点 → ``task`` 工具 → 子代理自己的 model 节点 → 回到主代理）。谁在第几轮被调用取决于
图的实现细节，按固定顺序喂响应会在 deepagents 升级时错位。绑定工具名区分了两种角色
（主代理有 ``task``、子代理没有），「是否已出现过某个工具结果」则区分了同角色的前后轮
——两者都从当次 ``messages`` 里读，**与调用轮数解耦**。

几个必须遵守的约束（都踩过）：

- ``_llm_type`` 必须是 **property**（``BaseChatModel`` 的抽象方法）；
- ``bind_tools`` 基类直接 ``raise NotImplementedError``，必须覆盖（返回带工具名的副本）；
- 必须能**终止**：没有终态时图会一路撞上 ``recursion_limit``（deepagents 默认 9999）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool

# 自己发出的 tool_call id 前缀。用它判断「这个工具结果是不是我发的」——比依赖
# ToolMessage.name 可靠（后者由工具节点决定要不要填）。
TASK_CALL_ID = "stub-task-call"
EDIT_CALL_ID = "stub-edit-call"

# 每次「调用」上报的假用量。数值本身无意义，用途是让 token 记账链路有数据可断言。
PROMPT_TOKENS = 11
COMPLETION_TOKENS = 7


class StubChatModel(BaseChatModel):
    """按绑定到的工具集合决定发 tool_call 还是给终态文本。"""

    tool_names: tuple[str, ...] = ()
    """由 :meth:`bind_tools` 注入——它随角色变化，是本 stub 判断「我在哪一层」的唯一依据。"""

    final_text: str = "已完成修改。"
    edit_path: str = ""
    """要编辑的文件（沙箱内绝对路径）。空串表示不需要改文件的用例。"""
    old_string: str = ""
    new_string: str = ""

    @property
    def _llm_type(self) -> str:
        # 必须是 property：BaseChatModel 把 _llm_type 声明为抽象方法，
        # 而 summarization 中间件还会去读 model._llm_type.startswith(...)。
        return "nightwatch-stub"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | BaseTool | Any],
        **kwargs: Any,
    ) -> StubChatModel:
        """记录被绑定的工具名。基类实现是 ``raise NotImplementedError``，必须覆盖。"""
        names = tuple(str(getattr(tool, "name", "")) for tool in tools)
        return self.model_copy(update={"tool_names": names})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """按角色与已有工具结果，产出这一轮的 AIMessage。"""
        calls = self._next_calls(messages)
        message = AIMessage(
            content="" if calls else self.final_text,
            tool_calls=calls,
            usage_metadata={
                "input_tokens": PROMPT_TOKENS,
                "output_tokens": COMPLETION_TOKENS,
                "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _next_calls(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        """决定这一轮发什么 tool_call（空列表即终态）。"""
        names = set(self.tool_names)
        if "task" in names:
            # 主代理：委派一次 fix，之后收敛。用 `write_todos` 兜住「主代理被赋予
            # 其他工具」的情形——不影响本测试的关注点，保持最小动作。
            if _has_result(messages, TASK_CALL_ID):
                return []
            return [
                {
                    "name": "task",
                    "args": {"description": "按任务描述修改目标文件", "subagent_type": "fix"},
                    "id": TASK_CALL_ID,
                }
            ]
        if "edit_file" in names and self.edit_path:
            # 子代理：改一次就收敛。
            if _has_result(messages, EDIT_CALL_ID):
                return []
            return [
                {
                    "name": "edit_file",
                    "args": {
                        "file_path": self.edit_path,
                        "old_string": self.old_string,
                        "new_string": self.new_string,
                    },
                    "id": EDIT_CALL_ID,
                }
            ]
        return []


def _has_result(messages: list[BaseMessage], call_id: str) -> bool:
    """``messages`` 里是否已经有这次 tool_call 的结果。"""
    return any(
        isinstance(message, ToolMessage) and message.tool_call_id == call_id
        for message in messages
    )
