"""DeepAgent 内层运行时（阶段内循环）。

职责（见 doc/NightWatch产品说明.md 3.2）：
- 用 ``create_deep_agent`` 装配主代理与 fix 子代理
- 由主代理在**一次运行内**通过 ``task`` 工具委派，完成 定位→出补丁
- 子代理默认 isolated：不继承父代理完整历史，由主代理显式传递精炼上下文

边界：子代理是 DeepAgent 的 subagents，**不是** LangGraph 节点；阶段顺序只由主代理决定，
避免与 ``nw_agent.graph`` 的边重复编排。

进度：P1 第一周只显式声明一个 ``fix`` 子代理（``analyze`` / ``search`` 留到第二周）。
``create_deep_agent`` 会自动补一个 ``general-purpose`` 子代理——这是**有意识的接受默认**，
理由与重审时机写在 ``factory`` 的模块 docstring 里。
"""

from __future__ import annotations

from nw_agent.agents._token_sink import BudgetExceeded, TokenSink
from nw_agent.agents.factory import (
    FIX_MODEL_ENV,
    MAIN_MODEL_ENV,
    AgentConfigError,
    AgentModels,
    build_main_agent,
    main_system_prompt,
    resolve_agent_models,
)
from nw_agent.agents.runner import DEFAULT_RECURSION_LIMIT, RunTaskResult, run_task

__all__ = [
    "DEFAULT_RECURSION_LIMIT",
    "FIX_MODEL_ENV",
    "MAIN_MODEL_ENV",
    "AgentConfigError",
    "AgentModels",
    "BudgetExceeded",
    "RunTaskResult",
    "TokenSink",
    "build_main_agent",
    "main_system_prompt",
    "resolve_agent_models",
    "run_task",
]
