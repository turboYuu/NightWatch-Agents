"""DeepAgent 内层运行时（阶段内循环）。

职责（见 doc/NightWatch产品说明.md 3.2）：
- 用 ``create_deep_agent`` 装配主代理与 analyze / search / fix 子代理
- 由主代理在**一次运行内**按序通过 task 工具委派，完成 理解→定位→出补丁
- 子代理默认 isolated：不继承父代理完整历史，由主代理显式传递精炼上下文

边界：子代理是 DeepAgent 的 subagents，**不是** LangGraph 节点；
阶段顺序只由主代理决定，避免与 ``nw_agent.graph`` 的边重复编排。

P0 尚未实现，仅占位以固定目录结构。
"""
