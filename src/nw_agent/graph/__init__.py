"""LangGraph 外层状态机（编排层）。

职责（见 doc/NightWatch产品说明.md 3.1）：
- 定义 ``MaintenanceState`` 与各阶段节点：
  fetch_issue / classify / run_agent / human_review / apply_fix / test_result / create_pr
- 持久化检查点（SqliteSaver）、HITL 中断/恢复、失败重试与预算熔断

边界：阶段**内部**的工具调用与子代理调度不在这里，而在 ``nw_agent.agents``。
本层只决定「下一步走哪个阶段」。

P0 尚未实现，仅占位以固定目录结构。
"""
