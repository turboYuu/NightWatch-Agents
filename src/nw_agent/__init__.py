"""NightWatch 夜间维护者（Coding Agent）。

一个可在本地或 CI 中无头运行的 CLI Agent：接收 GitHub Issue 或自然语言任务，
自动完成「理解需求 → 定位代码 → 生成补丁 → 沙箱验证 → 创建 PR 草案」的全流程。

包结构与 doc/NightWatch产品说明.md 第 2 章的分层一一对应：

    cli/      命令行入口与参数契约
    graph/    LangGraph 外层状态机（阶段 / 检查点 / HITL / 重试 / 预算熔断）
    agents/   DeepAgent 内层运行时（主代理 + analyze/search/fix 子代理）
    tools/    交给 Agent 的沙箱工具（read_file / write_file / execute_shell ...）
    backends/ SandboxBackend 抽象与实现（E2B 默认 / Fake 仅测试）
    memory/   项目级记忆（AGENTS.md）与按 repo 隔离的自动记忆积累

关键边界（勿越界）：
- 外层 LangGraph 决定「走哪个阶段」，内层 DeepAgent 决定「本次运行内怎么把活干完」。
- 凭据（GitHub Token）只存在于宿主机编排层，绝不进入沙箱。
- 记忆由宿主机写入，沙箱内不落记忆。
"""

# 项目版本，供 CLI / 打包使用。与 pyproject.toml 的 version 保持一致。
__version__ = "0.0.1"
