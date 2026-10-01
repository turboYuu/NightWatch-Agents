**代码仓库“夜间维护者”（Coding Agent）**

# 1. 场景定义与边界

**做什么**：一个本地运行的 CLI Agent，接收 GitHub Issue 编号或自然语言任务描述，自动完成从“理解需求 → 定位代码 → 生成修复 → 沙箱验证 → 创建 PR”的全流程。

**不做什么**：不追求“全自动重构整个系统”这类模糊需求。MVP 阶段只处理边界清晰的技术债类任务，例如依赖升级、补充单元测试、修复 lint 错误、处理简单的 bug 报告。

**为什么值得做**：OpenHands 已有 77K+ stars，但它的定位是“通用自主软件工程 Agent”，体量大、部署复杂。你的差异化在于做“夜间维护”这个极窄切片——不需要交互式 UI，不需要多轮对话，只需要在 CI 或定时任务中跑通“Issue → PR”的闭环。

# 2. 技术架构总览

```text
┌─────────────────────────────────────────────────────────────┐
│                     CLI 入口 (osc-agent)                     │
│        参数: --repo owner/name --issue 123                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              LangGraph 主图 (Orchestrator)                   │
│  状态: issue → plan → subagent_results → fix → test → pr    │
│  能力: 检查点持久化、中断恢复、人工审核门控                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ 分析子代理   │  │ 搜索子代理   │  │ 修复子代理   │
│ Analyze     │  │ Search      │  │ Fix         │
│ SubAgent    │  │ SubAgent    │  │ SubAgent    │
└─────────────┘  └─────────────┘  └─────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│           DeepAgent 运行时 (create_deep_agent)               │
│  工具: ripgrep_search, read_file, write_file, execute_shell │
│  后端: OpenSandboxBackend (本地 Docker 隔离)                 │
│  记忆: AGENTS.md (项目级) + memories/ (自动积累)            │
└─────────────────────────────────────────────────────────────┘
```

# 3. 核心章节详解

## 3.1 第1章：LangGraph 状态图设计

为什么需要 LangGraph：Agent 的工作流不是简单的“一次 LLM 调用 → 输出”，而是有明确的阶段边界：获取 Issue → 分析 → 搜索 → 修复 → 测试 → 提交 PR。每个阶段可能需要中断等待人工审核，也可能失败后回退重试。

状态定义（简化版）：

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class MaintenanceState(TypedDict):
    messages: Annotated[list, add_messages]  # 对话历史
    issue_title: str                          # GitHub Issue 标题
    issue_body: str                           # Issue 描述
    repo_path: str                            # 本地仓库路径
    target_files: list[str]                   # 搜索到的相关文件
    proposed_fix: str                         # 修复方案
    sandbox_id: str | None                    # 沙箱实例 ID
    test_passed: bool | None                  # 测试结果
    pr_url: str | None                        # 最终 PR 链接
```

节点划分：

|节点|职责| 是否可中断  |
|--- |--- |--------|
|fetch_issue|从 GitHub API 拉取 Issue 内容	| 否      
|analyze|	调用子代理理解问题、提取技术关键词|	否
|search	|调用子代理在代码库中定位相关文件	|否
|propose_fix	|调用子代理生成修复方案	|是（人工审核）
|apply_fix	|在沙箱中写入修改并运行测试|否
|test_result|	根据测试结果决定继续或回退	|否
|create_pr	|推送分支、创建 Pull Request	|是（人工确认）

关键实现：

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

builder = StateGraph(MaintenanceState)
builder.add_node("fetch_issue", fetch_issue_node)
builder.add_node("analyze", analyze_node)
builder.add_node("search", search_node)
builder.add_node("propose_fix", propose_fix_node)
builder.add_node("apply_fix", apply_fix_node)
builder.add_node("create_pr", create_pr_node)

builder.add_edge(START, "fetch_issue")
builder.add_edge("fetch_issue", "analyze")
builder.add_edge("analyze", "search")
builder.add_edge("search", "propose_fix")
builder.add_conditional_edges(
    "propose_fix",
    should_proceed,
    {"apply": "apply_fix", "abort": END}
)
builder.add_conditional_edges(
    "apply_fix",
    test_decision,
    {"pass": "create_pr", "retry": "propose_fix", "fail": END}
)
builder.add_edge("create_pr", END)

graph = builder.compile(checkpointer=InMemorySaver())
```

为什么用 LangGraph 而不是纯 LangChain：LangChain 的 create_agent 是一个循环，适合“工具调用直到完成”的简单模式。但夜间维护者的流程是有阶段的流水线，每个阶段有不同的上下文、不同的工具权限、不同的人工审核要求。LangGraph 的状态图 + 检查点能力正好匹配这种需求——你可以随时暂停图执行、保存状态、等人工审批后从断点恢复。

## 3.2 第2章：DeepAgent 子代理编排

为什么需要子代理：夜间维护者需要做三件上下文完全不同的事：分析 Issue（需要理解自然语言）、搜索代码（需要 ripgrep/glob）、生成修复（需要读写文件 + 执行测试）。如果用一个 Agent 串行做所有事，主对话历史会被大量中间结果污染。

DeepAgent 的子代理机制：DeepAgent 允许主代理通过 task 工具将工作委派给子代理。子代理在独立的上下文窗口中执行，只返回简洁的最终结果给主代理。

子代理定义：

```python
from deepagents import create_deep_agent

main_agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    system_prompt="""你是代码仓库维护编排器。
    使用 task 工具将工作委派给子代理：
    - analyze: 理解 Issue 的技术需求
    - search: 在代码库中定位相关文件
    - fix: 生成并验证修复方案
    不要自己直接搜索或修改文件。""",
    subagents=[
        {
            "name": "analyze",
            "description": "分析 Issue 内容，提取技术关键词、受影响的模块、建议的搜索策略",
            "system_prompt": "你是技术分析师。阅读 Issue 后输出：1) 问题类型 2) 涉及模块 3) 建议搜索关键词",
        },
        {
            "name": "search",
            "description": "在本地代码库中搜索与问题相关的文件和代码片段",
            "system_prompt": "你是代码搜索专家。使用 ripgrep 搜索文件，返回最相关的 5 个文件路径和匹配行",
        },
        {
            "name": "fix",
            "description": "根据分析结果和搜索结果，生成代码修复方案并在沙箱中验证",
            "system_prompt": "你是修复工程师。基于提供的上下文修改文件，然后运行测试验证。报告测试结果。",
        },
    ],
)
```

关键设计决策：子代理默认是 isolated 模式，只接收委派任务，不继承父代理的完整对话历史。这意味着分析子代理看不到搜索子代理的结果，反之亦然。这种隔离是故意的——它迫使主代理扮演“调度者”角色，负责在子代理之间传递精炼后的信息，而不是让所有 Agent 都淹没在冗长的上下文中。

## 3.3 第3章：沙箱执行与安全隔离

为什么需要沙箱：代码修复 Agent 需要执行 npm install、pytest、cargo build 等命令。如果直接在宿主机运行，一个错误的 rm -rf 或恶意依赖脚本可能破坏开发环境。

方案选择：

|方案|隔离级别|部署复杂度| 适用场景 |
|---|---|---|---|
|LocalShellBackend|	无（直接在宿主机执行）|	零	|仅本地开发调试
|OpenSandboxBackend|Docker 容器级	|中（需本地 Docker）|	MVP 推荐
|LangSmithSandbox|云端 VM	|高（需 API Key）	|生产部署

OpenSandboxBackend 配置：

```python
from deepagents import create_deep_agent
from deepagents_opensandbox_backend import OpenSandboxBackend

backend = OpenSandboxBackend.create(
    api_key="你的随机密钥",  # 必须设置，否则默认无认证[citation:6]
    use_server_proxy=True,   # Windows/Mac 需要
)

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    backend=backend,
    system_prompt="你在隔离沙箱中工作。使用 execute 工具运行测试命令。",
)

try:
    result = agent.invoke({
        "messages": "运行 pytest tests/test_parser.py"
    })
    print(result["messages"][-1].content)
finally:
    backend.kill()  # 确保沙箱销毁
```

安全警告：OpenSandbox 的默认配置极度不安全——无 API 认证、允许挂载 Docker socket、CORS 全开。在 .sandbox.toml 中必须修改：

```toml
[server]
api_key = "生成长随机字符串"

[storage]
allowed_host_paths = ["/tmp/opensandbox-data"]  # 明确白名单
```

安全原则：沙箱只应挂载仓库的克隆副本，绝不挂载 ~/.ssh、~/.aws、.env 文件或宿主机 Docker socket。Agent 的 execute 工具即使被 prompt injection 攻击，也最多只能破坏一个一次性的容器。

## 3.4 第4章：项目级记忆（AGENTS.md）

为什么需要记忆：如果每次运行都从零开始理解项目结构，Agent 会反复做同样的探索。项目应该有一个持久化的“项目知识文件”，记录：代码规范、测试命令、架构约定、已知陷阱。

AGENTS.md 机制：DeepAgent 在会话启动时自动加载：
- 全局：~/.deepagents/<agent_name>/AGENTS.md 
- 项目级：.deepagents/AGENTS.md（Git 仓库根目录）

项目级 AGENTS.md 示例：
```markdown
# 项目上下文：my-python-service

## 架构
- 入口：src/app/main.py
- 测试：pytest，配置在 pyproject.toml
- 依赖管理：uv（不用 pip install）

## 代码规范
- 使用 snake_case 命名
- 所有公共函数必须有 type hints
- 禁止直接修改 migrations/ 目录

## 测试命令
- 单元测试：`uv run pytest tests/unit/ -x`
- 集成测试：`uv run pytest tests/integration/ -x --timeout=60`

## 已知陷阱
- `config.py` 中的 `LEGACY_MODE` 默认为 True，修改前需要先确认
- 数据库迁移只能在测试环境中运行
自动记忆积累：Agent 在运行过程中会自动将新发现写入 ~/.deepagents/<agent_name>/memories/，例如 api-conventions.md、dependency-patterns.md。下一次运行时，Agent 会先检索记忆再开始工作。
```
## 3.5 第5章：工具集设计

Coding Agent 的工具不在于“多”，而在于精确和最小化。每个工具都应该有清晰的输入输出契约。

|工具|用途| 权限范围 |
|---|---|----|
|ripgrep_search	|在代码库中搜索文本模式	|只读，限定 repo_path
|read_file	|读取文件内容	|只读，限定 repo_path
|write_file	|写入修复后的文件	|写权限，限定沙箱内 /workspace/
|execute_shell	|在沙箱中运行命令	|受限：白名单命令（pytest, uv, npm）
|create_github_pr	|创建 Pull Request	|需要 GitHub Token，人工确认后触发

工具实现示例：

```python
from langchain.tools import tool

@tool
def ripgrep_search(pattern: str, repo_path: str) -> str:
    """在代码库中搜索文本模式。返回匹配的文件路径和行号。"""
    import subprocess
    result = subprocess.run(
        ["rg", "--line-number", "--max-count", "50", pattern, repo_path],
        capture_output=True, text=True, timeout=30
    )
    return result.stdout[:5000]  # 截断，避免上下文爆炸
```

为什么不用通用 execute 工具：DeepAgent 的 LocalShellBackend 自带一个 execute 工具，但它允许任意 shell 命令。夜间维护者应该显式定义白名单工具，而不是给 Agent 一个万能命令执行器。这既降低了安全风险，也让 Agent 的行为更可预测。

## 3.6 第6章：人工审核门控（HITL）

为什么需要审核：完全自主的 PR 创建是危险的。Agent 可能在“修复”一个简单 lint 错误时，意外删除了一个关键函数。人工审核门控让 Agent 在关键决策点暂停。

LangGraph 的中断机制：

```python
from langgraph.types import interrupt, Command

def propose_fix_node(state: MaintenanceState):
    # ... Agent 生成修复方案 ...
    fix_plan = agent_output
    
    # 在应用修复前暂停，等待人工确认
    approval = interrupt({
        "action": "review_fix",
        "fix_plan": fix_plan,
        "target_files": state["target_files"],
        "question": "是否批准此修复方案？回复 'approve' 或提供修改意见。"
    })
    
    if approval.get("decision") == "reject":
        return {"proposed_fix": None, "aborted": True}
    
    return {"proposed_fix": fix_plan}
```
恢复执行：

```python
# 人工审核后恢复
thread_config = {"configurable": {"thread_id": "issue-123"}}
graph.invoke(Command(resume={"decision": "approve"}), config=thread_config)
```

HITL 的本质：不是简单的“确认弹窗”，而是 Runtime 层面的状态暂停与恢复。LangGraph 的检查点机制让图执行可以在任意节点中断，保存完整状态，等待外部输入后从断点继续。

# 4. 与现有方案的差异化

|维度|OpenHands|OSC-Agent| 夜间维护者（本方案） |
|---|---|---|------------|
|定位	|通用自主软件工程师	|开源贡献 CLI	|定时/CI 触发的技术债清理
|交互模式|	对话式 + 自主执行	|CLI 命令	|无头模式 + 人工审核门控
|沙箱	|Docker 全环境	|E2B 云端	|本地 Docker（OpenSandbox）
|子代理	|未明确分层	|多 Agent 但非 DeepAgent	|DeepAgent isolated 子代理
|记忆	|无项目级持久记忆	|无	|AGENTS.md + 自动记忆积累
|目标用户	|开发者手动使用	|开源贡献者	|CI 系统 / 团队夜间自动化

# 5. 开发路线图

## 第1周：骨架跑通

搭建 create_deep_agent + 一个子代理 + OpenSandbox
实现最简单的任务：“给指定的函数补充 docstring”
验证：Agent 能在沙箱中读取文件、修改、输出 diff
## 第2周：搜索与定位

集成 ripgrep_search 工具
实现 analyze 和 search 子代理
验证：给定一个 Issue 描述，Agent 能找到正确的目标文件
## 第3周：LangGraph 编排

将线性流程替换为 LangGraph 状态图
加入 interrupt 审核门控
验证：Agent 在应用修复前暂停，人工确认后继续
## 第4周：GitHub 集成与记忆

实现 create_github_pr 工具（需要 GitHub Token）
写入项目级 AGENTS.md，配置自动记忆积累
验证：完整的 Issue → PR 闭环
## 第5周+：开源准备

编写 README（包含 Docker 一键启动脚本）
录制演示视频
提交到 GitHub，配置 GitHub Actions 自动运行
# 6. 关键风险与规避

|风险| 规避策略  |
|---|-------|
|Agent 修改了不相关的文件	|在 propose_fix 节点后加人工审核；限制 write_file 只能操作搜索阶段确定的 target_files
|沙箱逃逸	|使用 OpenSandbox Docker 隔离，绝不挂载宿主机敏感目录
|LLM 幻觉导致错误修复	|强制在沙箱中运行测试；测试不通过则回退并报告
|上下文爆炸	|子代理 isolated 模式；工具结果截断（5000 字符上限）
|Token 成本失控	|简单任务用便宜模型（如 Claude Haiku）；子代理按需调用

一句话总结：这个项目的核心价值不在于“让 AI 写代码”，而在于用 DeepAgent 的架构能力（子代理隔离 + 沙箱 + 记忆）构建一个可控、可审核、可复现的自动化维护流水线。范围越窄（只做“夜间维护”），完成度越高，在 GitHub 上被 star 和复用的概率越大。