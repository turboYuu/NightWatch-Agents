**代码仓库“夜间维护者”（Coding Agent）**

# 1. 场景定义与边界

**做什么**：一个可在本地或 CI 中无头运行的 CLI Agent，接收 GitHub Issue 编号或自然语言任务描述，自动完成从“理解需求 → 定位代码 → 生成修复 → 沙箱验证 → 创建 PR”的全流程。Agent 的编排与调度在本地/CI 进程内运行，代码执行与测试则放入隔离沙箱（默认云端 E2B，见 3.3）。

**不做什么**：不追求“全自动重构整个系统”这类模糊需求。MVP 阶段只处理边界清晰的技术债类任务，例如依赖升级、补充单元测试、修复 lint 错误、处理简单的 bug 报告。

**为什么值得做**：OpenHands 已有 77K+ stars，但它的定位是“通用自主软件工程 Agent”，体量大、部署复杂。你的差异化在于做“夜间维护”这个极窄切片——不需要交互式 UI，不需要多轮对话，只需要在 CI 或定时任务中跑通“Issue → PR”的闭环。

# 2. 技术架构总览

```text
┌─────────────────────────────────────────────────────────────┐
│                    CLI 入口 (nw-agent)                      │
│        参数: --repo owner/name  --issue 123  --task ...     │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│       LangGraph 外层状态机 (Orchestrator) ── 管「阶段」     │
│  职责: 阶段边界 / 检查点持久化 / HITL 门控 / 失败重试路由   │
│  状态: issue → plan → fix → test → pr                       │
│  检查点: 持久化 checkpointer (SQLite)，支持跨进程恢复       │
└─────────────────────────────────────────────────────────────┘
                          │ 每个阶段节点内调用
                          ▼
┌─────────────────────────────────────────────────────────────┐
│     DeepAgent 运行时 (create_deep_agent) ── 管「阶段内循环」│
│  职责: 单阶段内的工具调用循环 + 子代理委派 (task 工具)      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                      │
│  │ analyze │  │ search  │  │  fix    │  ← isolated 子代理   │
│  └─────────┘  └─────────┘  └─────────┘                      │
│  工具: ripgrep_search, read_file, write_file, execute_shell │
│  后端: E2BSandboxBackend (SandboxBackend 接口可插拔)        │
│  记忆: AGENTS.md (项目级) + memories/ (按 repo 隔离)        │
└─────────────────────────────────────────────────────────────┘
```

**两层职责切分（关键）**：项目里存在两套“编排”能力，若不划清边界，实现时会互相打架：

| 层 | 组件 | 负责 | 不负责 |
|---|---|---|---|
| 外层 | LangGraph 状态图 | 阶段边界、检查点持久化、HITL 中断/恢复、失败与重试路由 | 单阶段内的具体工具调用 |
| 内层 | DeepAgent 运行时 | 单个阶段内的工具调用循环、子代理委派（task 工具） | 跨阶段的流程控制、状态持久化 |

一句话：**LangGraph 决定“下一步走哪个阶段”，DeepAgent 决定“这个阶段内怎么把活干完”**。`analyze/search/fix` 是 DeepAgent 的 isolated 子代理，**不是** LangGraph 的节点；LangGraph 的节点是“调用一次 DeepAgent 完成某阶段”的厚节点。

# 3. 核心章节详解

## 3.1 LangGraph 状态图设计（外层编排）

为什么需要 LangGraph：Agent 的工作流不是简单的“一次 LLM 调用 → 输出”，而是有明确的阶段边界：获取 Issue → 分析 → 搜索 → 修复 → 测试 → 提交 PR。每个阶段可能需要中断等待人工审核，也可能失败后回退重试。

状态定义（补全版）：

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class MaintenanceState(TypedDict):
    messages: Annotated[list, add_messages]  # 对话历史
    issue_title: str                          # GitHub Issue 标题
    issue_body: str                           # GitHub Issue 描述
    repo_path: str                            # 本地仓库路径
    target_files: list[str]                   # 搜索到的相关文件
    proposed_fix: str | None                  # 修复方案
    sandbox_id: str | None                    # 沙箱实例 ID（生命周期见 3.3）
    test_output: str                          # 校验输出（摘要或落盘引用）
    test_passed: bool | None                  # 校验结果
    retry_count: int                          # 回退重试计数，超限即 fail
    aborted: bool                             # 人工拒绝或错误终止标记
    error: str | None                         # 最近一次失败原因
    pr_url: str | None                        # 最终 PR 链接
```

节点划分：

| 节点 | 职责 | 是否可中断 |
|---|---|---|
| fetch_issue | 从 GitHub API 拉取 Issue 内容 | 否 |
| analyze | 调用 analyze 子代理，理解问题、提取技术关键词 | 否 |
| search | 调用 search 子代理，在代码库中定位相关文件 | 否 |
| propose_fix | 调用 fix 子代理生成修复方案 | 是（人工审核） |
| apply_fix | 在沙箱中写入修改并执行校验 | 否 |
| test_result | 根据校验结果决定继续或回退 | 否 |
| create_pr | 推送分支、创建 Pull Request | 是（人工确认） |

关键实现：

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

builder = StateGraph(MaintenanceState)
for name, fn in [
    ("fetch_issue", fetch_issue_node),
    ("analyze", analyze_node),
    ("search", search_node),
    ("propose_fix", propose_fix_node),
    ("apply_fix", apply_fix_node),
    ("test_result", test_result_node),   # 必须注册，否则是悬空节点
    ("create_pr", create_pr_node),
]:
    builder.add_node(name, fn)

builder.add_edge(START, "fetch_issue")
builder.add_edge("fetch_issue", "analyze")
builder.add_edge("analyze", "search")
builder.add_edge("search", "propose_fix")

builder.add_conditional_edges(
    "propose_fix",
    should_proceed,                       # abort / error → END
    {"apply": "apply_fix", "abort": END},
)
builder.add_edge("apply_fix", "test_result")
builder.add_conditional_edges(
    "test_result",
    test_decision,                        # 见下方路由函数
    {"pass": "create_pr", "retry": "propose_fix", "fail": END},
)
builder.add_edge("create_pr", END)

# 持久化检查点：HITL 跨进程恢复的前提，不能用 InMemorySaver
with SqliteSaver.from_conn_string("checkpoints.sqlite") as saver:
    graph = builder.compile(checkpointer=saver)
```

路由函数必须显式定义，并给 `retry` 设上限，否则状态图会死循环：

```python
MAX_RETRY = 2

def should_proceed(state: MaintenanceState) -> str:
    if state.get("aborted") or state.get("error"):
        return "abort"
    return "apply"

def test_decision(state: MaintenanceState) -> str:
    if state.get("error"):
        return "fail"
    if state.get("test_passed"):
        return "pass"
    if state.get("retry_count", 0) < MAX_RETRY:
        return "retry"
    return "fail"
```

> `retry_count` 由 `test_result` 节点在判定为 retry 时自增并写回 state；超过 `MAX_RETRY` 一律转 `fail`，绝不无限回退。

**分级验证（重要）**：技术债类任务未必都有现成测试，“测试不通过”不等于“修复错误”。`apply_fix` / `test_result` 按下列层级取证据，任一层可用即可，并记录命中的层级：

| 层级 | 判据 | 适用任务 |
|---|---|---|
| L1 | 仓库测试通过（pytest 等） | 有测试覆盖的 bug 修复 |
| L2 | 编译 / 类型检查 / lint 通过 | 依赖升级、lint 修复、补测试 |
| L3 | diff 断言：未修改 `target_files` 之外的文件 | 所有任务（兜底） |

L1 缺位时降级到 L2/L3，并在 PR 描述里**如实标注“验证层级”**，避免把 L3 通过包装成“测试通过”。

为什么用 LangGraph 而不是纯 LangChain：LangChain 的 create_agent 是一个循环，适合“工具调用直到完成”的简单模式。但夜间维护者的流程是有阶段的流水线，每个阶段有不同的上下文、不同的工具权限、不同的人工审核要求。LangGraph 的状态图 + 检查点能力正好匹配这种需求——你可以随时暂停图执行、保存状态、等人工审批后从断点恢复。

## 3.2 DeepAgent 子代理编排（内层工具循环）

为什么需要子代理：夜间维护者需要做三件上下文完全不同的事：分析 Issue（需要理解自然语言）、搜索代码（需要 ripgrep/glob）、生成修复（需要读写文件 + 执行测试）。如果用一个 Agent 串行做所有事，主对话历史会被大量中间结果污染。

**子代理的归属**：`analyze/search/fix` 是 DeepAgent 的 `subagents`，由**某个 LangGraph 节点内部**调用主代理、再由主代理通过 `task` 工具委派。它们不是 LangGraph 节点——LangGraph 只看到“某阶段完成了”这个结果，看不到子代理内部的工具调用。

DeepAgent 的子代理机制：DeepAgent 允许主代理通过 task 工具将工作委派给子代理。子代理在独立的上下文窗口中执行，只返回简洁的最终结果给主代理。

子代理定义（示意伪代码，`deepseek_model` 等以配置注入）：

```python
from deepagents import create_deep_agent

main_agent = create_deep_agent(
    model=MAIN_MODEL,
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

**子代理间的信息注入契约**（isolated 模式的落地方式）：既然子代理不继承历史，主代理必须显式传上下文。为防上下文爆炸，**只传“精炼引用”而非全文**：

- `analyze → search`：传关键词列表 + 涉及模块，不传 Issue 全文。
- `search → fix`：只传**命中的文件路径 + 符号名 + 行区间**（如 `src/app/config.py:42-58`），**不塞文件正文**；`fix` 子代理在沙箱内自行 `read_file` 按需读取。
- 注入 `fix` 的文件数设硬上限（如 5），超出截断并在日志标注被丢弃项。

**模型路由表**（成本与质量分层，夜间批量跑的关键）：

| 角色 | 建议档位 | 理由 |
|---|---|---|
| 主代理（编排/调度） | 中等档（Sonnet 级） | 需稳定遵循委派协议 |
| analyze | 便宜档（Haiku 级） | 结构化抽取，任务简单 |
| search | 便宜档 | 关键词检索，可规则兜底 |
| fix | 强档（Sonnet/Opus 级） | 代码正确性最关键，值得花钱 |
| test_result 判定 | 不用 LLM | 纯规则判定（见 3.1 路由函数） |

> 具体模型名随供应商更新，实现时以**配置项注入**，不写死在代码里。

## 3.3 沙箱执行与安全隔离

为什么需要沙箱：代码修复 Agent 需要执行 npm install、pytest、cargo build 等命令。如果直接在宿主机运行，一个错误的 rm -rf 或恶意依赖脚本可能破坏开发环境。

方案选择：上层代码只依赖统一的 `SandboxBackend` 接口，**默认实现为 `E2BSandboxBackend`**——云端一次性沙箱，隔离级别高、用完即销毁，不污染宿主机。接口与实现解耦，后续如需更换隔离方案，只需替换 `backend=` 的实现类，上层编排与工具代码零改动：

E2BSandboxBackend 配置（换后端只需替换 `backend=` 传参）：

```python
from e2b import Sandbox
from deepagents import create_deep_agent
from langchain_e2b import E2BSandbox

e2b_sandbox = Sandbox.create()
backend = E2BSandbox(sandbox=e2b_sandbox)

agent = create_deep_agent(
    model=MAIN_MODEL,
    system_prompt="You are a Python coding assistant with sandbox access.",
    backend=backend,
)

try:
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Create a small Python package and run pytest",
                }
            ]
        }
    )
    for msg in result["messages"]:
        msg.pretty_print()

finally:
    e2b_sandbox.kill()  # 销毁沙箱
```

**⚠️ 代码外发风险（必须正视）**：默认后端 E2B 是**云端**沙箱，意味着目标仓库的代码会被上传到第三方服务执行。这是企业采用的**第一道否决线**，必须显式声明：

- README 与文档中**明确写出**“代码将离开本地、在第三方云沙箱运行”。
- 提供两条规避路径：`--dry-run`（不建沙箱、不调用远端），以及**可插拔的本地后端**（替换 `SandboxBackend` 实现即可，接口不变）。
- 私有仓库、含密钥的仓库**默认不建议**使用云端后端。

**沙箱生命周期管理**（与检查点强耦合）：`sandbox_id` 会被写入 checkpoint，但沙箱是**外部资源**，可能在中断/等待审核期间被 E2B 回收或超时。规则如下：

1. **懒创建**：首次进入 `apply_fix` 时才创建沙箱，不在图启动时创建。
2. **resume 时校验**：恢复执行（人审之后）先探测 `sandbox_id` 是否存活；不存活则**重建沙箱并从仓库快照恢复工作区**。
3. **重试即重置**：`test_result` 触发 `retry` 回到 `propose_fix` 时，**丢弃当前沙箱、按干净快照重建**，避免上一次的脏改动污染重试。
4. **终止必清理**：图走到 `END`（含 `fail`/`abort`）时在收尾节点统一 `kill()`；异常路径也要保证清理（context manager 或 finally）。

安全原则：沙箱只应挂载仓库的克隆副本，绝不挂载 ~/.ssh、~/.aws、.env 文件或宿主机 Docker socket。Agent 的 execute 工具即使被 prompt injection 攻击，也最多只能破坏一个一次性的容器。

## 3.4 项目级记忆（AGENTS.md）

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
- 依赖管理：conda（用 environment.yml 声明环境，不用 pip install）

## 代码规范
- 使用 snake_case 命名
- 所有公共函数必须有 type hints
- 禁止直接修改 migrations/ 目录

## 测试命令
- 单元测试：`conda run -n my-service pytest tests/unit/ -x`
- 集成测试：`conda run -n my-service pytest tests/integration/ -x --timeout=60`

## 已知陷阱
- `config.py` 中的 `LEGACY_MODE` 默认为 True，修改前需要先确认
- 数据库迁移只能在测试环境中运行
```

**自动记忆的存储位置（按 repo 隔离）**：Agent 运行中自动积累的记忆**不能写进一个全局大目录**，否则 A 项目的经验会污染 B 项目。改为按仓库分桶：

- 路径：`~/.deepagents/<agent_name>/memories/<repo_key>/`，`repo_key` 取 `owner__name` 或仓库路径 hash。
- 更推荐：直接落在目标仓库的 `.deepagents/memories/` 下，随仓库走、可 review、可提交。
- 每次运行只加载当前 `repo_key` 下的记忆 + 全局通用记忆；两层分层，不混袋。

## 3.5 工具集设计

Coding Agent 的工具不在于“多”，而在于精确和最小化。每个工具都应该有清晰的输入输出契约。

| 工具 | 用途 | 权限范围 |
|---|---|---|
| ripgrep_search | 在代码库中搜索文本模式 | 只读，作用域绑定到 `repo_path`（非模型参数） |
| read_file | 读取文件内容 | 只读，服务端校验路径在 `repo_path` 内 |
| write_file | 写入修复后的文件 | 写权限，服务端校验在沙箱 `/workspace/` 内且属于 `target_files` |
| execute_shell | 在沙箱中运行命令 | 受限：白名单命令（pytest, conda, npm） |
| create_github_pr | 创建 Pull Request | 需要 GitHub Token，人工确认后触发 |

工具实现示例（**路径不交给模型**）：

```python
from langchain.tools import tool

def make_ripgrep_search(repo_path: str):
    """用闭包把 repo_path 绑定为受信配置，绝不出现在模型可传参数里。"""
    @tool
    def ripgrep_search(pattern: str) -> str:
        """在代码库中搜索文本模式。返回匹配的文件路径和行号。"""
        import subprocess
        result = subprocess.run(
            ["rg", "--line-number", "--max-count", "50", pattern, repo_path],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout[:5000]  # 截断，避免上下文爆炸
    return ripgrep_search
```

**权限必须由服务端强制，而非依赖模型自觉**：`repo_path`、写目标路径都**不该出现在工具签名里**让模型填；`write_file` 在实现内部校验 `target_path` 是否在 `/workspace/` 且属于 `target_files`，越界直接 `raise`，不落到沙箱。

为什么不用通用 execute 工具：DeepAgent 的 LocalShellBackend 自带一个 execute 工具，但它允许任意 shell 命令。夜间维护者应该显式定义白名单工具，而不是给 Agent 一个万能命令执行器。这既降低了安全风险，也让 Agent 的行为更可预测。

## 3.6 人工审核门控（HITL）

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
        "fix_plan": fix_plan,          # 必须是可 JSON 序列化的摘要/结构体
        "target_files": state["target_files"],
        "question": "是否批准此修复方案？回复 'approve' 或提供修改意见。",
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

> 两点约束：① `interrupt()` 的载荷会随 checkpoint 落盘，**必须可 JSON 序列化**，因此 `fix_plan` 传结构化摘要，不要把大对象或不可序列化的中间态塞进去；② `interrupt()` 的返回值就是 `Command(resume=...)` 传入的值，`should_proceed`（见 3.1）据 `aborted` 字段决定走 `apply` 还是 `abort`。

# 4. 与现有方案的差异化

| 维度 | OpenHands | OSC-Agent | 夜间维护者（本方案） |
|---|---|---|---|
| 定位 | 通用自主软件工程师 | 开源贡献 CLI | 定时/CI 触发的技术债清理 |
| 交互模式 | 对话式 + 自主执行 | CLI 命令 | 无头模式 + 人工审核门控 |
| 沙箱 | Docker 全环境 | E2B 云端 | E2B 云端（SandboxBackend 可插拔） |
| 子代理 | 未明确分层 | 多 Agent 但非 DeepAgent | DeepAgent isolated 子代理 |
| 记忆 | microagents（人工编写、静态） | 无 | AGENTS.md + **Agent 自动积累**（按 repo 隔离） |
| 目标用户 | 开发者手动使用 | 开源贡献者 | CI 系统 / 团队夜间自动化 |

# 5. 开发路线图

## 5.0 路线图总览

路线图按“能力自内向外逐层加固”的顺序推进：先让单个 Agent 能在沙箱里改对一行代码（最小可信闭环），再叠加搜索、编排、门控、集成，最后才做开源包装。每周都是一个可独立演示、可验收的里程碑，前一周的产出物是后一周的输入，任何一周的验收标准不达标就停下来修，不带着技术债往下走。

| 阶段 | 周次 | 核心目标 | 关键交付物 | 里程碑验收 |
|---|---|---|---|---|
| P0 | 准备（0.5 周） | 环境与脚手架 | CLI 骨架 + SandboxBackend 抽象 + seed 评测集 + tracing | `nw-agent --help` 可跑通，沙箱能起停，评测脚本能跑 |
| P1 | 第 1 周 | 骨架跑通（含最小权限） | 单 Agent + 沙箱改写 demo + 服务端路径校验 | 沙箱内读文件→改→出 diff，越界写入被拒 |
| P2 | 第 2 周 | 搜索与定位 | `analyze` / `search` 子代理 + 工具白名单 | Issue 文本 → 命中正确目标文件 |
| P3 | 第 3 周 | LangGraph 编排 | 状态图 + **持久化检查点** + 分级验证 | 跨进程中断后可从断点恢复，节点流可回放 |
| P4 | 第 4 周 | 安全加固与 HITL 门控 | HITL `interrupt` 门控 + 注入回归 | 应用修复前暂停等人审，注入用例全拦截 |
| P5 | 第 5 周 | GitHub 集成与记忆 | `create_github_pr` + 按 repo 隔离的 AGENTS.md | 完整 Issue → PR 闭环 |
| P6 | 第 6 周+ | 开源准备 | README / CI / 演示 | 陌生人可一键复现 |

**贯穿全程的三条纪律**：
1. **安全左移**：最小权限（`write_file` 白名单、路径服务端校验）从 P1 第一天就定死，不等 P4 补。
2. **评测集先行**：P0 就建 seed 评测集（哪怕 3 条），每周追加，作为 5.7 度量表的唯一数据源。
3. **沙箱与工具边界只收紧、不放松**：任何时候发现越权路径，先堵死再继续功能。

---

## 第0阶段：环境与脚手架（准备，约 0.5 周）

**目标**：把后续所有阶段依赖的地基一次性搭好，避免后面边写业务边改脚手架。

**任务清单**：
- 初始化 Python 项目（用 `conda` 管理环境与依赖，`environment.yml` 锁定版本），目录分层：`src/nw_agent/{cli,graph,agents,tools,backends,memory}`。
- 定义 CLI 入口 `nw-agent`，参数契约：`--repo owner/name`、`--issue 123`、`--task "自然语言"`、`--dry-run`、`--no-hitl`、`--trace`。
- 抽象 **SandboxBackend 接口**（见 3.3 节），默认实现 `E2BSandboxBackend`。接口与实现解耦——上层只依赖接口，后续可插拔替换其他隔离方案而无需改动编排与工具代码。
- 落实最小权限骨架：工具签名**不含**路径参数，`write_file` 白名单校验在实现层。
- **建 seed 评测集**：先攒 3 条 golden issue（含 1 条恶意注入用例），写成一个可重复运行的评测脚本。
- **接可观测性**：接入 LangSmith tracing（或本地结构化 JSON 日志）+ 每节点状态快照落盘。
- 配置密钥读取：`GITHUB_TOKEN`、模型 API Key 统一走环境变量，**禁止写入仓库**，`.gitignore` 覆盖 `.env`。
- 搭 `pytest` 最小测试骨架 + `ruff`/`mypy` 基础检查（自举：这个项目自己也要能被 NightWatch 维护）。

**验收标准**：
- `nw-agent --help` 打印完整参数说明。
- 一段 10 行脚本能创建并销毁一个沙箱，并在其中执行 `echo hello`。
- 评测脚本能跑出首个基线数字；CI（本地 hook 即可）能跑通 `ruff check` 与 `pytest`。

**依赖与风险**：E2B 属云端依赖（需联网 + API Key），成本与可用性受外部服务影响，且存在代码外发问题（见 3.3）。**决策：以 `E2BSandboxBackend` 为默认后端，同时通过 `SandboxBackend` 接口保持可插拔——若后续需要离线或自托管隔离，替换实现类即可，不阻塞主流程。**

---

## 第1周：骨架跑通（最小可信闭环）

**目标**：证明“Agent 能在隔离环境里安全地改对代码”，这是整个项目的地基假设。

**任务清单**：
- 用 `create_deep_agent` 装配主代理，**只给一个 `fix` 子代理**（见 3.2 节），跑通 `main → task(fix) → 返回` 的最短链路。
- 落地三个最小工具（见 3.5 节）：`read_file`、`write_file`（限定沙箱 `/workspace/`）、`execute_shell`（白名单：`pytest`、`conda`、`npm`），**路径校验在服务端**。
- 写死一个窄任务：“给指定函数补充 docstring”，用 3~5 个真实函数作为 golden case。
- 实现 diff 输出：修复后 `git diff` 展示改动，供人工肉眼判断对错。

**交付物**：`nw-agent --repo <local> --task "给 foo() 补 docstring"` 能端到端跑出 diff 的一段 demo。

**验收标准**：
- golden 用例中至少 4/5 正确补充 docstring，且**未修改任务外文件**（用 `git status` 断言改动文件数 == 1）。
- **越界写入被拒**：构造一个写 `target_files` 之外路径的调用，断言被服务端拒绝。
- 沙箱销毁后宿主机工作区零污染。
- 单次任务端到端 < 2 分钟，记录 token 消耗基线。

**依赖与风险**：
- 风险：模型自由发挥改了多个文件。**规避**：`write_file` 只允许写 `target_files`，其余路径直接拒绝。
- 风险：白名单命令不够用导致任务失败。**规避**：白名单做成配置文件，但新增命令需显式 review，不默认放开。

---

## 第2周：搜索与定位（Issue → 目标文件）

**目标**：把“人告诉它改哪个函数”升级为“给它一个 Issue 描述，它自己找到该改哪”。

**任务清单**：
- 实现 `ripgrep_search` 工具（见 3.5 节示例），结果截断到 5000 字符防止上下文爆炸。
- 实现 `analyze` 子代理：读 Issue → 输出【问题类型 / 涉及模块 / 建议搜索关键词】。
- 实现 `search` 子代理：用关键词多轮 ripgrep → 返回**最相关的 Top-5 文件路径 + 匹配行**。
- 主代理作为调度者，把 analyze 的关键词传给 search，把 search 的**路径/行号**传给 fix（利用子代理 isolated 特性，见 3.2 节的信息注入契约）。
- 扩充定位评测集：整理 10 个真实 Issue，人工标注“正确目标文件”。

**交付物**：`analyze → search → fix` 三子代理链路，输入 Issue 文本直接产出 diff。

**验收标准**：
- 10 个评测 Issue 中，Top-5 命中率 ≥ 7/10（正确文件出现在 search 结果里）。
- 搜索耗时单次 < 30 秒；子代理之间不共享完整对话历史（抽查 prompt 确认上下文未被污染）。
- 关键词提取可解释：`analyze` 输出结构化 JSON，而非一段自由文本。

**依赖与风险**：
- 风险：Issue 描述模糊、代码库大，Top-5 命中率低。**规避**：先用小仓库（如本仓库自身）做评测，命中率稳定后再上大仓库；必要时增加一轮“关键词扩展”重试。
- 风险：搜索结果里塞太多文件把 fix 带偏。**规避**：硬性限制注入 fix 的文件数，超出截断并在日志里标注被丢弃项。

---

## 第3周：LangGraph 编排（从线性脚本到状态图）

**目标**：把第 2 周的“一条链”重写为 3.1 节定义的 `MaintenanceState` 状态图，获得可中断、可恢复、可回放的执行流。

**任务清单**：
- 定义 `MaintenanceState`（含 `retry_count` / `aborted` / `test_output` / `error`，见 3.1 节补全版）。
- 实现节点：`fetch_issue`、`analyze`、`search`、`propose_fix`、`apply_fix`、`test_result`、`create_pr`，并**全部在 builder 中注册**。
- 接两组条件边：`propose_fix → {apply | abort}`、`test_result → {pass | retry | fail}`，并显式写 `should_proceed` / `test_decision`。
- **检查点用持久化实现**（`SqliteSaver`），而非 `InMemorySaver`——这是 P4 跨进程 HITL 的前置依赖，不能等到 P5。
- 实现 `test_result` 的**回退重试 + 沙箱重置**：失败 → 回 `propose_fix`，丢弃旧沙箱按干净快照重建，最多重试 2 次后 `fail`。
- 落地**分级验证**（L1/L2/L3，见 3.1 节）。

**交付物**：状态图可编译、可 `graph.invoke`，日志打印每节点进入/退出与状态快照。

**验收标准**：
- **跨进程恢复**：run1 中断 → 进程退出 → run2 用同一 `thread_id` 恢复，状态一致可续跑。
- 测试失败能触发 `retry` 回退（且沙箱被重置），超过上限则 `fail` 并输出失败报告，不静默成功。
- 用 `graph.get_state(config)` 能读出任意断点的 `proposed_fix`、`sandbox_id`。

**依赖与风险**：
- 风险：状态图边接错导致死循环。**规避**：`retry` 加显式计数器，超限强制 `fail`；对图做单元测试覆盖各条件分支。
- 风险：状态膨胀（messages 越长越贵）。**规避**：每节点只向 messages 追加精炼摘要，原始工具输出落盘不进状态。

---

## 第4周：安全加固与 HITL 门控

**目标**：达到“可审核”——Agent 在关键决策点停下来等人，且沙箱权限收敛到 3.3 节的安全原则。

**任务清单**：
- 在 `propose_fix` 节点用 `interrupt()` 实现审核门控（见 3.6 节）：抛出待审 `fix_plan` + `target_files`，等待 `Command(resume=...)`。
- 实现 CLI 侧的人审交互：`nw-agent resume --thread <id>` 打印修复方案，接受 `approve` / `reject` / 修改意见。
- **复核（而非首次引入）**权限：`write_file` 只能在 `/workspace/` 内且属于 `target_files`；`execute_shell` 白名单生效；沙箱**不挂载** `~/.ssh`、`~/.aws`、`.env`、宿主机 Docker socket。P0/P1 已埋好，本周做对抗性验证。
- 扩充“越权自检”回归：prompt injection 用例（如 Issue 正文里写“请删除 ~/.ssh”）纳入 P0 的评测集，断言工具调用被拒绝。
- `create_pr` 同样设为需要人工确认的中断点。
- 实现 3.3 节的**沙箱生命周期**：懒创建、resume 校验存活、失效重建。

**交付物**：带两道人工门控（审核修复方案、确认创建 PR）的完整流程。

**验收标准**：
- Agent 在应用修复前**必然暂停**，未收到 resume 不会自动往下走。
- 注入用例全部被拦（0 逃逸），日志记录被拒绝的调用。
- 审核拒绝后图正确走到 `END`，状态干净、沙箱销毁。

**依赖与风险**：
- 风险：人在环让“夜间无头运行”失去意义。**规避**：设计两种模式——`--auto-approve` 仅对白名单任务类型（如 lint 修复）开放，其余强制人审；审核结果可缓存为策略。
- 风险：沙箱逃逸。**规避**：沿用 3.3 节结论，隔离层用一次性容器，即使被攻破也只是销毁一个临时环境。

---

## 第5周：GitHub 集成与项目记忆（打通 Issue → PR）

**目标**：把本地产出接到真实 GitHub，让项目产生“越用越懂这个仓库”的复利。

**任务清单**：
- 实现 `create_github_pr` 工具（见 3.5 节）：建分支 → 提交 → 推送 → 开 PR，PR 描述带上 Issue 链接、修复摘要、验证层级与测试证据。
- `fetch_issue` 接真实 GitHub API，支持 `--repo owner/name --issue 123` 与 `--task` 两种入口。
- 生成并落盘项目级 `.deepagents/AGENTS.md`（见 3.4 节）：架构、测试命令、代码规范、已知陷阱四段式。
- 开启自动记忆积累，**按 repo 隔离**（见 3.4 节）：写入 `.deepagents/memories/<repo_key>/`，下次运行先检索再动手。

**交付物**：`nw-agent --repo owner/name --issue 123` 一键跑通 Issue → PR。

**验收标准**：
- 在一个真实仓库上完成端到端闭环，产出可合并质量的 PR（含验证证据）。
- 记忆生效验证：第二次跑同类 Issue 时，Agent 复用 AGENTS.md 中的约定（如自动用 `conda run -n my-service pytest` 而非裸 `pytest`），探索步数下降；且换仓库时**不加载**其他仓库的记忆。
- PR 只在人工确认后创建；`--dry-run` 不触碰远端。

**依赖与风险**：
- 风险：Token 权限过大造成误操作。**规避**：使用**最小权限 GitHub Token**（仅目标仓库 repo 权限），不授予组织级/多仓库权限。
- 风险：自动记忆写入错误结论污染后续运行。**规避**：记忆文件可 review、可删除；写入前做冲突检查，与 AGENTS.md 冲突时以项目文件为准。

---

## 第6周+：开源准备与持续运营

**目标**：让陌生人快速复现并信任这个工具。

**任务清单**：
- 编写 README：定位一句话、架构图（复用第 2 章）、**代码外发声明**（见 3.3）、`--dry-run` 零密钥体验路径、三个真实 demo（docstring / lint 修复 / 依赖升级）。
- 配 GitHub Actions：定时触发（夜间 cron）跑维护任务；CI 里跑本项目的 `ruff` + `pytest` + 评测集。
- 录制 3~5 分钟演示视频：从 Issue 到 PR 的完整过程，含人审暂停点。
- 补齐 `CONTRIBUTING.md`、Issue/PR 模板、LICENSE、`AGENTS.md` 示例。
- 发布 v0.1.0，建立 Issue 反馈闭环（用 NightWatch 自己处理自己仓库的 good-first-issue）。

**交付物**：公开仓库 + 可复现 demo + 自动化运行示例。

**验收标准**：
- 在一台干净机器上按 README 步骤 ≤ 10 分钟跑通首个 demo。
- Actions 夜间任务至少成功产出一次真实 PR。
- **零密钥路径**：无 API Key 的用户可用 `--dry-run` 走完非沙箱流程（生成方案、展示 diff 预览），不建沙箱、不调远端。

**依赖与风险**：
- 风险：上手门槛高（配置 Token / Key 劝退）。**规避**：`--dry-run` + 可插拔本地后端双路径，零密钥也能体验。
- 风险：成本失控。**规避**：模型路由分层（见 3.2 节）；子代理按需调用；README 明示单次运行的大致 token/费用区间。

---

## 5.7 全程贯穿的度量与回归

路线图不是“做完功能就完事”，每阶段都要往同一张表里加数据，作为是否可进入下一阶段的硬门槛：

| 指标 | 采集时点 | 目标方向 |
|---|---|---|
| 端到端耗时 | 每周 | 下降或持平 |
| 单任务 token 成本 | 每周 | 下降或持平 |
| 目标文件 Top-5 命中率 | 第 2 周起 | 上升 |
| 端到端成功率（golden issue） | 第 0 周起 | 上升 |
| 越权/注入逃逸次数 | 第 0 周起 | 恒为 0 |
| 人工审核一次通过率 | 第 4 周起 | 上升（衡量修复质量） |

**回归机制**：维护一组固定 golden issue + 固定仓库快照，每次改动后重跑，命中率或成功率跌破阈值即视为回归，先修复再继续。这套评测集本身就是项目最值钱的资产之一——它把“感觉能用”变成“可量化的可信”。

# 6. 关键风险与规避

| 风险 | 规避策略 |
|---|---|
| Agent 修改了不相关的文件 | propose_fix 后加人工审核；write_file 只能操作 target_files（服务端强制） |
| 代码外发到第三方云沙箱 | 显式声明；提供 `--dry-run` 与可插拔本地后端；敏感仓库禁用云端后端 |
| 沙箱逃逸 | 使用 E2B 云端一次性沙箱隔离，绝不挂载宿主机敏感目录 |
| 沙箱在中断期被回收 | 懒创建 + resume 时校验存活性，失效则按快照重建（见 3.3） |
| LLM 幻觉导致错误修复 | 沙箱内分级验证（L1/L2/L3）；不通过则回退并报告 |
| 上下文爆炸 | 子代理 isolated；注入只传路径/行号；工具结果截断（5000 字符） |
| Token 成本失控 | 模型路由分层（见 3.2）；子代理按需调用 |
| 记忆跨仓库污染 | 记忆按 repo 隔离（见 3.4） |

一句话总结：这个项目的核心价值不在于“让 AI 写代码”，而在于用 DeepAgent 的架构能力（子代理隔离 + 沙箱 + 记忆）构建一个可控、可审核、可复现的自动化维护流水线。范围越窄（只做“夜间维护”），完成度越高，在 GitHub 上被 star 和复用的概率越大。
