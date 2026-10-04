**代码仓库“夜间维护者”（Coding Agent）**

> 一句话：CI/定时触发的无头 CLI Agent，把边界清晰的技术债 Issue 变成**待人工确认的 PR 草案**——“Issue → 补丁 → 沙箱验证 → PR 草案”全链路可审核、可复现。

# 1. 场景定义与边界

**做什么**：一个可在本地或 CI 中无头运行的 CLI Agent，接收 GitHub Issue 编号或自然语言任务描述，自动完成从“理解需求 → 定位代码 → 生成补丁 → 沙箱验证 → 创建 PR”的全流程。Agent 的编排与调度在本地/CI 进程内运行，代码执行与测试则放入隔离沙箱（默认云端 E2B，见 3.3）。

**定位澄清（异步 HITL）**：本项目不是“夜里无人值守直接合并代码”，而是 **“夜间产出 PR 草案，清晨人工确认后合并”**。无头运行的是“产出草图”这段；人工审核门控（3.6）是异步的——夜里跑到门控处挂起并留档，白天人来 approve/reject。这样“无头”与“人审”不再冲突。

**不做什么**：不追求“全自动重构整个系统”这类模糊需求。MVP 阶段只处理边界清晰的技术债类任务，例如依赖升级、补充单元测试、修复 lint 错误、处理简单的 bug 报告；范围外的 Issue 由 `classify` 节点拒单（见 3.1）。

**为什么值得做**：OpenHands 已有 77K+ stars，但它的定位是“通用自主软件工程 Agent”，体量大、部署复杂。你的差异化在于做“夜间维护”这个极窄切片——不需要交互式 UI，不需要多轮对话，只需要在 CI 或定时任务中跑通“Issue → PR 草案”的闭环。

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
│  流程: fetch→classify→agent→review→apply→test→pr            │
│  检查点: 持久化 checkpointer (SQLite)，支持跨进程恢复       │
└─────────────────────────────────────────────────────────────┘
                          │ run_agent 节点内一次性调用
                          ▼
┌─────────────────────────────────────────────────────────────┐
│     DeepAgent 运行时 (create_deep_agent) ── 管「阶段内循环」│
│  职责: 单次运行内跑完 理解→定位→出补丁（主代理用 task 编排）│
│  ┌─────────┐   ┌─────────┐   ┌─────────┐                    │
│  │ analyze │ → │ search  │ → │  fix    │  ← isolated 子代理 │
│  └─────────┘   └─────────┘   └─────────┘                    │
│  工具: ripgrep_search, read_file, write_file, execute_shell │
│  后端: E2BSandboxBackend (SandboxBackend 接口可插拔)        │
│  记忆: AGENTS.md (项目级) + memories/ (按 repo 隔离)        │
└─────────────────────────────────────────────────────────────┘
```

**两层职责切分（关键）**：项目里存在两套“编排”能力，必须划清边界，否则同一段流程会被编排两遍：

| 层 | 组件 | 负责 | 不负责 |
|---|---|---|---|
| 外层 | LangGraph 状态图 | 阶段边界、检查点持久化、HITL 中断/恢复、失败重试与预算熔断 | 阶段内的具体工具调用与子代理调度 |
| 内层 | DeepAgent 运行时 | **单次运行内**跑完 理解→定位→出补丁（子代理 isolated 委派） | 跨阶段的流程控制、状态持久化 |

一句话：**LangGraph 决定“下一步走哪个阶段”，DeepAgent 决定“本次运行内怎么把活干完”**。关键取舍是——**`analyze/search/fix` 由 DeepAgent 在一次运行内按序委派完成，不再由 LangGraph 逐阶段驱动**。否则“阶段顺序”会同时被 LangGraph 的边和主代理的 task 工具决定，等于编排两遍。LangGraph 的 `run_agent` 节点是“调用一次 DeepAgent”的**厚节点**。

# 3. 核心章节详解

## 3.1 LangGraph 状态图设计（外层编排）

为什么需要 LangGraph：Agent 的工作流不是简单的“一次 LLM 调用 → 输出”，而是有明确的阶段边界：获取 Issue → 判定范围 → 理解与修复 → 人工审核 → 应用与验证 → 创建 PR 草案。每个阶段可能需要中断等待人工审核，也可能失败后回退重试。

状态定义（补全版）：

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class MaintenanceState(TypedDict):
    messages: Annotated[list, add_messages]  # 跨阶段的精炼事件流（非子代理原始历史）
    issue_title: str                          # GitHub Issue 标题
    issue_body: str                           # GitHub Issue 描述
    repo_path: str                            # 本地仓库路径
    in_scope: bool                            # classify 判定：是否属于 MVP 范围
    target_files: list[str]                   # 搜索命中的相关文件
    proposed_fix: str | None                  # 生成的补丁（diff/结构化摘要）
    sandbox_id: str | None                    # 沙箱实例 ID（生命周期见 3.3）
    test_output: str                          # 校验输出（摘要或落盘引用）
    test_passed: bool | None                  # 校验结果
    retry_count: int                          # 回退重试计数，超限即 fail
    token_used: int                           # 已消耗 token（预算熔断用）
    token_budget: int                         # 单任务 token 预算上限
    aborted: bool                             # 人工拒绝或错误终止标记
    error: str | None                         # 最近一次失败原因
    pr_url: str | None                        # 最终 PR 链接
```

节点划分：

| 节点 | 职责 | 是否可中断 |
|---|---|---|
| fetch_issue | 从 GitHub API 拉取 Issue 内容 | 否 |
| classify | 判定是否属于 MVP 范围（范围外直接拒单） | 否 |
| run_agent | 调用 DeepAgent 一次，内部跑完 analyze→search→fix，产出补丁 | 否 |
| human_review | 人工审核门控（节点入口即 `interrupt`） | 是（人工审核） |
| apply_fix | 在沙箱中应用补丁并执行校验 | 否 |
| test_result | 根据校验结果决定继续、回退或终止 | 否 |
| create_pr | 推送分支、创建 Pull Request 草案 | 是（人工确认） |

> `classify` 用便宜模型输出结构化判定（是否 in-scope + 任务类型，映射见[开发路线图](开发路线图.md) 5.8），**默认保守**：判不准时归入“需人审”而非自动放行，宁可多审不可误放。

关键实现：

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

builder = StateGraph(MaintenanceState)
for name, fn in [
    ("fetch_issue", fetch_issue_node),
    ("classify", classify_node),
    ("run_agent", run_agent_node),        # 厚节点：内部跑一次 DeepAgent
    ("human_review", human_review_node),
    ("apply_fix", apply_fix_node),
    ("test_result", test_result_node),
    ("create_pr", create_pr_node),
]:
    builder.add_node(name, fn)

builder.add_edge(START, "fetch_issue")
builder.add_edge("fetch_issue", "classify")
builder.add_conditional_edges(
    "classify", scope_gate,               # 越界拒单
    {"in_scope": "run_agent", "out_of_scope": END},
)
builder.add_edge("run_agent", "human_review")
builder.add_conditional_edges(
    "human_review", review_decision,      # reject / error → END
    {"approve": "apply_fix", "reject": END},
)
builder.add_edge("apply_fix", "test_result")
builder.add_conditional_edges(
    "test_result", test_decision,
    {"pass": "create_pr", "retry": "run_agent", "fail": END},
)
builder.add_edge("create_pr", END)

# 关键：saver 的生命周期必须覆盖整个运行期（含跨进程 resume）。
# 用 with 时，所有 invoke / 等待人审 / resume 都必须在块内，别 compile 完就退出块。
with SqliteSaver.from_conn_string("checkpoints.sqlite") as saver:
    graph = builder.compile(checkpointer=saver)
    run_cli(graph)                        # CLI 主流程都在这里
```

路由函数必须显式定义，并给 `retry` 与预算设上限，否则状态图会死循环或夜间失控烧钱：

```python
MAX_RETRY = 2

def scope_gate(state: MaintenanceState) -> str:
    return "in_scope" if state.get("in_scope") else "out_of_scope"

def review_decision(state: MaintenanceState) -> str:
    if state.get("aborted") or state.get("error"):
        return "reject"
    return "approve"

def test_decision(state: MaintenanceState) -> str:
    # 预算熔断优先：超预算一律终止，避免单任务失控
    if state.get("token_used", 0) > state.get("token_budget", 0):
        return "fail"
    if state.get("error"):
        return "fail"
    if state.get("test_passed"):
        return "pass"
    if state.get("retry_count", 0) < MAX_RETRY:
        return "retry"
    return "fail"
```

> `retry_count` 由 `test_result` 节点在判定为 retry 时自增并写回 state；超过 `MAX_RETRY` 一律转 `fail`。`token_used` 由 `run_agent` 节点累加。

**分级验证**：技术债类任务未必都有现成测试，“测试不通过”不等于“修复错误”。验证分两层：**正确性证据**（L1/L2，至少满足其一）与**强制护栏**（L3，永远必须满足）：

| 层级 | 判据 | 性质 | 适用任务 |
|---|---|---|---|
| L1 | 仓库测试通过（pytest 等） | 正确性证据（优先） | 有测试覆盖的 bug 修复 |
| L2 | 编译 / 类型检查 / lint 通过 | 正确性证据（最低要求） | 依赖升级、lint 修复、补测试 |
| L3 | 未修改 `target_files` 之外的文件 | **强制护栏（AND）** | 所有任务 |

**L3 只证明“没乱改”，不证明“改对了”**——一个空改动也能通过 L3。因此它不能单独作为通过依据：`test_passed = (L1 或 L2) 且 L3`。PR 描述里要**如实标注命中的验证层级**，避免把仅有护栏通过包装成“测试通过”。

为什么用 LangGraph 而不是纯 LangChain：LangChain 的 create_agent 是一个循环，适合“工具调用直到完成”的简单模式。但夜间维护者的流程是有阶段的流水线，每个阶段有不同的上下文、不同的工具权限、不同的人工审核要求。LangGraph 的状态图 + 检查点能力正好匹配这种需求——你可以随时暂停图执行、保存状态、等人工审批后从断点恢复。

## 3.2 DeepAgent 子代理编排（内层工具循环）

为什么需要子代理：夜间维护者需要做三件上下文完全不同的事：分析 Issue（需要理解自然语言）、搜索代码（需要 ripgrep/glob）、生成补丁（需要读写文件）。如果用一个 Agent 串行做所有事，主对话历史会被大量中间结果污染。

**子代理的归属与编排**：`analyze/search/fix` 是 DeepAgent 的 `subagents`，由主代理在**一次运行内**通过 `task` 工具按 `analyze → search → fix` 顺序委派完成。**LangGraph 不介入这一步**——它只在 `run_agent` 节点调用一次 DeepAgent，拿回最终补丁。这样“阶段顺序”只有 DeepAgent 一个决定者，避免与 LangGraph 的边重复编排。

DeepAgent 的子代理机制：DeepAgent 允许主代理通过 task 工具将工作委派给子代理。子代理在独立的上下文窗口中执行，只返回简洁的最终结果给主代理。

子代理定义（示意伪代码，模型名以配置注入）：

```python
from deepagents import create_deep_agent

main_agent = create_deep_agent(
    model=MAIN_MODEL,
    system_prompt="""你是代码仓库维护编排器。
    在一次运行内按顺序使用 task 工具完成：
    1) analyze: 理解 Issue 的技术需求
    2) search: 在代码库中定位相关文件
    3) fix: 生成并验证补丁
    完成后返回补丁（diff）与验证摘要。不要自己直接搜索或修改文件。""",
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
            "description": "根据分析结果和搜索结果，生成代码补丁",
            "system_prompt": "你是修复工程师。基于提供的上下文修改文件，产出清晰的 diff，并自检是否只动了目标文件。",
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
from deepagents import create_deep_agent
from nw_agent.backends import create_backend

# 当前唯一后端：云端 E2B（需 E2B_API_KEY）
backend = create_backend("e2b")

with backend:                       # 退出即 kill()，异常路径也兜得住
    agent = create_deep_agent(
        model=MAIN_MODEL,
        system_prompt="You are a Python coding assistant with sandbox access.",
        backend=backend,            # 换隔离方案只需换这一处的实现类
    )
    result = agent.invoke({...})
```

> **接口落地的三个取舍**（实现见 `src/nw_agent/backends/`）：
>
> 1. `SandboxBackend` **继承** deepagents 的 `BaseSandbox`，而非自带 `typing.Protocol`——
>    deepagents 判定后端能力用的是**类属性身份比对**（`type(self).ls_info is not
>    BackendProtocol.ls_info`），鸭子类型通不过。代价是 `deepagents` 必须进运行时依赖。
> 2. 「五动作」中的**执行命令**直接采用 deepagents 的 `execute(command, *, timeout)`
>    签名，不另立一套；**创建**做成模块级工厂 `create_backend(kind, config)` 而不是
>    实例方法——未创建完的实例根本无法构造。
> 3. E2B 实现**不用** `langchain-e2b`：它不带 `py.typed`、是 `0.0.x`，而裸 `e2b` 的
>    `commands.run` 原生支持 `cwd`/`envs`。理由详见 [依赖管理.md](依赖管理.md)。

**唯一后端 = E2B。** 早期曾随接口提供一个本地 subprocess 的「假沙箱」以支持无 Key 开发，现已移除：它不是沙箱，一条 `rm -rf ~` 就能毁掉开发机，长期存在误用风险。无 E2B Key 时的离线回归改由**注入式 SDK 测试替身**承担——测试向 `E2BSandboxBackend` 注入一个伪造的 `e2b.Sandbox`（只存在于 `tests/`，不是后端实现，生产代码无法引用），从而不联网覆盖上传 / 执行 / 导出 diff / 销毁的全部逻辑。

**⚠️ 代码外发风险（必须正视）**：默认后端 E2B 是**云端**沙箱，意味着目标仓库的代码会被上传到第三方服务执行。这是企业采用的**第一道否决线**，必须显式声明：

- README 与文档中**明确写出**“代码将离开本地、在第三方云沙箱运行”。
- 提供规避路径：`--dry-run` 只跑非沙箱流程（生成方案、展示 diff 预览），不建沙箱、不调用远端。
- 接口保持可插拔：将来若要换成自建/本地隔离方案，替换 `SandboxBackend` 的实现类即可，上层编排与工具代码零改动。
- 私有仓库、含密钥的仓库**默认不建议**使用云端后端。

**沙箱预热（P1 前必须验证的地基）**：E2B 是**远端容器**，代码与运行环境都需要“送进去”，这是全流程最大的隐形工作量，不能留到写业务时才碰。设计要点：

- **基础镜像**：预构建带 `conda` 与常用工具链的模板镜像，用 `Sandbox.create(template=...)` 复用，避免每次冷装。
- **代码进入方式**：只**上传**仓库的必要文件子集（源码 + 元数据）到 `/workspace/`；E2B **无法挂载宿主机目录**，因此“挂载”一词不适用。
- **环境准备**：依赖按镜像缓存，或按**目标仓库自己的依赖清单**增量安装（`pyproject.toml` / `requirements.txt` / `environment.yml` 皆可能，需探测而非假定）；对大仓库要评估上传耗时与超时。
- **冷启动基准**：P0 必须实测“创建沙箱 → 能跑通 `pytest`”的耗时，作为后续所有时间预算的基线。

**补丁的流转形态（先定清楚）**：子代理在沙箱里改的是**文件**，但跨节点的产物必须是**可序列化的 diff 字符串**（要进 checkpoint、要给人审、要跨进程传输）。因此约定：

- `run_agent` 在**工作沙箱**里跑完 analyze→search→fix，用 `git diff` 导出 `proposed_fix`，**随即销毁工作沙箱**。
- `apply_fix` 在**全新的验证沙箱**里应用这段 diff，再跑测试。
- 宿主与沙箱之间**只传 diff 与测试输出**；不传凭据、不传宿主目录（见安全原则）。

**沙箱生命周期管理**（两段式，与检查点强耦合）：核心原则是 **“不跨人工门控持有沙箱”**——人审可能等数小时，沙箱必被 E2B 回收，跨门控持有既不经济也不可靠：

1. **run_agent 段**：节点入口创建工作沙箱 → 子代理在其内读写/试改 → 导出 diff → 节点结束前 `kill()`。
2. **human_review 段**：**不持有任何沙箱**（补丁以 diff 落盘在 checkpoint 里），因此等多久都不烧钱、也不怕被回收。
3. **apply_fix 段**：创建验证沙箱 → 应用 diff → 运行校验。
4. **重试即重置**：`test_result → retry` 回到 `run_agent` 时天然走第 1 步，得到的是干净沙箱，无需额外重置逻辑。
5. **终止必清理**：任何走到 `END`（含 `fail`/`abort`）的路径，收尾节点统一 `kill()` 当前沙箱；异常路径用 context manager / finally 兜底。

> `sandbox_id` 只记录**当前**活动沙箱（run_agent 段或 apply_fix 段存在，human_review 段为 `None`）。resume 时若发现 `sandbox_id` 非空但已失效，直接丢弃重建。

**安全原则**：
- 沙箱只**上传**仓库的必要文件子集，绝不把宿主目录、`~/.ssh`、`~/.aws`、`.env` 送入沙箱。
- **GitHub Token 只存在于宿主机编排层，沙箱内永不可见**——沙箱只产出代码改动，`git push` / PR 创建全部在宿主机侧完成（见 3.5）。这样即便 `execute` 工具被 prompt injection 攻破，也偷不到能改写整个仓库的凭据。
- 攻击面被限制在“一个一次性的云端容器”内，最坏结果只是删除一个临时环境。

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

**记忆由宿主机写入**：沙箱里跑的 Agent **不直接写记忆文件**（沙箱是一次性、隔离的，写了也留不下）。`run_agent` 结束后，由宿主机节点把“本次值得沉淀的结论”提取出来写入 `memories/`，下次运行再注入——这与 3.5 中 `write_memory` 属宿主机动作一致。

## 3.5 工具集设计

Coding Agent 的工具不在于“多”，而在于精确和最小化。**先分清两类东西**：交给 Agent 的**沙箱工具**，与编排层自己执行的**宿主机动作**——后者是对外凭据的唯一出口，绝不下放给 Agent。

**（A）交给 Agent 的沙箱工具**——构成 DeepAgent 的工具集，全部在沙箱内执行：

| 工具 | 用途 | 权限范围 |
|---|---|---|
| ripgrep_search | 在代码库中搜索文本模式 | 只读，作用域绑定到 `repo_path`（非模型参数） |
| read_file | 读取文件内容 | 只读，服务端校验路径在 `/workspace/` 内 |
| write_file | 写入修改后的文件 | 写权限，服务端校验在 `/workspace/` 内且属于 `target_files` |
| execute_shell | 在沙箱中运行命令 | 受限：白名单命令（pytest, conda, npm） |

**（B）编排层的宿主机动作**——由 LangGraph 节点直接调用，**不进沙箱、不暴露给 Agent**：

| 动作 | 归属节点 | 凭据 / 触达范围 |
|---|---|---|
| fetch_issue | fetch_issue | GitHub Token（宿主机） |
| create_github_pr | create_pr | GitHub Token（宿主机），人工确认后触发 |
| write_memory | 各节点收尾 | 写 `.deepagents/memories/`（宿主机，见 3.4） |

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

**凭据隔离铁律**：`fetch_issue` / `create_github_pr` 运行在**宿主机编排层**，GitHub Token 从不进入沙箱、不写入任何进沙箱的文件。沙箱与宿主机之间只交换**代码补丁（diff）**，不交换凭据。

为什么不用通用 execute 工具：DeepAgent 的 LocalShellBackend 自带一个 execute 工具，但它允许任意 shell 命令。夜间维护者应该显式定义白名单工具，而不是给 Agent 一个万能命令执行器。这既降低了安全风险，也让 Agent 的行为更可预测。

## 3.6 人工审核门控（HITL）

为什么需要审核：完全自主的 PR 创建是危险的。Agent 可能在“修复”一个简单 lint 错误时，意外删除了一个关键函数。人工审核门控让 Agent 在关键决策点暂停。这是**异步门控**——夜间挂起并留档，白天人来拍板。

LangGraph 的中断机制。**关键：中断必须放在独立节点的入口，不能和“生成方案”同节点**——`interrupt()` 触发后，resume 时整个节点会从头重跑，若生成代码在 `interrupt()` 之前，会**重新生成一份方案**，导致“人批准的补丁 ≠ 实际应用的补丁”：

```python
from langgraph.types import interrupt, Command

def human_review_node(state: MaintenanceState):
    # 入口即是 interrupt，节点内无任何副作用；resume 重跑也不会重做工作
    approval = interrupt({
        "action": "review_patch",
        "patch": state["proposed_fix"],     # 已生成的补丁（可 JSON 序列化）
        "target_files": state["target_files"],
        "question": "是否批准此补丁？回复 'approve' 或提供修改意见。",
    })
    if approval.get("decision") == "reject":
        return {"aborted": True}            # review_decision → END
    return {}                               # 批准：无状态变更，交给 apply_fix
```

对应的生成节点保持“纯生成、不中断”：

```python
def run_agent_node(state: MaintenanceState):
    # 调用一次 DeepAgent，内部跑完 analyze→search→fix
    result = main_agent.invoke(...)
    return {
        "proposed_fix": result["patch"],
        "target_files": result["target_files"],
        "token_used": state.get("token_used", 0) + result["tokens"],
    }
```

恢复执行：

```python
thread_config = {"configurable": {"thread_id": "issue-123"}}
graph.invoke(Command(resume={"decision": "approve"}), config=thread_config)
```

HITL 的本质：不是简单的“确认弹窗”，而是 Runtime 层面的状态暂停与恢复。LangGraph 的检查点机制让图执行可以在任意节点中断，保存完整状态，等待外部输入后从断点继续。

> 两点约束：① `interrupt()` 的载荷会随 checkpoint 落盘，**必须可 JSON 序列化**；② `interrupt()` 的返回值就是 `Command(resume=...)` 传入的值，`review_decision`（见 3.1）据 `aborted` 字段决定走 `apply_fix` 还是 `END`。

# 4. 与现有方案的差异化

| 维度 | OpenHands | OSC-Agent | 夜间维护者（本方案） |
|---|---|---|---|
| 定位 | 通用自主软件工程师 | 开源贡献 CLI | 定时/CI 触发的技术债清理 |
| 交互模式 | 对话式 + 自主执行 | CLI 命令 | 无头产出草案 + 异步人工审核 |
| 沙箱 | Docker 全环境 | E2B 云端 | E2B 云端（SandboxBackend 可插拔） |
| 子代理 | 未明确分层 | 多 Agent 但非 DeepAgent | DeepAgent isolated 子代理 |
| 记忆 | microagents（人工编写、静态） | 无 | AGENTS.md + **Agent 自动积累**（按 repo 隔离） |
| 目标用户 | 开发者手动使用 | 开源贡献者 | CI 系统 / 团队夜间自动化 |

> 注：OSC-Agent 指面向开源贡献场景的 CLI 型编码 Agent；此处仅作定位对比，具体能力以其官方说明为准。

# 5. 开发路线图

> 路线图已拆分为独立文档：**[开发路线图.md](开发路线图.md)**。
> 内含各阶段任务清单、验收标准、P0 实现进度对照表、度量与回归机制、任务类型 → 流程与验证映射。

# 6. 关键风险与规避

| 风险 | 规避策略 |
|---|---|
| Agent 修改了不相关的文件 | propose 后加人工审核；write_file 只能操作 target_files（服务端强制）+ L3 护栏 |
| 代码外发到第三方云沙箱 | 显式声明；提供 `--dry-run` 与可插拔本地后端；敏感仓库禁用云端后端 |
| 沙箱内窃取 GitHub Token | token 只在宿主机，沙箱不可见；宿主与沙箱只交换 diff（见 3.3/3.5） |
| 沙箱预热超时 / 上传失败 | 预构建带 conda 的模板镜像；P0 实测冷启动基准；重试与超时兜底 |
| 沙箱逃逸 | 使用 E2B 云端一次性沙箱隔离，绝不向沙箱传入宿主敏感文件 |
| 沙箱在等待期被回收 | 不跨人工门控持有沙箱：run_agent 出 diff 即销毁，apply_fix 用新沙箱（见 3.3） |
| 审核通过后补丁被重生成 | `interrupt()` 独立成 `human_review` 节点，入口即中断（见 3.6） |
| LLM 幻觉导致错误修复 | 分级验证 `(L1 或 L2) 且 L3`；不通过则回退并报告 |
| 上下文爆炸 | 子代理 isolated；注入只传路径/行号；工具结果截断（5000 字符） |
| Token 成本失控 | 模型路由分层（见 3.2）+ 单任务预算熔断（见 3.1） |
| 记忆跨仓库污染 | 记忆按 repo 隔离（见 3.4） |
| 记忆写进一次性沙箱而丢失 | 记忆由宿主机节点写入 memories/，沙箱内不落记忆（见 3.4） |

一句话总结：这个项目的核心价值不在于“让 AI 写代码”，而在于用 DeepAgent 的架构能力（子代理隔离 + 沙箱 + 记忆）构建一个可控、可审核、可复现的自动化维护流水线。范围越窄（只做“夜间维护”），完成度越高，在 GitHub 上被 star 和复用的概率越大。
