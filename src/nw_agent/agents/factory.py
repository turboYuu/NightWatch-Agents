"""用 ``create_deep_agent`` 装配主代理（P1 第一周：只显式声明一个 ``fix`` 子代理）。

对应 doc/NightWatch产品说明.md 3.2：``analyze`` / ``search`` / ``fix`` 是 DeepAgent 的
``subagents``，由主代理在**一次运行内**通过 ``task`` 工具按序委派。第一周只落地 ``fix``
——先把「主代理委派 → 子代理改文件 → 宿主导出补丁」这条最短链路跑通，三个子代理的完整
编排留到第二周。**LangGraph 不介入这一步**：它只在 ``run_agent`` 节点调用一次 DeepAgent。

⚠️ **一个必须知道的差异：deepagents 会自己补一个 ``general-purpose`` 子代理。**

    ``create_deep_agent`` 在调用方没有提供同名 spec 时会自动插入一个 ``general-purpose``
    子代理，它**继承主代理的全部工具**（含 ``write_file`` / ``edit_file`` / ``execute``）。
    也就是说 ``task`` 工具实际暴露的是 ``fix`` + ``general-purpose`` 两个类型，主代理因此
    比「只给一个 fix」拥有更宽的权限。

    这是**有意识的接受默认**，不是遗漏：关掉它要用 ``register_harness_profile`` 的 beta
    API，而现在（第一周）还没有写白名单，关不关都不改变「模型能写沙箱任意路径」这个事实。
    重审时机：**下一周落地 ``write_file`` / ``edit_file`` 路径白名单时**——那时若发现
    general-purpose 能绕过白名单，就必须关掉它或给它换一份受限 spec。
    ``tests/test_agent_offline.py`` 里有一条**差异哨兵**测试钉住当前行为。

⚠️ **本轮没有工具权限收紧**（这是已知风险，不是疏忽）：``edit_file`` / ``write_file`` 的
    路径由模型给出，能写沙箱内任意位置（包括验收目录）；``execute`` 能跑任意 shell 命令。
    缓解仅限：沙箱是一次性的、凭据不进沙箱、宿主与沙箱之间只有 diff 单向流出。
    ``write_file`` 白名单与 ``execute`` 白名单是下一周的第一优先级。

**模型配置为什么是参数而不是环境变量**：装配函数不读环境——读环境是 :func:`resolve_agent_models`
的职责，由调用方（CLI）决定何时读。这样离线测试可以直接传预建模型实例，一行环境都不碰，
也不必为了测试去伪造 provider key。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from deepagents import SubAgent, create_deep_agent
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from nw_agent.backends import SandboxBackend

# 模型配置的环境变量名。值必须是 ``init_chat_model`` 认得的 ``provider:model`` 串。
MAIN_MODEL_ENV = "NW_MODEL_MAIN"
FIX_MODEL_ENV = "NW_MODEL_FIX"

# 错误信息里给出的样例。只用来示范 ``provider:model`` 的**格式**，不代表项目绑定某个供应商
# （产品说明 L240：模型名以配置项注入）。这里用 deepseek 是因为它是当前实际在用的 provider，
# 而且它的包已显式声明在 pyproject.toml 里——照着样例写就能跑起来。
MODEL_SPEC_EXAMPLE = "deepseek:deepseek-chat"

# 主代理可见的仓库根（写进系统提示词）。下面由 build_main_agent 用真实 config 覆盖。
_DEFAULT_REPO_PATH = "/workspace/repo"

MAIN_SYSTEM_PROMPT = """\
你是代码仓库的维护编排器。待维护的仓库位于沙箱内的 {repo_path}，所有文件操作都相对它。

工作方式：
1. 把修复工作通过 task 工具委派给 fix 子代理（subagent_type="fix"）。交给它**精炼的上下文**
   ——要改哪个文件、哪个符号、行区间、期望结果。不要把整份需求原文或整份文件正文塞进去。
2. **不要自己搜索或修改文件**：定位与改动都交给 fix。
3. fix 返回后，用一两句话说明「改了哪个文件的什么」，不要复述 diff（diff 由宿主侧导出）。

硬约束：
- 修改**已存在**的文件必须用 edit_file（先 read_file 看清原文，再给出 old_string / new_string）。
  write_file 只能新建文件，对已存在的路径会直接失败——不要拿它当覆盖写用。
- 只改任务要求的文件，不要顺手重构，也不要改动测试与配置文件。
"""

FIX_SUBAGENT_SYSTEM_PROMPT = """\
你是修复工程师。你会在沙箱里直接读写文件，完成主代理交代的代码修改。

做法：
1. 先 read_file 看清目标文件的真实内容（不要凭主代理的描述猜）。
2. 用 edit_file 做**最小**改动：old_string 要足够独特，new_string 只包含必要变化。
   新建文件才用 write_file。
3. 改完再 read_file 自检一次：确认改动生效、且**没有动到目标之外的文件**。
4. 返回时简述：改了哪个文件、改了什么、有没有依赖或风险需要注意。不要粘贴整份 diff。
"""

# ``fix`` 子代理的声明。对应产品说明 3.2 里 fix 的职责。
FIX_SUBAGENT: SubAgent = {
    "name": "fix",
    "description": "根据给出的上下文修改代码文件，并自检是否只动了目标文件",
    "system_prompt": FIX_SUBAGENT_SYSTEM_PROMPT,
}


class AgentConfigError(RuntimeError):
    """模型配置缺失或非法。

    刻意显式失败而不是回落到某个默认模型：产品说明 L240 要求「模型名随供应商更新，
    以配置项注入，不写死」。猜一个默认模型会在供应商改价/下线时**静默**换掉行为。
    """


@dataclass(frozen=True)
class AgentModels:
    """两个角色的模型配置。

    生产路径传 ``provider:model`` 字符串（交 deepagents 内部 ``resolve_model`` 解析），
    离线测试传预建实例（如 ``tests/agent_stub.py`` 的 stub）。两者都**原样透传**给
    ``create_deep_agent``，因此走的是同一条装配路径，只是模型不同。
    """

    main: str | BaseChatModel
    fix: str | BaseChatModel

    def specs(self) -> tuple[str, ...]:
        """可打印/可入账本的模型标识（实例则取其类名，避免把对象塞进 JSON）。"""
        return tuple(_label(item) for item in (self.main, self.fix))


def resolve_agent_models(environ: Mapping[str, str] | None = None) -> AgentModels:
    """从环境变量解析两个角色的模型配置。

    Args:
        environ: 环境映射；None 时读 ``os.environ``（测试可传字典避免碰真实环境）。

    Returns:
        :class:`AgentModels`；两个字段都是**原始字符串**，不做 provider 可用性校验。

    Raises:
        AgentConfigError: 任一变量未设置、为空（含纯空白）、不含 ``:`` 或 ``:`` 一侧为空。
    """
    env = os.environ if environ is None else environ
    return AgentModels(
        main=_require_model_spec(env, MAIN_MODEL_ENV),
        fix=_require_model_spec(env, FIX_MODEL_ENV),
    )


def main_system_prompt(repo_path: str) -> str:
    """渲染主代理的系统提示词（把仓库根写进去）。

    单独抽出来是为了让「提示词里引用的路径」与「后端实际使用的路径」同源——
    否则离线链路（repo_path 指向临时目录）与 E2B 链路会各写死一份。
    """
    return MAIN_SYSTEM_PROMPT.format(repo_path=repo_path)


def build_main_agent(
    backend: SandboxBackend,
    models: AgentModels,
    *,
    system_prompt: str | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """装配主代理（带一个 ``fix`` 子代理）。

    Args:
        backend: 沙箱后端。deepagents 会据此注入 ``ls`` / ``read_file`` / ``write_file`` /
            ``edit_file`` / ``glob`` / ``grep`` / ``execute`` 七个工具，全部作用在沙箱内。
        models: 两个角色的模型配置，见 :class:`AgentModels`。
        system_prompt: 覆盖默认提示词；None 用 :func:`main_system_prompt`。注意 deepagents
            对 ``str`` 是**追加**（我们这段在前、它自带的 base prompt 在后），不是替换。

    Returns:
        已编译的状态图。调用方负责用它 ``invoke``；本函数不碰沙箱、不跑模型。

    Note:
        ``create_deep_agent`` 的返回类型里第四个泛型参数（ContextT）在未传
        ``context_schema`` 时退化为 ``None``，无法在调用点闭合，故此处 ``cast`` 一次、
        统一放宽成 ``Any``。**不要让这处 cast 扩散**：对外只暴露这一个宽类型，
        收窄（取最终文本、取 messages）发生在 ``runner`` 里唯一的一个解析函数内。
    """
    repo_path = backend.config.repo_path
    return cast(
        "CompiledStateGraph[Any, Any, Any, Any]",
        create_deep_agent(
            model=models.main,
            system_prompt=system_prompt or main_system_prompt(repo_path),
            subagents=[
                # fix 用强档模型（产品说明 3.2 的模型路由表）；未单独配置时同样传 fix 的串。
                {**FIX_SUBAGENT, "model": models.fix},
            ],
            backend=backend,
            # 名字进 LangSmith 的 run 名，便于在 tracing 里一眼分清是哪条链路。
            name="nightwatch-main",
        ),
    )


def _require_model_spec(env: Mapping[str, str], name: str) -> str:
    """读取并校验一个模型配置串。

    空串与纯空白都按「未设置」处理（与 ``observability/store.py`` 对 ``NW_HOME`` 的
    处理同源）：环境变量存在但为空是常见形态，当成已配置会让错误推迟到调用期才炸。
    **不校验 provider 是否受支持**——受支持的 provider 集合随 deepagents 升级变化，
    在这里写死等于给自己埋雷；那件事交给 ``init_chat_model`` 去报错。
    """
    raw = (env.get(name) or "").strip()
    if not raw:
        raise AgentConfigError(
            f"缺少模型配置环境变量 {name}；请设为 provider:model 形式，例如 "
            f"{name}={MODEL_SPEC_EXAMPLE}"
        )
    provider, sep, model = raw.partition(":")
    if not sep or not provider.strip() or not model.strip():
        raise AgentConfigError(
            f"{name}={raw!r} 不是合法的 provider:model（provider 与 model 都不能为空），"
            f"例如 {MODEL_SPEC_EXAMPLE}"
        )
    return raw


def _label(model: str | BaseChatModel) -> str:
    """把模型配置转成可打印的标签；实例取类名（`str()` 一个模型对象没有意义）。"""
    return model if isinstance(model, str) else type(model).__name__
