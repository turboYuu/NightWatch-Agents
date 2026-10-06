"""主代理装配与 run_task 的离线回归（零成本：替身沙箱 + stub 模型）。

覆盖四件事：

1. **整链路**：主代理委派 ``fix`` → 子代理改文件 → 宿主侧导出非空补丁。
2. **token 记账**：用量经 callback 落进 ``RunContext``——这是 5.7「单任务 token 成本」
   指标的第一个真实数据来源。
3. **预算熔断**：超预算必须真的中断，而不是只记一笔。
4. **差异哨兵**：``task`` 工具当前暴露哪些子代理类型（deepagents 会自动补
   ``general-purpose``）——它是**升级哨兵**，不是功能契约，见测试内的说明。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent_stub import COMPLETION_TOKENS, PROMPT_TOKENS, StubChatModel
from eval_harness import make_backend
from nw_agent.agents import (
    FIX_MODEL_ENV,
    MAIN_MODEL_ENV,
    AgentConfigError,
    AgentModels,
    BudgetExceeded,
    build_main_agent,
    resolve_agent_models,
    run_task,
)
from nw_agent.backends import SandboxBackend, SandboxConfig
from nw_agent.observability import RunContext


def _setup(tmp_path: Path) -> tuple[SandboxBackend, StubChatModel, object]:
    """装一个「仓库里只有 app.py」的替身沙箱，并装配好主代理。"""
    backend, _double = make_backend(tmp_path, SandboxConfig())
    backend.upload_repo([("app.py", b"x = 1\n")])
    stub = StubChatModel(
        edit_path=f"{backend.config.repo_path}/app.py",
        old_string="x = 1",
        new_string="x = 2",
    )
    agent = build_main_agent(backend, AgentModels(main=stub, fix=stub))
    return backend, stub, agent


# ------------------------------------------------------------ 整链路
def test_delegation_to_fix_produces_a_patch(tmp_path: Path) -> None:
    """主代理委派 fix、fix 改文件、宿主导出补丁——P1 第一周的最短链路。"""
    backend, _stub, agent = _setup(tmp_path)

    result = run_task(agent, backend, "把 app.py 的 x 改成 2", model_label="stub:offline")

    assert "-x = 1" in result.patch
    assert "+x = 2" in result.patch
    assert result.final_text, "主代理应当给出一句说明"
    # 子代理确实是在**沙箱内**改的文件。
    assert (tmp_path / "workspace" / "repo" / "app.py").read_text(encoding="utf-8") == "x = 2\n"


def test_usage_is_recorded_into_the_run_context(tmp_path: Path) -> None:
    """token 用量经 callback 落进账本——含**子代理**的调用。

    这条如果只读 ``invoke`` 的返回值就会失败：子代理在独立上下文里跑，顶层 messages
    里看不到它的用量（见 ``_token_sink`` 的模块 docstring）。
    """
    backend, _stub, agent = _setup(tmp_path)
    context = RunContext(kind="cli", home=tmp_path / "obs", run_id="20260101T000000Z-agent")

    result = run_task(
        agent, backend, "把 app.py 的 x 改成 2", model_label="stub:model", recorder=context
    )
    context.finish("success")

    assert result.n_llm_calls >= 2, "主代理与子代理至少各一次调用"
    assert result.total_tokens == result.n_llm_calls * (PROMPT_TOKENS + COMPLETION_TOKENS)
    record = context.record
    assert record is not None
    assert len(record.tokens) == result.n_llm_calls
    assert record.tokens[0].model == "stub:model"  # 回落标签
    assert record.total_tokens == result.total_tokens


def test_budget_exceeded_aborts_the_run(tmp_path: Path) -> None:
    """超预算要真的抛出来。

    这里断言的是**异常链里含** ``BudgetExceeded``：它从 callback 里抛出，可能被
    langgraph 包一层再冒出来。真正要证明的是「熔断不是空话」——
    ``TokenSink.raise_error`` 不设成真时，这个异常会被 langchain 静默吞掉。
    """
    backend, _stub, agent = _setup(tmp_path)

    with pytest.raises(Exception) as excinfo:  # noqa: B017 - 见上：可能被包装
        run_task(
            agent,
            backend,
            "把 app.py 的 x 改成 2",
            model_label="stub:offline",
            budget_tokens=1,  # 远小于第一次调用的用量
        )
    assert _has_budget_error(excinfo.value), f"异常链里没有 BudgetExceeded：{excinfo.value!r}"


def test_empty_patch_is_not_an_error(tmp_path: Path) -> None:
    """模型什么都没改时，补丁为空但运行本身是成功的（空补丁 ≠ 报错）。"""
    backend, _double = make_backend(tmp_path, SandboxConfig())
    backend.upload_repo([("app.py", b"x = 1\n")])
    # edit_path 为空 → stub 不发 edit_file，直接给终态文本。
    stub = StubChatModel()
    agent = build_main_agent(backend, AgentModels(main=stub, fix=stub))

    result = run_task(agent, backend, "什么都不用改", model_label="stub:offline")

    assert result.patch == ""
    assert result.final_text


# ------------------------------------------------------------ 模型配置
def test_resolve_agent_models_reads_both_env_vars() -> None:
    models = resolve_agent_models(
        {MAIN_MODEL_ENV: "anthropic:claude-sonnet-4-6", FIX_MODEL_ENV: "anthropic:claude-opus-4-1"}
    )
    assert models.main == "anthropic:claude-sonnet-4-6"
    assert models.fix == "anthropic:claude-opus-4-1"
    assert models.specs() == ("anthropic:claude-sonnet-4-6", "anthropic:claude-opus-4-1")


@pytest.mark.parametrize(
    ("main", "fix", "expected_env"),
    [
        (None, "anthropic:m", MAIN_MODEL_ENV),
        ("   ", "anthropic:m", MAIN_MODEL_ENV),
        ("anthropic:m", None, FIX_MODEL_ENV),
        ("claude-sonnet-4-6", "anthropic:m", MAIN_MODEL_ENV),  # 缺 provider 前缀
        ("anthropic:", "anthropic:m", MAIN_MODEL_ENV),  # 冒号一侧为空
        (":m", "anthropic:m", MAIN_MODEL_ENV),
    ],
)
def test_resolve_agent_models_rejects_bad_config(
    main: str | None, fix: str | None, expected_env: str
) -> None:
    """缺配置必须显式报错，不猜默认模型（产品说明 L240）。"""
    env = {k: v for k, v in ((MAIN_MODEL_ENV, main), (FIX_MODEL_ENV, fix)) if v is not None}
    with pytest.raises(AgentConfigError, match=expected_env):
        resolve_agent_models(env)


def test_config_error_message_shows_a_valid_example() -> None:
    """错误信息要能照抄——用户看到 provider:model 的样例才知道该怎么写。"""
    with pytest.raises(AgentConfigError) as excinfo:
        resolve_agent_models({})
    assert ":" in str(excinfo.value)
    assert "provider:model" in str(excinfo.value)


# ------------------------------------------------------------ 差异哨兵
def test_available_subagents_sentinel(tmp_path: Path) -> None:
    """**升级哨兵**：钉住 ``task`` 工具当前暴露的子代理类型集合。

    它**故意**会在 deepagents 升级导致默认行为变化时变红——这不是「功能正确性」断言，
    而是「升级需要人工确认差异」的信号。背景：``create_deep_agent`` 会在调用方没有提供
    同名 spec 时**自动补一个 ``general-purpose`` 子代理**，于是主代理实际能委派的类型
    比「只有一个 fix」更多（见 ``agents/factory.py`` 的模块 docstring）。

    断言用**集合包含**而非精确相等：deepagents 若再加第三个默认子代理，那属于需要知晓
    但不必立刻失败的信息；而 ``fix`` 消失或 ``general-purpose`` 被顶掉，就必须有人看一眼。

    变红时怎么办：读 deepagents 的 changelog，判断是「默认子代理改名/移除」还是
    「我们的 spec 覆盖了它」，再决定改断言还是改装配——**不要**直接把断言改成当前实际值。
    """
    _backend, _stub, agent = _setup(tmp_path)
    names = _available_subagent_types(agent)
    assert "fix" in names, f"我们声明的 fix 子代理没出现在 task 工具里：{sorted(names)}"
    assert "general-purpose" in names, (
        f"deepagents 不再自动补 general-purpose 了（当前：{sorted(names)}）——"
        "这是行为变化，请确认后再更新本哨兵"
    )


def _available_subagent_types(agent: object) -> set[str]:
    """从已编译图里取出 ``task`` 工具描述中列出的子代理名。

    走的是 langgraph 的私有结构（``nodes["tools"].bound._tools_by_name``）：deepagents
    没有提供公开 accessor。取不到时**让测试直接报错**而不是返回空集合——哨兵失效必须是
    「红」，不能是「悄悄通过」。
    """
    nodes = getattr(agent, "nodes", None)
    assert isinstance(nodes, dict), "拿不到已编译图的节点表，langgraph 结构变了"
    tools_node = nodes.get("tools")
    bound = getattr(tools_node, "bound", None)
    tools_by_name = getattr(bound, "_tools_by_name", None)
    assert isinstance(tools_by_name, dict), "拿不到工具表，langgraph 结构变了"
    description = str(getattr(tools_by_name["task"], "description", ""))
    return set(re.findall(r"^- ([A-Za-z0-9_.-]+): ", description, re.MULTILINE))


def _has_budget_error(exc: BaseException) -> bool:
    """异常链（含 ``__cause__`` / ``__context__``）里是否含 ``BudgetExceeded``。"""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, BudgetExceeded):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False
