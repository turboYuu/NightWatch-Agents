"""LangSmith 接线的**真链路**自测：默认 skip，有 key 时手动跑一次。

    conda run -n nightwatch_agents pytest tests/test_tracing_live.py -o addopts="" -s

与 ``tests/test_backends_e2b.py`` 的真链路段同一个姿态：CI（无 key）应当 skip 而不是
fail——本仓库的离线回归必须能在没有凭据的机器上全绿。

⚠️ **这条测试绿 ≠ 可观测性有数据**。它证明的是「环境变量、项目名、Client 鉴权这条线
接通了」，而不是「有东西被 trace」：它用一个平凡函数造出真实 run，不依赖任何模型调用
（P0 本来也没有）。LangSmith 上有没有**业务**数据，要等 P1 的 Agent 落地。

为什么要在这里 import langsmith（生产代码不 import 它）：本测试必须能清掉
``langsmith.utils.get_env_var`` 的 ``lru_cache``——同进程里先跑过其它用例时，缓存可能
已经记下了「tracing 未开启」，之后再怎么设环境变量都会被屏蔽。这是测试特有的手段，
不该泄漏进生产路径。
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

# 生产代码不依赖 langsmith（它由 deepagents 连带引入），故缺它时整文件跳过。
pytest.importorskip("langsmith", reason="真链路自测需要 langsmith（deepagents 的连带依赖）")

needs_key = pytest.mark.skipif(
    not os.environ.get("LANGSMITH_API_KEY"),
    reason="需要 LANGSMITH_API_KEY 才能真跑 LangSmith",
)


@needs_key
def test_trace_reaches_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """造一个真实 run，并断言它确实到了服务端。"""
    import langsmith
    from langsmith import Client
    from langsmith.run_helpers import traceable

    # 独立 project，避免污染真实项目；也让「这次断言的是哪批 run」没有歧义。
    project = f"nw-selftest-{uuid.uuid4().hex[:8]}"
    monkeypatch.setenv("LANGSMITH_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", project)
    # 关键：清掉进程级缓存，否则本次设置可能被先前的读取结果屏蔽（见模块 docstring）。
    langsmith.utils.get_env_var.cache_clear()

    @traceable(name="nw-selftest")
    def trivial(value: int) -> int:
        return value + 1

    assert trivial(1) == 2

    # post() 是后台线程异步上传，落地有延迟，故轮询而不是立刻断言。
    client = Client()
    found = None
    for _ in range(30):
        runs = list(client.list_runs(project_name=project, limit=5))
        if runs:
            found = runs[0]
            break
        time.sleep(0.5)

    assert found is not None, "run 未出现在服务端：检查 key 的权限与 project 是否生效"
    assert found.name == "nw-selftest"
