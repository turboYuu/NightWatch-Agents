"""可观测性：运行账本、状态快照、LangSmith tracing 接线。

三件事各自独立，落在同一个包是因为它们的生命周期一致（都由一次运行驱动）：

====================  ==========================================================
运行账本              ``~/.nightwatch/ledger.jsonl``，一次运行一行，供 5.7 度量聚合
状态快照              ``~/.nightwatch/runs/<run_id>/snapshots/<seq>-<阶段>.json``
tracing               只设环境变量，生产代码不 import langsmith（见 ``tracing``）
====================  ==========================================================

⚠️ **P0 的现实**：本阶段没有任何 LLM 调用，因此 LangSmith 侧收不到数据——本包在
P0 交付的是**接线**（开关、状态、账本、快照），不是可观测的数据。token 成本要等 P1
的 ``run_agent`` 节点接入后才产生。别把「``--trace`` 打开了」读成「有 trace 可看」。

约定与理由详见 doc/可观测性.md。
"""

from __future__ import annotations

from nw_agent.observability.ledger import (
    LEDGER_SCHEMA_VERSION,
    OUTCOMES,
    DuplicateRunError,
    Ledger,
    PriceTable,
    RunRecord,
    StageRecord,
    TokenUsage,
    append,
    load_prices,
    read,
    summarize,
)
from nw_agent.observability.run_context import NoOpRecorder, Recorder, RunContext
from nw_agent.observability.store import (
    ENV_HOME,
    ledger_path,
    nw_home,
    prices_path,
    run_dir,
    runs_dir,
)
from nw_agent.observability.tracing import (
    TracingStatus,
    configure_tracing,
    tracing_env_enabled,
)

__all__ = [
    "ENV_HOME",
    "LEDGER_SCHEMA_VERSION",
    "OUTCOMES",
    "DuplicateRunError",
    "Ledger",
    "NoOpRecorder",
    "PriceTable",
    "Recorder",
    "RunContext",
    "RunRecord",
    "StageRecord",
    "TokenUsage",
    "TracingStatus",
    "append",
    "configure_tracing",
    "load_prices",
    "ledger_path",
    "nw_home",
    "prices_path",
    "read",
    "run_dir",
    "runs_dir",
    "summarize",
    "tracing_env_enabled",
]