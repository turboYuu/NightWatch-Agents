"""可观测性的回归：路径解析、账本读写、成本口径、tracing 接线。

三处口径是本文件的重点，它们都属于「错了会很晚才发现」的那类：

1. **缺数据是 ``None`` 不是 0**——token 与费用都如此。写成 0 会被下游读成「零成本」。
2. **缺 API Key 时不假装开启**——否则 trace 静默丢失，而报告里一切正常。
3. **测试绝不写真实用户目录**——靠 ``conftest.py`` 的 autouse 隔离，这里再补一条断言
   把「隔离确实生效」也钉住。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nw_agent.observability import (
    ENV_HOME,
    LEDGER_SCHEMA_VERSION,
    DuplicateRunError,
    RunContext,
    RunRecord,
    StageRecord,
    TokenUsage,
    configure_tracing,
    ledger_path,
    nw_home,
    prices_path,
    read,
    run_dir,
    runs_dir,
    summarize,
)
from nw_agent.observability.ledger import append, load_prices
from nw_agent.observability.store import ensure_parent
from nw_agent.observability.tracing import ENV_API_KEY, ENV_PROJECT, ENV_TRACING, TRACING_ON

_UNSET_ENV = (str(ENV_HOME), *ENV_API_KEY, *ENV_TRACING, *ENV_PROJECT)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """清掉所有与本模块相关的环境变量，让用例从确定的状态出发。"""
    for name in _UNSET_ENV:
        monkeypatch.delenv(name, raising=False)


# ------------------------------------------------------------ 路径解析
def test_nw_home_prefers_explicit_over_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_HOME, str(tmp_path / "from-env"))
    assert nw_home(tmp_path / "from-arg") == tmp_path / "from-arg"


def test_nw_home_uses_env_over_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_HOME, str(tmp_path / "from-env"))
    assert nw_home() == tmp_path / "from-env"


def test_nw_home_treats_empty_env_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """空串是「变量在但没有值」的常见形态，不该被当成「根目录 = 当前目录」。"""
    monkeypatch.setenv(ENV_HOME, "")
    assert nw_home() == Path.home() / ".nightwatch"


def test_autouse_fixture_keeps_default_home_off_the_real_user_dir() -> None:
    """隔离机制本身的回归：不带参数解析出的根目录必须落在临时目录里。"""
    home = nw_home()
    assert home.name == "nw-home"
    assert home.parent != Path.home()


def test_resolving_paths_creates_nothing(tmp_path: Path) -> None:
    """解析路径是纯计算——import 期与解析期都不得在磁盘上留下目录。"""
    home = nw_home(tmp_path / "not-created")
    runs_dir(home)
    ledger_path(home)
    prices_path(home)
    run_dir(home, "20260101T000000Z-demo")
    assert not home.exists()


def test_ensure_parent_is_where_creation_happens(tmp_path: Path) -> None:
    """目录创建只发生在写文件前的那一刻。"""
    target = tmp_path / "a" / "b" / "file.json"
    ensure_parent(target)
    assert target.parent.is_dir()


# ------------------------------------------------------------ 账本读写
def _record(run_id: str = "20260101T000000Z-demo", **overrides: object) -> RunRecord:
    fields: dict[str, object] = {
        "run_id": run_id,
        "kind": "eval",
        "outcome": "success",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:02+00:00",
        "duration_seconds": 2.0,
        "stages": (StageRecord(name="create", seconds=0.5),),
    }
    fields.update(overrides)
    return RunRecord(**fields)


def test_append_then_read_roundtrips(tmp_path: Path) -> None:
    path = ledger_path(tmp_path)
    append(_record(), path=path)
    loaded = read(path)
    assert loaded.skipped_lines == 0
    assert len(loaded.records) == 1
    record = loaded.records[0]
    assert record.run_id == "20260101T000000Z-demo"
    assert record.stages == (StageRecord(name="create", seconds=0.5),)
    assert record.schema_version == LEDGER_SCHEMA_VERSION


def test_reading_a_missing_ledger_is_an_empty_ledger(tmp_path: Path) -> None:
    """首次运行时账本还不存在，这不是错误。"""
    loaded = read(ledger_path(tmp_path))
    assert loaded.records == ()
    assert loaded.skipped_lines == 0


def test_duplicate_run_id_is_rejected(tmp_path: Path) -> None:
    """同一次运行被写两遍必然让 5.7 的计数虚高，故直接拒绝。"""
    path = ledger_path(tmp_path)
    append(_record(), path=path)
    with pytest.raises(DuplicateRunError, match="拒绝重复写入"):
        append(_record(), path=path)


def test_bad_line_and_unknown_version_are_skipped_and_counted(tmp_path: Path) -> None:
    """坏行与未知 schema 版本都要跳过**并计数**——绝不用当前口径去解释它们。"""
    path = ledger_path(tmp_path)
    append(_record(), path=path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{ 这不是 JSON\n")
        fh.write(json.dumps({"schema_version": 999, "run_id": "未来版本"}) + "\n")
    loaded = read(path)
    assert len(loaded.records) == 1
    assert loaded.skipped_lines == 1
    assert loaded.skipped_unknown_version == 1


def test_summarize_excludes_errors_from_success_rate() -> None:
    """与评测报告同一口径：harness 故障不进成功率分母，但要单独计数。"""
    records = [
        _record("a", outcome="success", duration_seconds=1.0),
        _record("b", outcome="fail", duration_seconds=3.0),
        _record("c", outcome="error", duration_seconds=5.0),
    ]
    summary = summarize(records)
    assert summary["runs"] == 3
    assert summary["outcome_counts"] == {"success": 1, "fail": 1, "error": 1, "skeleton": 0}
    assert summary["success_rate"] == 0.5  # 1 / (3 - 1)
    assert summary["duration_seconds"] == {"mean": 3.0, "max": 5.0}
    # 没有一条有 token 记录 → None（不是 0）。
    assert summary["total_tokens"] is None
    assert summary["total_cost_usd"] is None
    assert summary["runs_with_tokens"] == 0


# ------------------------------------------------------------ 成本口径
def test_price_table_missing_model_returns_none() -> None:
    """模型查不到单价 → None。不内置价格、不猜、更不写 0。"""
    table = load_prices(None)
    assert table.source is None
    usage = TokenUsage(model="any-model", input_tokens=1000, output_tokens=10)
    assert table.cost_usd(usage) is None


def test_price_table_computes_cost_per_million(tmp_path: Path) -> None:
    prices = tmp_path / "prices.json"
    prices.write_text(
        json.dumps({"demo-model": {"input": 3.0, "output": 15.0}}), encoding="utf-8"
    )
    table = load_prices(prices)
    assert table.source == str(prices)
    usage = TokenUsage(model="demo-model", input_tokens=1_000_000, output_tokens=1_000_000)
    assert table.cost_usd(usage) == pytest.approx(18.0)
    assert table.missing((usage,)) == ()


def test_price_table_rejects_malformed_file(tmp_path: Path) -> None:
    """价格表写坏了要显式失败——静默退回空表会把「配置错了」伪装成「没花钱」。"""
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({"demo": {"input": True}}), encoding="utf-8")
    with pytest.raises(ValueError, match="数值型"):
        load_prices(prices)


def test_run_context_records_tokens_and_cost(tmp_path: Path) -> None:
    """``record_tokens`` 在 P0 没有调用点（没有 LLM），这里证明它可执行、口径正确。"""
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({"m": {"input": 1.0, "output": 2.0}}), encoding="utf-8")
    context = RunContext(kind="eval", home=tmp_path, prices_path=prices)
    context.stage("solve", 1.25)
    context.record_tokens("m", input_tokens=1000, output_tokens=500)
    context.finish("success")

    record = context.record
    assert record is not None
    assert record.total_tokens == 1500
    assert record.cost_usd == pytest.approx((1000 * 1.0 + 500 * 2.0) / 1e6)
    assert record.missing_price_models == ()
    assert record.stages == (StageRecord(name="solve", seconds=1.25),)


def test_run_context_without_price_file_keeps_cost_none(tmp_path: Path) -> None:
    """有 token 但没单价：总量照记，费用为 None 并列出缺哪个模型。"""
    context = RunContext(kind="eval", home=tmp_path)
    context.record_tokens("mystery-model", input_tokens=10, output_tokens=5)
    context.finish("success")

    record = context.record
    assert record is not None
    assert record.total_tokens == 15
    assert record.cost_usd is None
    assert record.missing_price_models == ("mystery-model",)


def test_partial_price_coverage_yields_none_not_a_partial_sum(tmp_path: Path) -> None:
    """只要有一个模型缺单价就整体记 None——部分求和会把「算不全」伪装成「很便宜」。"""
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({"known": {"input": 1.0, "output": 1.0}}), encoding="utf-8")
    context = RunContext(kind="eval", home=tmp_path, prices_path=prices)
    context.record_tokens("known", 1_000_000, 0)
    context.record_tokens("unknown", 1_000_000, 0)
    context.finish("success")

    record = context.record
    assert record is not None
    assert record.cost_usd is None
    assert record.missing_price_models == ("unknown",)


# ------------------------------------------------------------ 运行上下文
def test_snapshots_are_numbered_and_written_immediately(tmp_path: Path) -> None:
    """序号前缀让目录按发生顺序可读；写入是立即的（进程崩了也要留下已走过的阶段）。"""
    context = RunContext(kind="eval", home=tmp_path, run_id="20260101T000000Z-snap")
    context.snapshot("create", {"step": 1})
    context.snapshot("solve", {"step": 2})

    snapshots = sorted((context.run_dir / "snapshots").glob("*.json"))
    assert [path.name for path in snapshots] == ["01-create.json", "02-solve.json"]
    payload = json.loads(snapshots[0].read_text(encoding="utf-8"))
    assert payload["stage"] == "create"
    assert payload["state"] == {"step": 1}


def test_finish_writes_summary_and_appends_ledger(tmp_path: Path) -> None:
    context = RunContext(kind="cli", home=tmp_path, run_id="20260101T000000Z-fin", repo="demo")
    context.stage("parse", 0.01)
    context.finish("skeleton")

    summary = json.loads((context.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["outcome"] == "skeleton"
    assert summary["repo"] == "demo"
    assert json.loads(context.ledger_path.read_text(encoding="utf-8"))["run_id"] == (
        "20260101T000000Z-fin"
    )


def test_finish_cannot_run_twice(tmp_path: Path) -> None:
    """重复收尾必然让 5.7 计数虚高，故在写第二行之前就拒绝。"""
    context = RunContext(kind="cli", home=tmp_path)
    context.finish("skeleton")
    with pytest.raises(RuntimeError, match="已收尾"):
        context.finish("skeleton")


def test_escape_flag_is_recorded_from_the_caller(tmp_path: Path) -> None:
    """逃逸由调用方从评测报告里读出来传入，而不是让被测流程自己上报。"""
    context = RunContext(kind="eval", home=tmp_path)
    context.finish("fail", escaped=True)
    record = context.record
    assert record is not None
    assert record.escaped is True
    assert summarize([record])["escape_count"] == 1


# ------------------------------------------------------------ tracing 接线
def test_tracing_without_key_is_reported_not_faked(clean_env: None) -> None:
    """缺 key 时必须显式报告未启用，**且不碰任何变量**——否则 trace 静默丢失。"""
    status = configure_tracing(True)
    assert status.enabled is False
    assert status.reason is not None
    assert "LANGSMITH_API_KEY" in status.reason
    for name in ENV_TRACING:
        assert name not in os.environ


def test_tracing_sets_the_exact_true_string(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """值必须是精确的小写 ``true``：langsmith 判定是 ``== "true"``，``1``/``True`` 都无效。"""
    monkeypatch.setenv(ENV_API_KEY[0], "lsv2_demo_key")
    status = configure_tracing(True)
    assert status.enabled is True
    assert status.project == "default"
    assert status.key_env == ENV_API_KEY[0]
    for name in ENV_TRACING:
        assert os.environ[name] == TRACING_ON == "true"


def test_tracing_writes_the_project_name(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_API_KEY[0], "lsv2_demo_key")
    status = configure_tracing(True, project="nightwatch-demo")
    assert status.project == "nightwatch-demo"
    assert os.environ[ENV_PROJECT[0]] == "nightwatch-demo"


def test_tracing_accepts_the_legacy_key_name(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """旧名仍生效（langsmith 的命名空间是 LANGSMITH 优先、LANGCHAIN 兜底）。"""
    monkeypatch.setenv(ENV_API_KEY[1], "legacy_key")
    status = configure_tracing(True)
    assert status.enabled is True
    assert status.key_env == ENV_API_KEY[1]


def test_disabled_request_leaves_user_environment_untouched(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--trace`` 缺席不等于「关掉 LangSmith」：用户 shell 里可能自有采集。"""
    monkeypatch.setenv(ENV_API_KEY[0], "lsv2_demo_key")
    monkeypatch.setenv(ENV_TRACING[0], TRACING_ON)
    status = configure_tracing(False)
    assert status.enabled is False
    assert status.reason is not None and "未请求" in status.reason
    assert os.environ[ENV_TRACING[0]] == TRACING_ON
