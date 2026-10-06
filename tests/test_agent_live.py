"""真链路自测：真模型 + 真 E2B 沙箱跑一次任务，断言产出非空补丁。

    set -a; source .env; set +a          # 或直接 export
    conda run -n nightwatch_agents pytest tests/test_agent_live.py -o addopts="" -s

**默认 skip**，与 ``test_backends_e2b.py`` 的真链路段同一姿态：本仓库的离线回归必须能在
没有凭据的机器上全绿，CI 里这条应当是 skip 而不是 fail。

⚠️ **会花钱**：一次运行 = 一个 E2B 沙箱（几十秒的生命周期）+ 若干次模型调用。

⚠️ **凭据必须真的在进程环境里**：``skipif`` 在 import 期求值，而 pytest **不会**加载
``.env``。把 key 只放在 ``.env`` 里会导致这条测试静默跳过（而不是报错）——这正是要用
``source .env`` 的原因。

**不断言模型改对了**：那是评测集（``evals/``）的职责。这里只证明链路通——模型能通过
``task`` 委派、能在沙箱里动文件、补丁能被宿主导出来。真模型甚至可能不委派就直接用
``edit_file`` 改（那条路径同样合法），所以断言落在「补丁非空」而不是「经过了 fix 子代理」。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nw_agent.agents import build_main_agent, resolve_agent_models, run_task
from nw_agent.backends import SandboxConfig, create_backend, read_local_repo
from nw_agent.observability import RunContext

needs_e2b = pytest.mark.skipif(
    not os.environ.get("E2B_API_KEY"), reason="需要 E2B_API_KEY 才能真建云端沙箱"
)
needs_model = pytest.mark.skipif(
    not (os.environ.get("NW_MODEL_MAIN") or "").strip(),
    reason="需要 NW_MODEL_MAIN / NW_MODEL_FIX 才能真跑模型",
)

_INITIAL = (
    '"""示例模块。"""\n\n\ndef add(a: int, b: int) -> int:\n    return a + b\n'
)


def _write_fixture_repo(root: Path) -> Path:
    """现场造一个迷你仓库。

    **刻意不用本仓库自身**：它太大（上传耗时不可控、还带 conda-lock 之类的无关文件），
    且万一被改动会污染工作区。
    """
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "calc.py").write_text(_INITIAL, encoding="utf-8")
    (root / "conftest.py").write_text(
        "import pathlib, sys\nsys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add() -> None:\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    return root


@needs_e2b
@needs_model
def test_real_run_produces_a_patch(tmp_path: Path) -> None:
    """真模型在真沙箱里跑一次「给 add() 补 docstring」，宿主拿回一段 diff。"""
    repo = _write_fixture_repo(tmp_path / "repo")
    host_before = (repo / "calc.py").read_text(encoding="utf-8")
    local = read_local_repo(repo)
    assert local.files, "fixture 仓库应当有文件可上传"
    models = resolve_agent_models()

    context = RunContext(kind="cli", home=tmp_path / "obs", repo=str(repo), backend_kind="e2b")
    with create_backend("e2b", SandboxConfig()) as backend:
        backend.upload_repo(local.files)
        agent = build_main_agent(backend, models)
        result = run_task(
            agent,
            backend,
            "给 calc.py 的 add() 补一个说明参数与返回值的中文 docstring，其它内容不要动。",
            model_label=models.specs()[0],
            recorder=context,
        )
        # 在**沙箱内**读回目标文件——顺带断言「宿主工作区零污染」。
        in_sandbox = backend.execute(f"cat {backend.config.repo_path}/calc.py")
    context.finish("success" if result.patch else "fail")

    assert result.patch, f"没有产出补丁；主代理说：{result.final_text!r}"
    assert result.patch.startswith("diff --git") or "diff --git" in result.patch
    assert in_sandbox.exit_code == 0
    assert "docstring" in in_sandbox.output or '"""' in in_sandbox.output, (
        f"沙箱内的 calc.py 看起来没被改；内容：{in_sandbox.output[:300]}"
    )
    # 宿主侧零污染：补丁是唯一的出口，本地文件不该被动。
    assert (repo / "calc.py").read_text(encoding="utf-8") == host_before

    # token 用量真的记进了账本——5.7 的「单任务 token 成本」由此有真数据。
    assert result.total_tokens > 0, "没拿到任何 token 用量，callback 接线可能断了"
    record = context.record
    assert record is not None
    assert record.total_tokens == result.total_tokens
    assert context.ledger_path.read_text(encoding="utf-8").strip(), "账本应当有至少一行"


@needs_e2b
@needs_model
def test_cli_end_to_end(tmp_path: Path) -> None:
    """CLI 端到端：``python -m nw_agent --repo <local> --task ...`` 能跑出补丁。

    直接调用 ``main()`` 而不是起子进程：这样能断言退出码，也能让覆盖率看得见。
    """
    repo = _write_fixture_repo(tmp_path / "repo")
    from nw_agent.cli.main import main

    code = main(
        [
            "--repo",
            str(repo),
            "--task",
            "给 calc.py 的 add() 补一个中文 docstring。",
        ]
    )
    assert code == 0, "跑出补丁时退出码应当是 0"
    # 宿主工作区不该被动过——CLI 只上传与导出 diff。
    assert (repo / "calc.py").read_text(encoding="utf-8") == _INITIAL


@needs_e2b
@needs_model
def test_sandbox_is_killed_on_exit(tmp_path: Path) -> None:
    """沙箱随 ``with`` 退出即销毁——终止必清理（产品说明 3.3 第 5 条）。

    用 ``e2b.Sandbox.connect`` 反查沙箱是否已不存在：已销毁时应当抛异常。
    """
    import e2b

    repo = _write_fixture_repo(tmp_path / "repo")
    local = read_local_repo(repo)
    with create_backend("e2b", SandboxConfig()) as backend:
        backend.upload_repo(local.files)
        sandbox_id = backend.id
        assert backend.is_alive
    assert not backend.is_alive

    with pytest.raises(Exception):  # noqa: B017 - 已销毁的沙箱连不上，具体异常类型由 SDK 决定
        e2b.Sandbox.connect(sandbox_id)
