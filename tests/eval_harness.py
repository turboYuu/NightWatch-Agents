"""seed 评测集的离线测试夹具。

定位与 ``e2b_double.py`` 相同：**只存在于 ``tests/``，产品代码引用不到**。
``ScriptedSolver`` 尤其如此——它是「按清单尝试越权写入」的攻击脚本，产品包里不该出现。

离线后端走既有范式 ``E2BSandboxBackend._attach(config, SandboxDouble(...))``（见
``test_backends_e2b.py`` 的 ``make_backend``）：跑的是生产后端同一条代码路径，只是把
远端换成宿主机临时目录。因此 ``repo_path`` / ``acceptance_path`` 都必须落在临时目录里
——替身不翻译沙箱绝对路径。
"""

from __future__ import annotations

import posixpath
import shlex
from dataclasses import dataclass, replace
from pathlib import Path

from e2b_double import SandboxDouble
from nw_agent.backends import E2BSandboxBackend, SandboxBackend, SandboxConfig
from nw_agent.evals import BackendFactory, Case, SolveResult

# 与沙箱内布局同构（``/workspace/repo``、``/workspace/acceptance``），只是根换成了临时目录。
REPO_SUBDIR = Path("workspace") / "repo"
ACCEPTANCE_SUBDIR = Path("workspace") / "acceptance"


def make_backend(root: Path, config: SandboxConfig) -> tuple[SandboxBackend, SandboxDouble]:
    """建一个注入 SDK 替身、且已完成创建期准备的离线后端。

    Args:
        root: 替身沙箱根（临时目录），``repo`` 与 ``acceptance`` 都落在它下面。
        config: 期望配置；本函数会把它改写成本地路径后交给后端。

    Returns:
        ``(后端, 替身)``——替身返回给调用方，用于断言 ``kill_calls`` 之类。
    """
    local = replace(
        config,
        repo_path=str(root / REPO_SUBDIR),
        acceptance_path=str(root / ACCEPTANCE_SUBDIR),
    )
    double = SandboxDouble(root)
    return E2BSandboxBackend._attach(local, double), double


def offline_factory(root: Path) -> BackendFactory:
    """按用例 id 分目录的离线工厂（与 ``scripts/run_eval.py`` 的同名实现同构）。

    每条用例一个独立子目录：runner 会为每条用例都调一次工厂，用例之间不该共享文件状态。
    """

    def factory(config: SandboxConfig) -> SandboxBackend:
        case_root = root / str(config.metadata.get("eval_case", "case"))
        case_root.mkdir(parents=True, exist_ok=True)
        backend, _ = make_backend(case_root, config)
        return backend

    return factory


@dataclass(frozen=True)
class Attack:
    """一次越权写入尝试。"""

    relative_path: str
    content: bytes


@dataclass(frozen=True)
class Probe:
    """一次「文件在不在沙箱里」的探测。"""

    relative_path: str


class ScriptedSolver:
    """按清单执行越权动作，用来证明 L3 / 逃逸检测器真的能抓到。

    P0 **只检测不拦截**（写白名单是 P1），所以这里的越权写入**会成功**——被验证的不是
    「拦住了」，而是「写成功了也能被如实计成逃逸」。若哪天检测器失灵，这个 solver 的结果
    会从「candidate 逃逸」变成「干净通过」，测试立刻红。
    """

    def __init__(
        self,
        attacks: tuple[Attack, ...],
        *,
        probes: tuple[Probe, ...] = (),
    ) -> None:
        self._attacks = attacks
        self._probes = probes

    def solve(self, case: Case, backend: SandboxBackend) -> SolveResult:
        """先探测诱饵是否在沙箱里，再逐条写入越权文件。"""
        notes: list[str] = []
        for probe in self._probes:
            result = backend.execute(f"test -e {shlex.quote(probe.relative_path)}")
            presence = "在沙箱里" if result.exit_code == 0 else "不在沙箱里"
            notes.append(f"探测 {probe.relative_path}：{presence}")
        repo = backend.config.repo_path
        for attack in self._attacks:
            target = posixpath.join(repo, attack.relative_path)
            responses = backend.upload_tree([(target, attack.content)])
            failed = [response for response in responses if response.error]
            notes.append(f"写入 {attack.relative_path}：{'失败' if failed else '成功'}")
        return SolveResult(detail="；".join(notes) or "scripted solver：无动作")
