"""评测求解器接缝：把「谁来解决这条 issue」与「怎么评」解耦。

P0 还没有 Agent（P1 才有），但评测框架不能等它——**用例数据与判定逻辑先钉死**，
P1 只需新增一个 :class:`Solver` 实现，用例与报告格式一律不动。

本模块只放**产品会用到**的 solver：
- :class:`NullSolver`——什么都不做的对照组，用来证明用例「改前必红」。
- :class:`ReferenceSolver`——套用用例自带的 golden 补丁，用来证明 harness 能识别成功。

**攻击型 solver 不放这里**（见 ``tests/eval_harness.py`` 的 ``ScriptedSolver``）：
产品包里不该出现「按清单尝试越权写」的代码，哪怕只在测试里用到。

**判定权不在 solver**：solver 返回的 :class:`SolveResult` 只是自述，成功与否由
runner 依据验收命令与 L3 护栏决定——否则 solver 可以自称成功。
"""

from __future__ import annotations

import posixpath
import shlex
from dataclasses import dataclass
from typing import Protocol

from nw_agent.backends import SandboxBackend
from nw_agent.evals.cases import Case


@dataclass(frozen=True)
class SolveResult:
    """solver 对本次求解的自述。"""

    detail: str
    """人类可读的过程摘要，进报告便于归因。"""

    tokens: int | None = None
    """本次求解消耗的 token。P0 无模型调用，恒为 ``None``——**不是 0**，
    以免被下游读成「零成本」。P1 的 DeepAgentSolver 在这里填真值。"""


class Solver(Protocol):
    """求解一条用例。实现可以是任意对象，只要满足本协议。

    刻意不声明 ``name`` 属性：名字由 runner 从 ``type(solver).__name__`` 派生，
    少一处 mypy 在「实例属性 vs property」上的兼容雷。
    """

    def solve(self, case: Case, backend: SandboxBackend) -> SolveResult:
        """在已上传初始快照的沙箱里求解。

        Args:
            case: 用例（含 issue 正文与 target_files）。
            backend: 已就绪的沙箱后端，cwd 为仓库根。

        Returns:
            自述结果。

        Raises:
            任意异常：视为 harness/solver 故障，由 runner 记为 ``error`` 而非 ``fail``。
        """
        ...


class NullSolver:
    """什么都不做。用来证明每条用例「改前必红」。

    它的成功率**理应为 0**——这不是失败，而是用例有效性的证据：若 null 也能过，
    说明验收测试没有真正约束到改动。
    """

    def solve(self, case: Case, backend: SandboxBackend) -> SolveResult:
        """不做任何修改。"""
        return SolveResult(detail="null solver：刻意不做任何修改")


class ReferenceSolver:
    """把用例自带的 ``reference.patch`` 打进沙箱仓库。

    补丁**经验收目录上传**（``backend.config.acceptance_path``，在 git 仓库之外），
    绝不放仓库内——否则它会以未跟踪文件的形式混进 ``export_diff`` 的补丁里，
    把 L3 判定带偏，也等于把答案摆在了 solver 看得见的地方。
    """

    def solve(self, case: Case, backend: SandboxBackend) -> SolveResult:
        """上传并 ``git apply`` golden 补丁。

        Raises:
            ValueError: 用例没有 ``reference.patch``（ReferenceSolver 用错了用例）。
            RuntimeError: 补丁打不上——这是**评测数据与快照不同步**，属 harness
                故障（记 ``error``），不该混进 solver 的失败率里。
        """
        patch = case.reference_patch
        if patch is None:
            raise ValueError(f"{case.case_id}：用例没有 reference.patch，无法用 ReferenceSolver")

        content = patch.read_bytes()
        backend.upload_tree([("reference.patch", content)], base=backend.config.acceptance_path)
        # cwd 恒为仓库根（见 SandboxBackend.execute），故用相对路径指到验收目录。
        relative = posixpath.relpath(
            backend.config.acceptance_path, backend.config.repo_path
        )
        result = backend.execute(f"git apply {shlex.quote(f'{relative}/reference.patch')}")
        if result.exit_code != 0:
            raise RuntimeError(
                f"{case.case_id}：reference.patch 打不上（exit {result.exit_code}）："
                f"{result.output[:500]}。这通常意味着补丁与 repo/ 快照不同步，"
                "请重新生成补丁。"
            )
        return SolveResult(detail=f"应用 reference.patch（{len(content)} 字节）")
