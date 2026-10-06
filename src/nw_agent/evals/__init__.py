"""seed 评测集：固定 golden issue + 固定仓库快照 + 可重复运行的判定。

见 doc/开发路线图.md 第 0 阶段的「建 seed 评测集」与 5.7 的度量表。用例数据在
仓库根的 ``evals/cases/``（不随包发布），判定与编排逻辑在这里。

对齐 5.7 的口径：本包只产出**数字与判定**，不落盘、不打时间戳——落盘与时间戳由
``scripts/run_eval.py`` 负责，这样报告内容在测试里是确定性的。
"""

from __future__ import annotations

from nw_agent.evals.cases import (
    Case,
    EvidenceLevel,
    TaskType,
    VerifySpec,
    iter_case_dirs,
    load_case,
    read_acceptance_files,
    read_repo_files,
)
from nw_agent.evals.checks import L3Result, changed_paths, check_l3
from nw_agent.evals.runner import (
    THRESHOLDS,
    BackendFactory,
    CaseReport,
    build_report,
    run_case,
    run_suite,
)
from nw_agent.evals.solvers import NullSolver, ReferenceSolver, Solver, SolveResult

__all__ = [
    "THRESHOLDS",
    "BackendFactory",
    "Case",
    "CaseReport",
    "EvidenceLevel",
    "L3Result",
    "NullSolver",
    "ReferenceSolver",
    "SolveResult",
    "Solver",
    "TaskType",
    "VerifySpec",
    "build_report",
    "changed_paths",
    "check_l3",
    "iter_case_dirs",
    "load_case",
    "read_acceptance_files",
    "read_repo_files",
    "run_case",
    "run_suite",
]