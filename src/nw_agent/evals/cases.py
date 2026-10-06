"""seed 评测集的用例模型与加载器。

用例目录布局（``evals/cases/<case_id>/``）::

    case.json         用例元数据（本模块解析并逐字段校验）
    repo/             初始仓库快照——**solver 跑之前**上传到沙箱
    acceptance/       隐藏验收测试——**solver 跑完、补丁抓完**才上传到 ACCEPTANCE_PATH
    reference.patch   可选；golden 补丁，供 ReferenceSolver 用（与 repo/ 并列，绝不进 repo/）

**为什么验收测试要藏着**：它是判据，「改前必红、改后必绿」全靠它在初始快照下失败。
若放进 ``repo/``，它既会进 ``export_diff`` 的补丁（污染 L3 判定），又可能被 solver 改绿
——那样评测就变成了「改测试」而不是「改代码」。

**为什么加载期校验这么重**：评测数据坏了（patch 与快照不同步、decoy 凭据被
``.gitignore`` 吞掉）不会报错，只会让指标静默偏移，事后极难归因。因此把校验放在
``load_case``：**非法的用例根本构造不出来**，与 ``SandboxConfig.__post_init__``
拒绝凭据 env 是同一哲学。``json.load`` 的 ``Any`` 也就在这里被关住，不外泄到 runner。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from nw_agent.backends import WORKDIR

# 与 doc/开发路线图.md 5.8「任务类型 → 流程与验证映射」对齐；判不出的类型不入集。
TaskType = Literal["docstring", "bugfix", "injection"]

# 分级验证的证据层级（见 doc/NightWatch产品说明.md 3.1）。
# L3（未越界修改）不是「证据」而是强制护栏，故不在此列——它恒被检查。
EvidenceLevel = Literal["L1", "L2"]

# 用例命令里不许出现沙箱根路径（WORKDIR）。命令里写了它，说明用例在假设某条链路的
# 绝对路径，而离线链路（SDK 替身）不翻译绝对路径——两边命令必须逐字相同，故直接拒绝。
# 取常量而非字面量，也让「本包不出现沙箱绝对路径」这条规则能被一条源码断言守住。
_FORBIDDEN_IN_COMMANDS = (WORKDIR,)

# 评测 fixture 里不允许出现的文件：`.gitignore` 会让 `git add -N .` **静默**漏掉
# solver 新增的文件，于是「改了」被判成「没改」——假绿，且极难发现。
_FORBIDDEN_REPO_FILES = (".gitignore",)


@dataclass(frozen=True)
class VerifySpec:
    """一条用例的验收方式。"""

    command: str
    """在沙箱内执行的验收命令。**只能是相对路径**——``execute`` 的 cwd 恒为仓库根
    （见 ``SandboxBackend.execute`` 的工作目录约定），这样离线与 E2B 两条链路跑的是
    同一条命令字符串。"""

    evidence_level: EvidenceLevel
    """该命令提供的正确性证据层级。L1 = 仓库测试通过，L2 = 编译/类型/lint 通过。
    无论哪一层，仍须与 L3 同时成立才算通过（``test_passed = (L1 或 L2) 且 L3``）。"""


@dataclass(frozen=True)
class Case:
    """一条 golden issue。frozen：它会被反复引用，可变对象在这条链路上是隐患。

    列表字段用 ``tuple`` 而非 ``list``：frozen dataclass 里的 list 仍可被就地修改，
    等于冻结失败。
    """

    case_id: str
    title: str
    task_type: TaskType
    issue_text: str
    """交给 solver 的 Issue 正文。注入用例的载荷就写在这里。"""
    target_files: tuple[str, ...]
    """允许被修改的文件（L3 白名单）。"""
    forbidden_paths: tuple[str, ...]
    """显式禁止出现在补丁里的路径（如 ``.env``、``.ssh/id_rsa``）。它与
    ``target_files`` 的差集关系已在加载期校验，故判定强度上不超出白名单——
    保留它是为了让越权报告**指名道姓**（「它试图写 .env」比「写了白名单外的文件」
    醒目得多），并为 P1 的写拦截留接口。"""
    decoy_files: tuple[str, ...]
    """相对 ``repo/`` 的、**刻意放置的凭据诱饵**。加载期断言它们确实存在——
    因为它们很可能被根 ``.gitignore`` 的 ``.env`` 规则静默吞掉，那样用例看着存在、
    实则没有诱饵，注入验证就空转了。仅注入用例非空。"""
    verify: VerifySpec
    case_dir: Path
    repo_dir: Path
    acceptance_dir: Path
    reference_patch: Path | None

    @property
    def target_set(self) -> set[str]:
        """``target_files`` 的集合形态（L3 判定用）。"""
        return set(self.target_files)


def iter_case_dirs(root: Path) -> list[Path]:
    """列举用例根目录下的用例目录（含 ``case.json`` 的子目录），按名字排序。

    排序是为了让报告可比——两次运行的用例顺序不同会淹没真正的差异。
    """
    return sorted(
        (path for path in root.iterdir() if (path / "case.json").is_file()),
        key=lambda path: path.name,
    )


def load_case(case_dir: Path) -> Case:
    """解析并校验一条用例。

    Args:
        case_dir: 用例目录（含 ``case.json``）。

    Returns:
        校验通过的 :class:`Case`。

    Raises:
        ValueError: 字段缺失 / 类型不符 / 取值非法，或数据自检失败（``id`` 与目录名
            不一致、``repo/`` 里藏了 ``.gitignore``、注入用例缺诱饵、``reference.patch``
            的路径前缀不是 ``a/`` 等）。信息里带用例路径，便于直接定位。
    """
    manifest = case_dir / "case.json"
    if not manifest.is_file():
        raise ValueError(f"用例缺少 case.json：{case_dir}")

    with manifest.open(encoding="utf-8") as fh:
        data: object = json.load(fh)  # 立刻绑成 object，把 Any 关在本函数内

    where = str(manifest)
    case_id = _require_str(data, "id", where)
    if case_id != case_dir.name:
        raise ValueError(f"{where}：id={case_id!r} 与目录名 {case_dir.name!r} 不一致")

    task_type = cast(
        TaskType, _require_choice(data, "task_type", where, ("docstring", "bugfix", "injection"))
    )
    target_files = _require_str_tuple(data, "target_files", where)
    if not target_files:
        raise ValueError(f"{where}：target_files 不能为空（L3 需要白名单）")

    forbidden_paths = _require_str_tuple(data, "forbidden_paths", where, default=())
    overlap = set(target_files) & set(forbidden_paths)
    if overlap:
        raise ValueError(f"{where}：forbidden_paths 与 target_files 重叠：{sorted(overlap)}")

    verify = _parse_verify(data, where)
    decoy_files = _require_str_tuple(data, "decoy_files", where, default=())

    repo_dir = case_dir / "repo"
    acceptance_dir = case_dir / "acceptance"
    for required_dir in (repo_dir, acceptance_dir):
        if not required_dir.is_dir() or not any(required_dir.rglob("*")):
            raise ValueError(f"{where}：{required_dir.name}/ 不存在或为空")

    _assert_no_gitignore(repo_dir, where)
    _assert_decoys_present(repo_dir, decoy_files, where)

    patch = case_dir / "reference.patch"
    reference_patch = patch if patch.is_file() else None
    if reference_patch is not None:
        _assert_patch_prefix(reference_patch, where)

    return Case(
        case_id=case_id,
        title=_require_str(data, "title", where),
        task_type=task_type,
        issue_text=_require_str(data, "issue_text", where),
        target_files=target_files,
        forbidden_paths=forbidden_paths,
        decoy_files=decoy_files,
        verify=verify,
        case_dir=case_dir,
        repo_dir=repo_dir,
        acceptance_dir=acceptance_dir,
        reference_patch=reference_patch,
    )


def read_repo_files(case: Case) -> list[tuple[str, bytes]]:
    """把 ``repo/`` 递归读成 ``(相对路径, 内容)``，可直接喂给 ``upload_repo``。

    相对路径一律用 POSIX 分隔符：沙箱是 Linux，Windows 上的宿主不该把 ``\\`` 带进去。
    跳过 ``.git``（若 fixture 里误留），其余原样保留——**凭据过滤不在这里做**，
    那是 ``select_upload_files`` 的职责，两件事分开才能各自被单独测试。
    """
    return _read_tree(case.repo_dir)


def read_acceptance_files(case: Case) -> list[tuple[str, bytes]]:
    """把 ``acceptance/`` 递归读成 ``(相对路径, 内容)``。

    内容与 ``read_repo_files`` 同构，但这两个目录的**去向不同**（仓库根 vs
    ``ACCEPTANCE_PATH``），刻意分成两个入口，避免调用处写错 base。
    """
    return _read_tree(case.acceptance_dir)


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------
def _read_tree(root: Path) -> list[tuple[str, bytes]]:
    """递归读目录，返回按路径排序的 ``(POSIX 相对路径, bytes)`` 列表。"""
    collected: list[tuple[str, bytes]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        collected.append((path.relative_to(root).as_posix(), path.read_bytes()))
    return collected


def _assert_no_gitignore(repo_dir: Path, where: str) -> None:
    """``repo/`` 里不得有 ``.gitignore``（见 ``_FORBIDDEN_REPO_FILES`` 的理由）。"""
    for name in _FORBIDDEN_REPO_FILES:
        if any(path.name == name for path in repo_dir.rglob("*")):
            raise ValueError(
                f"{where}：repo/ 里出现了 {name}——它会让 git add -N 静默漏掉 solver "
                "新增的文件，导致「改了」被判成「没改」。请移出该文件。"
            )


def _assert_decoys_present(repo_dir: Path, decoy_files: tuple[str, ...], where: str) -> None:
    """断言声明的诱饵文件确实在 ``repo/`` 里。

    这是防「诱饵被 ``.gitignore`` 静默吞掉」的唯一手段：用例看起来完好，实际没有
    诱饵，注入验证就变成空转。
    """
    missing = [name for name in decoy_files if not (repo_dir / name).is_file()]
    if missing:
        raise ValueError(
            f"{where}：声明的诱饵文件不在 repo/ 里：{missing}。"
            "常见成因是根 .gitignore 的 .env 规则把它们吞了——"
            "需要在 .gitignore 里加否定规则，或换用非被忽略的文件名。"
        )


def _assert_patch_prefix(patch: Path, where: str) -> None:
    """``reference.patch`` 必须是 ``git diff`` 生成的（路径前缀 ``a/``）。

    手搓的 ``diff -u`` 输出前缀是裸路径，``git apply`` 默认按 ``-p1`` 会剥错，
    报错信息又只有一句「patch does not apply」，排查成本高。在这里拦下最便宜。
    """
    text = patch.read_text(encoding="utf-8")
    headers = [line for line in text.splitlines() if line.startswith("diff --git ")]
    if not headers:
        raise ValueError(f"{where}：reference.patch 里没有 diff --git 行，疑似不是补丁")
    # 路径含空格时 git 会给它们加引号（`diff --git "a/my file.py" "b/my file.py"`），
    # 故不能用 ``split(" ")[2]``——那不认被引号包住的 a/ 前缀。
    bad = [line for line in headers if not re.search(r'["\s]a/', line)]
    if bad:
        raise ValueError(
            f"{where}：reference.patch 的路径缺少 a/ 前缀（疑似用 diff -u 生成）：{bad[0]}。"
            "请用 `git diff` 生成，使 git apply 的默认 -p1 能剥对。"
        )


def _parse_verify(data: object, where: str) -> VerifySpec:
    """解析 ``verify`` 段。"""
    raw = _require_mapping(data, "verify", where)
    command = _require_str(raw, "command", where)
    for literal in _FORBIDDEN_IN_COMMANDS:
        if literal in command:
            raise ValueError(
                f"{where}：verify.command 含沙箱根路径 {literal!r}——"
                "离线链路不翻译绝对路径，两条链路的命令必须逐字相同。请改用相对路径。"
            )
    level = cast(EvidenceLevel, _require_choice(raw, "evidence_level", where, ("L1", "L2")))
    return VerifySpec(command=command, evidence_level=level)


# --- 逐字段校验：Any 的终点站 -------------------------------------------------
# 下面几个 helper 的入参一律是 ``object``，返回值一律是具体类型；`cast` 是刻意的：
# 它让「JSON 的键一定是 str」这条运行时事实显式化，同时挡住 mypy strict 的
# ``warn_return_any``（不 cast 的话 dict 取值是 Any，会顺着返回类型扩散到全项目）。
def _require_mapping(obj: object, key: str, where: str) -> dict[str, object]:
    value = _field(obj, key, where)
    if not isinstance(value, dict):
        raise ValueError(f"{where}：{key} 必须是对象，实际是 {type(value).__name__}")
    return cast("dict[str, object]", value)


def _field(obj: object, key: str, where: str) -> object:
    if not isinstance(obj, dict):
        raise ValueError(f"{where}：顶层必须是 JSON 对象")
    if key not in obj:
        raise ValueError(f"{where}：缺少字段 {key!r}")
    return cast("dict[str, object]", obj)[key]


def _require_str(obj: object, key: str, where: str) -> str:
    value = _field(obj, key, where)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}：{key} 必须是非空字符串")
    return value


def _require_str_tuple(
    obj: object,
    key: str,
    where: str,
    *,
    default: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if default is not None and isinstance(obj, dict) and key not in obj:
        return default
    value = _field(obj, key, where)
    if not isinstance(value, list):
        raise ValueError(f"{where}：{key} 必须是字符串数组")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{where}：{key} 的元素必须是非空字符串")
        if item.startswith("/") or ".." in Path(item).parts:
            raise ValueError(f"{where}：{key} 的元素必须是仓库内相对路径，实际 {item!r}")
        items.append(item)
    return tuple(items)


def _require_choice(obj: object, key: str, where: str, allowed: tuple[str, ...]) -> str:
    value = _require_str(obj, key, where)
    if value not in allowed:
        raise ValueError(f"{where}：{key}={value!r} 不在允许取值 {list(allowed)} 内")
    return value
