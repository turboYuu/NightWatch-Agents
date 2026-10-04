"""``E2BSandboxBackend`` 的行为回归。

分两段：

- **离线段**（默认跑）：注入 ``SandboxDouble``（e2b SDK 的测试替身，见
  ``tests/e2b_double.py``），不联网、不花 E2B 预算，覆盖后端的全部逻辑。这是删掉
  本地假沙箱后保住的回归——后端的行为不再靠「另一个无隔离实现」背书。
- **在线段**（默认 skip）：真建云端沙箱，只在有 Key 的环境手动跑：

      conda run -n nightwatch_agents pytest tests/test_backends_e2b.py -o addopts="" -s

  它把「代码从没被执行过」和「代码跑过」区分开；没有它，后端就只能靠 lint 与类型
  检查背书。
"""

import os
import shutil
import time
from pathlib import Path

import pytest

from e2b_double import SandboxDouble
from nw_agent.backends import (
    REPO_PATH,
    E2BSandboxBackend,
    SandboxClosedError,
    SandboxConfig,
)

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="宿主 PATH 上没有 git")
needs_key = pytest.mark.skipif(
    not os.environ.get("E2B_API_KEY"), reason="需要 E2B_API_KEY 才能真建云端沙箱"
)

# 仓库根相对替身沙箱根的位置，与 ``REPO_PATH``（/workspace/repo）同构。
REPO_SUBDIR = Path("workspace") / "repo"


def make_backend(tmp_path: Path) -> tuple[E2BSandboxBackend, SandboxDouble]:
    """建一个注入 SDK 替身、且已完成创建期准备的离线后端。

    ``repo_path`` 指向临时目录而非默认的 ``/workspace/repo``——替身在宿主机上跑命令，
    不翻译沙箱绝对路径，故把仓库根落在真实存在的临时目录里（见 ``e2b_double`` 的局限）。
    """
    config = SandboxConfig(repo_path=str(tmp_path / REPO_SUBDIR))
    double = SandboxDouble(tmp_path)
    return E2BSandboxBackend._attach(config, double), double


# ---------------------------------------------------------------- 生命周期
def test_id_is_sandbox_id(tmp_path: Path) -> None:
    """``id`` 取自 SDK 的 ``sandbox_id``，resume 时据此判断句柄是否仍有效。"""
    sb, double = make_backend(tmp_path)
    assert sb.id == double.sandbox_id


def test_default_repo_path_is_sandbox_workspace() -> None:
    """默认仓库根是沙箱内的 ``/workspace/repo``（离线测试另指到临时目录）。"""
    assert REPO_PATH == "/workspace/repo"
    assert SandboxConfig().repo_path == REPO_PATH


def test_context_manager_kills_underlying_sandbox_on_exit(tmp_path: Path) -> None:
    """with 退出后后端不再存活，且真的调了 SDK 的 kill。"""
    sb, double = make_backend(tmp_path)
    with sb:
        assert sb.is_alive is True
    assert sb.is_alive is False
    assert double.kill_calls == 1


def test_kill_is_idempotent_and_calls_sdk_once(tmp_path: Path) -> None:
    """重复 kill 不得抛异常，且 SDK 的 kill 只被调一次。"""
    sb, double = make_backend(tmp_path)
    sb.kill()
    sb.kill()
    assert double.kill_calls == 1


def test_kill_survives_exception_in_context(tmp_path: Path) -> None:
    """上下文里抛业务异常时，沙箱仍须被清理，且原异常不被掩盖。"""
    sb, double = make_backend(tmp_path)
    with pytest.raises(ValueError, match="业务异常"), sb:
        raise ValueError("业务异常")
    assert double.kill_calls == 1


def test_actions_after_kill_raise(tmp_path: Path) -> None:
    """销毁后再操作应显式失败，而不是拿到语焉不详的远端错误。"""
    sb, _ = make_backend(tmp_path)
    sb.kill()
    with pytest.raises(SandboxClosedError):
        sb.execute("true")
    with pytest.raises(SandboxClosedError):
        sb.upload_tree([("/x.txt", b"x")])
    with pytest.raises(SandboxClosedError):
        sb.export_diff()


# -------------------------------------------------------------------- 上传
def test_upload_then_download_roundtrip(tmp_path: Path) -> None:
    """上传的内容应与下载回来的字节完全一致。"""
    sb, _ = make_backend(tmp_path)
    target = str(tmp_path / "a.txt")
    sb.upload_tree([(target, b"hello")])
    (resp,) = sb.download_files([target])
    assert resp.content == b"hello"
    assert resp.error is None


def test_upload_creates_parent_dirs(tmp_path: Path) -> None:
    """深层路径不预先建目录也应上传成功（接口硬要求）。"""
    sb, _ = make_backend(tmp_path)
    (resp,) = sb.upload_tree([(str(tmp_path / "deep/nested/dir/f.txt"), b"x")])
    assert resp.error is None


def test_upload_without_base_needs_absolute_path(tmp_path: Path) -> None:
    """不给 base 时路径必须绝对——避免把相对路径静默落到意外位置。"""
    sb, _ = make_backend(tmp_path)
    (resp,) = sb.upload_tree([("relative.txt", b"x")])
    assert resp.error == "invalid_path"


def test_upload_partial_success_keeps_going(tmp_path: Path) -> None:
    """一个文件失败不得影响其余文件（deepagents 要求的部分成功语义）。"""
    sb, _ = make_backend(tmp_path)
    responses = sb.upload_tree(
        [
            (str(tmp_path / "ok.txt"), b"y"),
            ("bad-relative.txt", b"x"),
            (str(tmp_path / "ok2.txt"), b"z"),
        ]
    )
    assert [r.error for r in responses] == [None, "invalid_path", None]


def test_upload_falls_back_to_per_file_on_batch_failure(tmp_path: Path) -> None:
    """批量写入整体失败时应降级为逐文件重试，最终仍然全部成功。"""
    config = SandboxConfig(repo_path=str(tmp_path / REPO_SUBDIR))
    double = SandboxDouble(tmp_path, write_files_error=RuntimeError("批量接口炸了"))
    sb = E2BSandboxBackend._attach(config, double)

    responses = sb.upload_tree(
        [(str(tmp_path / "a.txt"), b"1"), (str(tmp_path / "b.txt"), b"2")]
    )
    assert [r.error for r in responses] == [None, None]
    assert (tmp_path / "a.txt").read_bytes() == b"1"


def test_upload_repo_raises_when_a_file_fails(tmp_path: Path) -> None:
    """upload_repo 有失败项时应抛 RuntimeError，让调用方丢弃沙箱重建。

    制造失败的方式：在目标路径先放一个**目录**，让写文件必然失败——比用 ``..``
    路径越界更贴近真实失败，也不依赖替身的护栏。
    """
    sb, _ = make_backend(tmp_path)
    (tmp_path / REPO_SUBDIR / "blocked.py").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="上传失败"):
        sb.upload_repo([("ok.py", b"x"), ("blocked.py", b"y")])


def test_upload_files_delegates_to_upload_tree(tmp_path: Path) -> None:
    """deepagents 的 ``upload_files`` 入口应委托给 ``upload_tree``，不是各写一遍。"""
    sb, _ = make_backend(tmp_path)
    responses = sb.upload_files([(str(tmp_path / "x.txt"), b"1")])
    assert [r.error for r in responses] == [None]


# -------------------------------------------------------------------- 下载
def test_download_missing_file_reports_file_not_found(tmp_path: Path) -> None:
    """下载不存在的文件应逐条报错，而不是整体抛异常。"""
    sb, _ = make_backend(tmp_path)
    (resp,) = sb.download_files([str(tmp_path / "nope.txt")])
    assert resp.error == "file_not_found"
    assert resp.content is None


def test_download_relative_path_reports_invalid_path(tmp_path: Path) -> None:
    """非绝对路径在调用 SDK 之前就被拦下。"""
    sb, _ = make_backend(tmp_path)
    (resp,) = sb.download_files(["relative.txt"])
    assert resp.error == "invalid_path"


def test_download_directory_reports_is_directory(tmp_path: Path) -> None:
    """下载目录应报 is_directory（与 langchain_e2b 的字面量一致）。"""
    sb, _ = make_backend(tmp_path)
    sb.upload_tree([(str(tmp_path / "dir/f.txt"), b"x")])
    (resp,) = sb.download_files([str(tmp_path / "dir")])
    assert resp.error == "is_directory"


# -------------------------------------------------------------------- 执行
def test_execute_captures_stdout_stderr_and_exit_code(tmp_path: Path) -> None:
    """stdout / stderr 合并进 output，退出码原样透传（非零走 SDK 异常路径）。"""
    sb, _ = make_backend(tmp_path)
    result = sb.execute("echo out; echo err 1>&2; exit 3")
    assert "out" in result.output
    assert "err" in result.output
    assert result.exit_code == 3


def test_execute_runs_in_repo_root(tmp_path: Path) -> None:
    """execute 的工作目录是 ``config.repo_path``。"""
    sb, _ = make_backend(tmp_path)
    assert sb.execute("pwd").output.strip().endswith(str(REPO_SUBDIR))


def test_execute_timeout_returns_124(tmp_path: Path) -> None:
    """超时用 124 表示，与 langchain_e2b.TIMEOUT_EXIT_CODE 对齐。"""
    sb, _ = make_backend(tmp_path)
    assert sb.execute("sleep 5", timeout=1).exit_code == 124


# --------------------------------------------------------- 创建期准备
def test_attach_prepares_workspace(tmp_path: Path) -> None:
    """``_attach`` 应建出仓库目录并探测 git。"""
    sb, double = make_backend(tmp_path)
    assert (tmp_path / REPO_SUBDIR).is_dir()
    assert any("git --version" in call for call in double.commands.calls)


def test_attach_kills_sandbox_when_workspace_prepare_fails(tmp_path: Path) -> None:
    """创建期准备失败时必须**主动销毁**已建好的沙箱，且原异常照常抛出。

    否则这个沙箱再也没人引用得到（``create`` 抛错时句柄没返回给任何人），只能等
    E2B 的 on_timeout 兜底回收——白占一个沙箱，最长 30 分钟。

    制造失败：把 repo_path 指到一个「父级是文件」的路径上，``mkdir -p`` 必然失败。
    """
    blocked = tmp_path / "blocked"
    blocked.write_text("occupied")
    config = SandboxConfig(repo_path=str(blocked / "repo"))
    double = SandboxDouble(tmp_path)

    with pytest.raises(RuntimeError, match="沙箱环境准备失败"):
        E2BSandboxBackend._attach(config, double)
    assert double.kill_calls == 1


# --------------------------------------------------------- git 基线与 diff
@needs_git
def test_upload_repo_builds_single_baseline_commit(tmp_path: Path) -> None:
    """upload_repo 后仓库应恰好有一个基线提交。"""
    sb, _ = make_backend(tmp_path)
    sb.upload_repo([("a.py", b"x = 1\n")])
    result = sb.execute("git rev-list --count HEAD")
    assert result.exit_code == 0
    assert result.output.strip() == "1"


@needs_git
def test_export_diff_is_empty_when_untouched(tmp_path: Path) -> None:
    """刚上传完不应有 diff。"""
    sb, _ = make_backend(tmp_path)
    sb.upload_repo([("a.py", b"x = 1\n")])
    assert sb.export_diff() == ""


@needs_git
def test_export_diff_includes_modified_file(tmp_path: Path) -> None:
    """修改已上传文件应体现在 diff 的增删行里。"""
    sb, _ = make_backend(tmp_path)
    sb.upload_repo([("a.py", b"x = 1\n")])
    sb.execute('printf "x = 2\\n" > a.py')
    diff = sb.export_diff()
    assert "-x = 1" in diff
    assert "+x = 2" in diff
    assert diff.endswith("\n")  # 规范化结尾换行，保证可直接 git apply


@needs_git
def test_export_diff_includes_new_untracked_file(tmp_path: Path) -> None:
    """新增的未跟踪文件应以 new file mode 出现在补丁里，而不是被静默丢弃。"""
    sb, _ = make_backend(tmp_path)
    sb.upload_repo([("a.py", b"x = 1\n")])
    sb.execute("echo new > c.py")
    assert "new file mode" in sb.export_diff()


@needs_git
def test_export_diff_survives_git_add(tmp_path: Path) -> None:
    """agent 执行过 git add 之后，diff 仍须非空。

    这条把「用 git diff HEAD 而非裸 git diff」的设计决策钉成回归：裸 diff 只比
    「工作树 vs 索引」，改动一旦进了索引就会**静默返回空**。
    """
    sb, _ = make_backend(tmp_path)
    sb.upload_repo([("a.py", b"x = 1\n")])
    sb.execute('printf "x = 2\\n" > a.py')
    sb.execute("git add -A")
    assert sb.export_diff().strip()


# ---------------------------------------------------------------- 安全约束
def test_rejects_credential_like_envs() -> None:
    """凭据类环境变量必须在构造时就被拒绝（安全红线落到代码）。"""
    with pytest.raises(ValueError, match="凭据"):
        SandboxConfig(envs={"GITHUB_TOKEN": "x"})


# ---------------------------------------------------------------- 全链路
@needs_git
def test_full_lifecycle_upload_execute_diff_kill(tmp_path: Path) -> None:
    """「上传 → 执行 → 导出 diff → 销毁」的完整闭环（离线，经 SDK 替身）。"""
    sb, double = make_backend(tmp_path)
    sb.upload_repo([("app.py", b"def foo():\n    return 1\n")])
    assert sb.execute("python3 -c 'import app; print(app.foo())'").output.strip() == "1"

    sb.execute("printf 'def foo():\\n    return 2\\n' > app.py")
    diff = sb.export_diff()
    assert "app.py" in diff
    assert "-    return 1" in diff
    assert "+    return 2" in diff

    sb.kill()
    assert sb.is_alive is False
    assert double.kill_calls == 1


# ------------------------------------------------------------ 在线真实链路
@needs_key
def test_full_lifecycle_against_real_sandbox() -> None:
    """创建 → 上传 → 执行 → 导出 diff → 销毁 的完整闭环（真云端）。"""
    with E2BSandboxBackend.create(SandboxConfig(timeout_seconds=600)) as sb:
        assert sb.id
        responses = sb.upload_repo([("app.py", b"x = 1\n")])
        assert [r.error for r in responses] == [None]

        assert sb.execute("python3 -c 'import app; print(app.x)'").output.strip() == "1"

        sb.execute("printf 'x = 2\\n' > app.py")
        diff = sb.export_diff()
        assert "-x = 1" in diff
        assert "+x = 2" in diff


@needs_key
def test_duplicate_kill_is_idempotent() -> None:
    """重复销毁不得抛异常——正常收尾 kill 一次、finally 再兜一次是常态。"""
    sb = E2BSandboxBackend.create(SandboxConfig(timeout_seconds=600))
    sb.kill()
    sb.kill()


@needs_key
def test_cold_start_latency_is_recorded() -> None:
    """记录冷启动耗时，对应路线图 P0 的「沙箱预热基准」验收项。

    用 ``-s`` 跑才看得到打印；量出来的是「创建沙箱 → 上传 → 跑通 python」，
    之所以不是 pytest，是因为默认 base 模板不保证装了 pytest。
    """
    started = time.monotonic()
    with E2BSandboxBackend.create(SandboxConfig(timeout_seconds=600)) as sb:
        sb.upload_repo([("app.py", b"x = 1\n")])
        assert sb.execute("python3 -c 'import app'").exit_code == 0
    print(f"\n[E2B 冷启动] 创建 → 上传 → 跑通 python：{time.monotonic() - started:.1f}s")
