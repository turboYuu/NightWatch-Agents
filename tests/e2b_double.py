"""``e2b.Sandbox`` 的测试替身。

⚠️ 这**不是一个后端**，而是 **e2b SDK 的双胞胎**：只在测试里经
``E2BSandboxBackend(config, sandbox=SandboxDouble(...))`` 注入，用来在不联网、不花
E2B 预算的前提下覆盖后端的全部逻辑。它不在 ``nw_agent`` 包内，``create_backend``
也造不出它，因此生产代码无法引用——这正是它与「本地假沙箱」的本质区别。

行为对齐真实 SDK 的两处关键点：
- ``commands.run`` 在**非零退出时抛 ``CommandExitException``**（已核对 e2b 2.52 的
  ``CommandHandle.wait``：非零退出走异常路径，不返回带码的结果），替身照此抛出真实
  的 ``e2b.CommandExitException``，从而覆盖后端真正走的那条分支。
- 超时抛真实的 ``e2b.TimeoutException``。

命令用本机 ``subprocess`` 在 ``cwd``（即调用方给的 ``config.repo_path``）下执行，
因此 git 基线与 diff 的行为是真实的。**局限**：不翻译命令字符串里的沙箱绝对路径，
故测试一律使用相对路径命令，并把 ``SandboxConfig.repo_path`` 指到临时目录。
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from e2b import CommandExitException, TimeoutException
from e2b.sandbox.commands.command_handle import CommandResult


class _Commands:
    """``sandbox.commands`` 的替身：本地 subprocess 执行 + 调用记录。"""

    def __init__(self, root: Path) -> None:
        self._root = root
        self.calls: list[str] = []

    def run(
        self,
        command: str,
        *,
        cwd: str | None = None,
        envs: dict[str, str] | None = None,
        timeout: int | float | None = None,
        **_ignored: object,
    ) -> CommandResult:
        """执行命令；非零退出抛 ``CommandExitException``，超时抛 ``TimeoutException``。"""
        self.calls.append(command)
        workdir = Path(cwd) if cwd is not None else self._root
        workdir.mkdir(parents=True, exist_ok=True)
        effective = None if timeout is None or timeout <= 0 else timeout
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=workdir,
                capture_output=True,
                timeout=effective,
                env={**os.environ, **(envs or {}), "GIT_PAGER": "cat"},
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutException(f"命令超时（>{effective}s）：{command}") from exc

        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            # 与真实 SDK 一致：非零退出走异常，而非返回带码的结果。
            raise CommandExitException(
                stderr=stderr, stdout=stdout, exit_code=proc.returncode, error=None
            )
        return CommandResult(stderr=stderr, stdout=stdout, exit_code=0, error=None)


class _Files:
    """``sandbox.files`` 的替身：直接落到本地文件系统（路径按宿主机绝对路径处理）。

    替身在宿主机上写文件，因此比真实 e2b 多一道**根目录护栏**：解析后越出 ``root``
    的路径一律拒绝，避免测试里的手滑路径写到临时目录之外。（真实 e2b 不需要这道
    护栏——沙箱本身就是边界。）
    """

    def __init__(self, root: Path, write_files_error: Exception | None = None) -> None:
        self._root = root.resolve()
        # 非 None 时让批量写入整体失败，用以触发后端「降级为逐文件重试」的分支。
        self._write_files_error = write_files_error

    def _local(self, path: str) -> Path:
        candidate = Path(path).resolve()
        if candidate != self._root and self._root not in candidate.parents:
            raise ValueError(f"路径越出替身沙箱根：{path}")
        return candidate

    def write_files(self, files: list[dict[str, object]]) -> None:
        if self._write_files_error is not None:
            raise self._write_files_error
        for entry in files:
            self.write(str(entry["path"]), entry["data"])  # type: ignore[arg-type]

    def write(self, path: str, content: bytes | str) -> None:
        target = self._local(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content if isinstance(content, bytes) else content.encode())

    def read(self, path: str, *, format: str = "text", **_ignored: object) -> bytes | str:
        # 不存在的文件抛 FileNotFoundError、目录抛 IsADirectoryError——
        # 二者的消息能分别映射到 file_not_found / is_directory 字面量。
        data = self._local(path).read_bytes()
        return data if format == "bytes" else data.decode("utf-8", errors="replace")


class SandboxDouble:
    """``e2b.Sandbox`` 的最小替身，只实现 ``E2BSandboxBackend`` 实际用到的表面。"""

    def __init__(
        self,
        root: str | Path,
        *,
        write_files_error: Exception | None = None,
    ) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self.sandbox_id = "double-000000000000"
        self.commands = _Commands(self._root)
        self.files = _Files(self._root, write_files_error)
        self.kill_calls = 0

    def kill(self) -> None:
        self.kill_calls += 1
