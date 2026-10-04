"""E2B 云端沙箱后端（默认生产后端）。

本后端是 ``SandboxBackend`` 的**唯一实现**。

⚠️ **代码外发风险**：本后端会把目标仓库的必要文件子集**上传到第三方云端**执行
（见 doc/NightWatch产品说明.md 3.3）。私有仓库、含密钥的仓库默认不建议使用；
无 Key 场景请改用 ``--dry-run``（不建沙箱、不调远端）。

**为什么直接用裸 ``e2b`` SDK 而不是复用 ``langchain-e2b``**：
    后者是 0.0.x 且不带 ``py.typed``（mypy strict 下不可用），而它提供的价值
    只有「命令执行 + 文件读写」两件事。``e2b`` 本身就自带类型标记，且
    ``commands.run`` 原生支持 ``cwd`` / ``envs``——比 ``execute(command)`` 那种
    只能拼命令字符串的接口更合用。自己写这几十行换来全程可静态检查。
"""

from __future__ import annotations

import logging
import shlex
from typing import TYPE_CHECKING

import e2b
from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)

from nw_agent.backends.errors import (
    ERR_FILE_NOT_FOUND,
    ERR_INVALID_PATH,
    ERR_IS_DIRECTORY,
    ERR_PERMISSION_DENIED,
    TIMEOUT_EXIT_CODE,
)
from nw_agent.backends.interface import (
    SandboxBackend,
    SandboxConfig,
    cap_output,
    combine_output,
    join_sandbox_path,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


def _normalize_file_error(exc: Exception) -> str:
    """把 e2b 的文件类异常映射成 deepagents 的错误字面量。

    e2b 没有为「路径是目录」「无权限」等情形提供稳定的异常子类，只能按消息
    关键字宽松映射。映射不出来时原样返回异常信息——上层至少能看到原因，
    而不是一个语焉不详的 ``unknown``。

    **判定顺序要紧**：POSIX 的「文件不存在」消息是 ``No such file or directory``，
    同时含 ``not found`` 与 ``directory`` 两族关键字，故 ``file_not_found`` 必须先判，
    否则「不存在」会被误报成「是个目录」。
    """
    message = str(exc).lower()
    if "not found" in message or "no such file" in message:
        return ERR_FILE_NOT_FOUND
    if "directory" in message:
        return ERR_IS_DIRECTORY
    if "permission" in message:
        return ERR_PERMISSION_DENIED
    if "invalid" in message or "absolute" in message:
        return ERR_INVALID_PATH
    return str(exc)


class E2BSandboxBackend(SandboxBackend):
    """E2B 云端一次性沙箱。

    生命周期：:meth:`create` 创建远端沙箱，:meth:`kill`（或 ``with``）销毁。
    E2B 侧 ``on_timeout`` 默认为 ``"kill"``，即超时自动回收——这与文档 3.3
    「不跨人工门控持有沙箱」一致，我们**不**改成 pause。

    销毁是幂等的，且 kill 失败不重试：超时回收是兜底，重试只会增加网络噪音。
    """

    def __init__(self, config: SandboxConfig, *, sandbox: e2b.Sandbox) -> None:
        """由 :meth:`create` 调用；不要直接构造。

        Args:
            config: 创建参数。
            sandbox: 一个已 ``e2b.Sandbox.create(...)`` 好的实例。
        """
        super().__init__(config)
        self._sandbox = sandbox

    @classmethod
    def create(
        cls,
        config: SandboxConfig | None = None,
        *,
        api_key: str | None = None,
    ) -> E2BSandboxBackend:
        """创建远端沙箱并包装为后端。

        Args:
            config: 创建参数；None 时用默认值。
            api_key: 显式 API Key；None 时由 SDK 读环境变量 ``E2B_API_KEY``。

        Returns:
            已创建的后端。**销毁由调用方负责**——推荐 ``with``。

        Raises:
            RuntimeError: 远端创建失败，或沙箱环境准备（建仓库目录 / 探测 git）失败。
        """
        backend_config = config if config is not None else SandboxConfig()
        try:

            sandbox = e2b.Sandbox.create(
                template=backend_config.template,
                timeout=backend_config.timeout_seconds,
                metadata=dict(backend_config.metadata) or None,
                envs=dict(backend_config.envs) or None,
                api_key=api_key,
            )
        except e2b.SandboxException as exc:
            raise RuntimeError(f"创建 E2B 沙箱失败：{type(exc).__name__}: {exc}") from exc

        return cls._attach(backend_config, sandbox)

    @classmethod
    def _attach(cls, config: SandboxConfig, sandbox: e2b.Sandbox) -> E2BSandboxBackend:
        """把已建好的 ``sandbox`` 包装成后端并完成创建期准备。

        与 :meth:`create` 分开，是为了留出**注入点**：测试可传入 ``e2b.Sandbox`` 的
        替身（见 ``tests/e2b_double.py``）走通同一条「构造 → 环境准备」路径，不必联网。
        """
        backend = cls(config, sandbox=sandbox)
        backend._prepare_workspace()
        return backend

    # ------------------------------------------------------------------ 状态
    @property
    def id(self) -> str:
        """E2B 沙箱 ID。resume 时据此判断句柄是否仍有效（见文档 3.3 对 ``sandbox_id`` 的约定）。"""
        return self._sandbox.sandbox_id

    # ------------------------------------------------------------------ 执行
    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """在沙箱内的仓库根目录下执行命令。

        Args:
            command: 完整 shell 命令字符串。
            timeout: 超时秒数；None 用 ``config.command_timeout_seconds``。
                传 0 表示不限制。

        Returns:
            ``ExecuteResponse``；非零退出码原样透传，超时为
            :data:`~nw_agent.backends.errors.TIMEOUT_EXIT_CODE`，其余远端错误为 1。
        """
        self._require_alive()
        effective = self._config.command_timeout_seconds if timeout is None else timeout
        try:
            result = self._sandbox.commands.run(
                command,
                cwd=self._config.repo_path,
                envs=dict(self._config.envs) or None,
                timeout=None if effective <= 0 else effective,
            )
        except e2b.CommandExitException as exc:
            # 非零退出走的是异常路径：e2b 把输出与退出码挂在异常对象上。
            output, truncated = cap_output(combine_output(exc.stdout, exc.stderr))
            return ExecuteResponse(output=output, exit_code=exc.exit_code, truncated=truncated)
        except e2b.TimeoutException:
            return ExecuteResponse(
                output=f"命令执行超时（>{effective}s）：{command}",
                exit_code=TIMEOUT_EXIT_CODE,
                truncated=False,
            )
        except e2b.SandboxException as exc:
            logger.warning("E2B 执行失败：%s", type(exc).__name__, exc_info=True)
            return ExecuteResponse(
                output=f"E2B 执行失败（{type(exc).__name__}）：{exc}",
                exit_code=1,
                truncated=False,
            )
        output, truncated = cap_output(combine_output(result.stdout, result.stderr))
        return ExecuteResponse(output=output, exit_code=result.exit_code, truncated=truncated)

    # ------------------------------------------------------------------ 文件
    def upload_tree(
        self,
        files: Sequence[tuple[str, bytes]],
        *,
        base: str | None = None,
    ) -> list[FileUploadResponse]:
        """批量上传，**部分成功**。

        优先走 ``files.write_files`` 的单次请求批量提交；只有批量失败时才降级为
        逐文件重试——降级是为了拿到 per-file 错误（deepagents 要求逐条报告），
        而正常路径不该为每个文件付一次 RTT。
        """
        self._require_alive()
        results: dict[int, FileUploadResponse] = {}
        prepared: list[tuple[int, str, str, bytes]] = []

        for index, (raw_path, content) in enumerate(files):
            path = join_sandbox_path(raw_path, base)
            if path.startswith("/"):
                prepared.append((index, raw_path, path, content))
            else:
                results[index] = FileUploadResponse(path=raw_path, error=ERR_INVALID_PATH)

        if prepared:
            try:
                self._sandbox.files.write_files(
                    [{"path": path, "data": content} for _, _, path, content in prepared]
                )
            except Exception:  # noqa: BLE001 - 批量失败要降级，不能整体抛
                logger.warning("批量上传失败，降级为逐文件重试", exc_info=True)
                for index, raw_path, path, content in prepared:
                    results[index] = self._write_one(raw_path, path, content)
            else:
                for index, raw_path, _, _ in prepared:
                    results[index] = FileUploadResponse(path=raw_path, error=None)

        return [results[index] for index in range(len(files))]

    def _write_one(self, raw_path: str, path: str, content: bytes) -> FileUploadResponse:
        """逐文件上传，把异常归一化成错误字面量（不抛）。"""
        try:
            self._sandbox.files.write(path, content)
        except Exception as exc:  # noqa: BLE001 - 逐文件降级路径必须吃掉异常
            return FileUploadResponse(path=raw_path, error=_normalize_file_error(exc))
        return FileUploadResponse(path=raw_path, error=None)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """逐文件读取，**部分成功**。"""
        self._require_alive()
        responses: list[FileDownloadResponse] = []
        for path in paths:
            if not path.startswith("/"):
                responses.append(
                    FileDownloadResponse(path=path, content=None, error=ERR_INVALID_PATH)
                )
                continue
            try:
                # e2b 的 read 在 bytes 模式下返回 bytearray，这里统一成 bytes，
                # 与 deepagents 的 FileDownloadResponse.content 契约对齐。
                content = bytes(self._sandbox.files.read(path, format="bytes"))
            except Exception as exc:  # noqa: BLE001 - 逐文件路径必须吃掉异常
                responses.append(
                    FileDownloadResponse(path=path, content=None, error=_normalize_file_error(exc))
                )
            else:
                responses.append(FileDownloadResponse(path=path, content=content, error=None))
        return responses

    # ------------------------------------------------------------------ 销毁
    def kill(self) -> None:
        """销毁远端沙箱。**幂等**：已销毁或远端已不存在都视作成功。"""
        if self._is_killed:
            return
        # 先置位再调用：kill 请求本身失败也不重试——E2B 的 on_timeout="kill"
        # 会兜底回收，重试只会增加网络噪音。
        self._is_killed = True
        try:
            self._sandbox.kill()
        except Exception:  # noqa: BLE001 - 收尾路径不能抛
            logger.warning("E2B kill 失败（沙箱将由 on_timeout 自动回收）", exc_info=True)

    # ------------------------------------------------------------- 内部工具
    def _prepare_workspace(self) -> None:
        """建出仓库目录并探测 git，失败即抛。

        两件事都放在创建阶段：仓库目录不存在的话，之后 ``execute`` 的 ``cwd``
        直接失效；而 git 是基线提交与导出 diff 的前提，留到 ``upload_repo``
        才炸会白费一次代码上传。

        Raises:
            RuntimeError: 镜像里没有 git，或目录创建失败（错误信息含换模板的指引）。
        """
        repo = shlex.quote(self._config.repo_path)
        try:
            result = self._sandbox.commands.run(
                f"mkdir -p {repo} && git --version",
                timeout=self._config.command_timeout_seconds,
            )
        except e2b.CommandExitException as exc:
            raise RuntimeError(
                f"沙箱环境准备失败（mkdir -p {self._config.repo_path} && git --version）："
                f"{combine_output(exc.stdout, exc.stderr)[:500]}\n"
                "请改用预装 git 的模板，或在模板构建阶段安装 git"
                "（见 doc/NightWatch产品说明.md 3.3 的沙箱预热）。"
            ) from exc
        if result.exit_code != 0:
            raise RuntimeError(
                f"沙箱环境准备失败：{combine_output(result.stdout, result.stderr)[:500]}"
            )
