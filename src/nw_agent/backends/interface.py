"""沙箱后端统一接口。

职责（见 doc/NightWatch产品说明.md 3.3）：
- 定义 ``SandboxBackend``——创建 / 上传代码 / 执行命令 / 导出 diff / 销毁
- 生命周期收口：``kill()`` 幂等 + ``__exit__`` 兜底（文档 3.3 第 5 条「终止必清理」）
- ``export_diff()`` 在沙箱内自建 git 基线后取 diff，由本基类实现，各后端共享

**为什么继承 deepagents 的 ``BaseSandbox`` 而不是自带 ``typing.Protocol``**：
    deepagents 判定后端能力用的是**类属性身份比对**，例如
    ``type(self).ls_info is not BackendProtocol.ls_info``。而 ``BackendProtocol``
    是 ``abc.ABC`` 而非 ``typing.Protocol``——鸭子类型无法通过探测，会被误判成
    「实现了旧 API」并走进废弃分支。代价是本模块依赖 deepagents（已声明为运行时依赖）。

    当前唯一实现是 :class:`~nw_agent.backends.e2b_backend.E2BSandboxBackend`。
    接口层保持与实现解耦，是为了让「换隔离方案只改 ``backend=`` 一处」这条承诺成立。

    ⚠️ 因此子类**不要覆写** ``ls_info`` / ``grep_raw`` / ``glob_info``：
    覆写这三个名字等于对 deepagents 宣称「我实现的是旧接口」。
"""

from __future__ import annotations

import logging
import shlex
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from deepagents.backends.protocol import FileUploadResponse
from deepagents.backends.sandbox import BaseSandbox

from nw_agent.backends.errors import SandboxClosedError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 沙箱内固定布局
# ---------------------------------------------------------------------------
# 与产品说明 3.5 节「路径校验在 /workspace/ 内」保持一致：P1 写 tools/ 的
# 白名单校验时直接复用这两个常量，不另立一套口径。
WORKDIR = "/workspace"
REPO_PATH = f"{WORKDIR}/repo"

# git 基线提交所用的身份。沙箱镜像里通常没有全局 git 配置，不显式给会让
# commit 以「Please tell me who you are」失败（退出码 128）。
GIT_AUTHOR_NAME = "nightwatch-agent"
GIT_AUTHOR_EMAIL = "nightwatch-agent@localhost"

# 单条命令输出的硬上限（字节）。deepagents 的 ``ExecuteResponse.truncated``
# 就是为此设计的；我们不开启它的 capture-offload（依赖镜像有 sh/coreutils，
# 默认模板不保证），截断责任落在本层。
MAX_OUTPUT_BYTES = 256 * 1024

# 凭据黑名单：这些字样出现在 ``SandboxConfig.envs`` 的键里一律拒绝。
# 把「凭据只存宿主机、绝不进沙箱」（产品说明 3.3 安全原则）从注释变成运行期约束。
_FORBIDDEN_ENV_MARKERS = ("TOKEN", "SECRET", "KEY", "PASSWORD", "CREDENTIAL")


# ---------------------------------------------------------------------------
# 结果处理工具（后端共用，保证输出形态一致）
# ---------------------------------------------------------------------------
def combine_output(stdout: str, stderr: str) -> str:
    """按 deepagents 的约定把 stdout 与 stderr 合并成单一字符串。

    约定与 ``langchain_e2b`` 的同名实现一致（stderr 非空时换行追加），
    这样上层的测试结果解析不必区分后端。
    """
    if stdout and stderr:
        return f"{stdout}\n{stderr}"
    return stdout or stderr


def cap_output(text: str, limit: int = MAX_OUTPUT_BYTES) -> tuple[str, bool]:
    """截断过长的命令输出。

    Returns:
        ``(截断后的文本, 是否发生截断)``；第二个值直接填进
        ``ExecuteResponse.truncated``。
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text, False
    # 按字节截断可能切断多字节字符，用 errors="ignore" 丢弃残片而非产生乱码。
    return encoded[:limit].decode("utf-8", errors="ignore"), True


def join_sandbox_path(path: str, base: str | None) -> str:
    """把（可能是相对路径的）``path`` 拼到 ``base`` 下。

    ``base`` 为 None 时原样返回，表示调用方给的是沙箱内绝对路径。
    仅用于 upload 系列的路径归一化。
    """
    if base is None:
        return path
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


# ---------------------------------------------------------------------------
# 创建参数
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SandboxConfig:
    """一次沙箱创建所需的全部参数。

    frozen 的理由：它会被写进 ``MaintenanceState`` 与 checkpoint，
    可变对象在这条链路上是隐患。
    """

    template: str | None = None
    """E2B 模板名或 ID；None 表示用后端默认模板。"""

    timeout_seconds: int = 1800
    """沙箱存活时长（秒）。E2B 侧的 ``on_timeout`` 默认为 "kill"，即超时自动回收——
    这与文档 3.3「不跨人工门控持有沙箱」一致，我们不改成 pause。"""

    command_timeout_seconds: int = 300
    """单条命令的默认超时（秒）；``execute(timeout=...)`` 可逐次覆盖。"""

    repo_path: str = REPO_PATH
    """仓库根路径。

    它同时是 :meth:`SandboxBackend.execute` 的工作目录——刻意不另设 workdir：
    两个上下文（沙箱根 vs 仓库根）并存时，上层很容易把命令跑错目录，
    而本项目的所有动作都发生在仓库内。需要跑在别处时在命令里显式 ``cd``。
    """

    metadata: Mapping[str, str] = field(default_factory=dict)
    """给沙箱打的元数据标签（E2B 控制面可见），便于对账与费用归因。"""

    envs: Mapping[str, str] = field(default_factory=dict)
    """显式注入沙箱的环境变量。**绝不放凭据**——构造时会校验。"""

    def __post_init__(self) -> None:
        """拒绝携带疑似凭据的环境变量。

        校验放在这里（而不是各后端自己的 ``__init__``）有两个好处：一是覆盖
        全部后端，不会因为新增后端时忘了写而漏掉；二是让**非法的 config 根本
        构造不出来**，而不是等到创建沙箱时才炸。

        Raises:
            ValueError: ``envs`` 的键里出现疑似凭据的字样。这是安全红线
                （产品说明 3.3：凭据只存宿主机编排层、绝不进沙箱），
                宁可让构造失败，也不静默放行。
        """
        offending = sorted(
            key
            for key in self.envs
            if any(marker in key.upper() for marker in _FORBIDDEN_ENV_MARKERS)
        )
        if offending:
            raise ValueError(
                f"SandboxConfig.envs 含疑似凭据的键 {offending}；"
                "凭据只允许存在于宿主机编排层，绝不注入沙箱。"
            )


# ---------------------------------------------------------------------------
# 接口
# ---------------------------------------------------------------------------
class SandboxBackend(BaseSandbox):
    """沙箱后端统一接口：创建 / 上传代码 / 执行命令 / 导出 diff / 销毁。

    「五动作」与 deepagents 协议的收口方式（见 doc/开发路线图.md 第 0 阶段）：

    ============  ==================================================
    文档的五动作   落地形态
    ============  ==================================================
    创建          模块级工厂 ``create_backend()``，**不是**实例方法——
                  未创建完的实例无法构造，创建失败即构造失败，不留半成品对象
    上传代码      ``upload_tree()``（子类实现）+ ``upload_repo()``（建 git 基线）
    执行命令      继承 deepagents 的 ``execute()``，签名逐字保留
    导出 diff     ``export_diff()``，由本类实现
    销毁          ``kill()``（子类实现，必须幂等）+ ``__exit__`` 兜底
    ============  ==================================================

    子类需要实现：``id`` / ``execute`` / ``upload_tree`` / ``download_files`` / ``kill``。
    其余文件类操作由 ``BaseSandbox`` 从 ``execute`` 与 ``upload_files`` 派生。

    **``execute`` 的工作目录约定**：一律在 ``config.repo_path`` 下执行。刻意不提供
    「沙箱根」与「仓库根」两个上下文——并存只会让上层把命令跑错目录，而本项目的
    所有动作都发生在仓库内；需要跑在别处时在命令里显式 ``cd``。
    """

    def __init__(self, config: SandboxConfig) -> None:
        """记录配置。

        Args:
            config: 创建参数。凭据类环境变量已在 ``SandboxConfig`` 构造时被拒，
                见其 ``__post_init__``。
        """
        self._config = config
        self._is_killed = False

    # ------------------------------------------------------------------ 状态
    @property
    def config(self) -> SandboxConfig:
        """本次创建的配置（只读，供日志与 checkpoint 记录）。"""
        return self._config

    @property
    def is_alive(self) -> bool:
        """沙箱是否仍被本进程视为可用。

        resume 时若 ``sandbox_id`` 非空但本属性为 False，说明句柄已失效，
        应直接丢弃重建（文档 3.3 对 ``sandbox_id`` 的约定）。
        """
        return not self._is_killed

    # ------------------------------------------------------------- 上传代码
    @abstractmethod
    def upload_tree(
        self,
        files: Sequence[tuple[str, bytes]],
        *,
        base: str | None = None,
    ) -> list[FileUploadResponse]:
        """批量上传文件到沙箱（「上传代码」动作）。

        调用方负责**只挑必要文件子集**——绝不传宿主目录、``~/.ssh``、
        ``~/.aws``、``.env``（产品说明 3.3 安全原则）。

        实现必须支持**部分成功**：逐文件捕获异常后填进对应 response，
        **不要整体抛异常**。这是 deepagents 对 ``upload_files`` 的硬要求。

        Args:
            files: ``(路径, 内容)`` 序列。
            base: 非 None 时，``files`` 里的路径按相对路径处理并拼到 ``base`` 下；
                None 时按沙箱内绝对路径处理。

        Returns:
            与 ``files`` 等长同序的响应列表。
        """

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """deepagents 协议的入口，委托给 :meth:`upload_tree`。

        保留它是为了满足 ``BaseSandbox`` 的抽象方法（它的文件类操作全部由
        ``execute`` 与 ``upload_files`` 派生）。子类只需实现 ``upload_tree``，
        不必把同一件事写两遍。
        """
        return self.upload_tree(files)

    def upload_repo(self, files: Sequence[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """把文件上传到 ``config.repo_path`` 下，并建立 git 基线提交。

        顺序不可换：必须**先全部传完再 git init/add/commit**。若先 init，
        随后上传的文件会落在基线之外，``export_diff`` 会把整个仓库当成新增。

        Args:
            files: ``(相对仓库根的路径, 内容)`` 序列。

        Returns:
            上传响应。

        Raises:
            RuntimeError: 有文件上传失败，或 git 基线建立失败。此时沙箱状态
                不可信（基线没建成的 diff 会包含整个仓库），调用方应丢弃重建。
        """
        self._require_alive()
        responses = self.upload_tree(files, base=self._config.repo_path)
        failed = [r for r in responses if r.error]
        if failed:
            raise RuntimeError(
                f"上传失败 {len(failed)}/{len(responses)} 个文件；首个错误：{failed[0].error}"
            )
        self._build_baseline()
        return responses

    # ------------------------------------------------------------- 导出 diff
    def export_diff(self, *, include_untracked: bool = True) -> str:
        """导出相对 git 基线的 unified diff（「导出 diff」动作）。

        宿主与沙箱之间**只交换 diff 与测试输出**（产品说明 3.3 安全原则），
        因此返回值是一段可 JSON 序列化、可进 checkpoint、可给人审的字符串。

        实现要点：

        - 用 ``git diff HEAD`` 而非裸 ``git diff``：裸 diff 只比「工作树 vs 索引」，
          若 agent 执行过 ``git add``，改动已进索引，裸 diff 会**静默返回空**。
        - ``git add -N`` 让新增的未跟踪文件以 ``new file mode`` 出现在补丁里，
          否则会被静默丢弃。注意 ``-N`` 对已被 ``.gitignore`` 忽略的文件无效。
        - ``--no-pager``：execute 返回的是拼接字符串，分页器会让输出挂起。

        Args:
            include_untracked: 是否把新增（未跟踪）文件算进 diff。

        Returns:
            unified diff 文本；**无改动或导出失败一律返回空串**。失败不抛异常：
            本方法处在节点收尾路径上（紧随其后就是 ``kill()``），抛异常会中断
            收尾并泄漏沙箱。
        """
        self._require_alive()
        try:
            if include_untracked:
                # 对已跟踪文件重复 -N 在部分 git 版本会告警，这里刻意静默。
                self.execute("git add -N . 2>/dev/null")
            result = self.execute("git --no-pager diff HEAD")
        except Exception:  # noqa: BLE001 - 收尾路径不能抛
            logger.warning("导出 diff 时出错，返回空补丁", exc_info=True)
            return ""
        if result.exit_code != 0:
            logger.warning("git diff 退出码 %s：%s", result.exit_code, result.output[:500])
            return ""
        if not result.output.strip():
            return ""
        # 规范化结尾换行，保证补丁可直接 git apply。
        return result.output.rstrip("\n") + "\n"

    # ---------------------------------------------------------------- 销毁
    @abstractmethod
    def kill(self) -> None:
        """销毁沙箱（「销毁」动作）。**必须幂等**。

        幂等是硬要求而非优化：文档 3.3 要求「终止必清理」且异常路径用 finally
        兜底，于是「节点正常结束 kill 一次 + finally 再兜一次」是常态。
        第二次调用不得抛异常；远端/目录已不存在也应视作成功。
        """

    # ------------------------------------------------------- 上下文管理器
    def __enter__(self) -> SandboxBackend:
        """进入上下文；返回自身，便于 ``with create_backend(...) as sb:``。"""
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> Literal[False]:
        """退出上下文时无条件销毁，且吞掉清理期的异常。

        返回 False 而非 True：清理失败**不得掩盖**原始业务异常，否则
        ``run_agent`` 段的真实错误会被一次 kill 超时吃掉。

        Returns:
            恒为 False，即不压制原异常。
        """
        try:
            self.kill()
        except Exception:  # noqa: BLE001 - 清理期必须兜住一切
            logger.warning("销毁沙箱时出错，已忽略以保留原始异常", exc_info=True)
        return False

    # ------------------------------------------------------------- 内部工具
    def _require_alive(self) -> None:
        """动作前断言沙箱可用。

        Raises:
            SandboxClosedError: 已销毁后再调用动作。上层误用应当显式失败，
                而不是拿到一个语焉不详的远端错误。
        """
        if self._is_killed:
            raise SandboxClosedError(f"沙箱 {self.id} 已销毁，不能再执行操作")

    def _build_baseline(self) -> None:
        """在 ``config.repo_path`` 建立 git 基线提交。

        命令经 :meth:`execute` 在仓库根下执行（``execute`` 的 cwd 即
        ``config.repo_path``），且必须显式配置 git 身份与关闭签名：
        沙箱镜像里通常没有全局 git 配置，也没有 GPG key。

        Raises:
            RuntimeError: 任一条 git 命令失败。基线没建成的沙箱导出的 diff
                会把整个仓库当成新增文件，必须放弃而非继续。

        TODO(P1): 若目标模板缺 git，可降级为「整目录快照 + diff -ruN」。
            P0 不做的理由：那会引入第二套 diff 语义，需与 git 路径分别测试。
        """
        steps = (
            "git init -q",
            f"git config user.email {shlex.quote(GIT_AUTHOR_EMAIL)}",
            f"git config user.name {shlex.quote(GIT_AUTHOR_NAME)}",
            # 模板可能继承了 gpgsign 配置却没有 GPG key，会让 commit 失败。
            "git config commit.gpgsign false",
            "git add -A",
            # --allow-empty：文件集为空时也要有基线，否则 diff HEAD 无从比较。
            'git commit -q --no-gpg-sign --allow-empty -m "baseline: uploaded snapshot"',
        )
        for step in steps:
            result = self.execute(step)
            if result.exit_code != 0:
                raise RuntimeError(f"建立 git 基线失败（{step}）：{result.output[:500]}")
