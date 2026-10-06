"""运行产物的路径解析（谁都不许在 import 期碰文件系统）。

约定（见 doc/可观测性.md）：运行目录、账本、价格表都落在**用户级** ``~/.nightwatch/``，
而不是目标仓库里——它们是机器本地的噪声，不该出现在被维护的仓库中，也不该被提交。
用 ``NW_HOME`` 环境变量或显式参数改根目录，测试与 CI 靠这个不碰真实用户目录。

**为什么路径解析与目录创建要分开**：``nw_home()`` 只做纯计算。若在解析时顺手
``mkdir``，那么「import 模块」或「跑一条只读命令」都会在用户磁盘上留下目录——副作用
应当只发生在真正要写文件的那一刻（见 :func:`ensure_parent`）。
"""

from __future__ import annotations

import os
from pathlib import Path

# 改根目录的环境变量名。显式参数优先于它，它优先于默认的 ~/.nightwatch。
ENV_HOME = "NW_HOME"
DEFAULT_DIRNAME = ".nightwatch"

LEDGER_FILENAME = "ledger.jsonl"
PRICES_FILENAME = "prices.json"
RUNS_DIRNAME = "runs"
SNAPSHOTS_DIRNAME = "snapshots"
SUMMARY_FILENAME = "summary.json"


def nw_home(explicit: Path | None = None) -> Path:
    """解析运行产物根目录。优先级：显式参数 > ``NW_HOME`` > ``~/.nightwatch``。

    ``NW_HOME`` 为空串时按未设置处理（用 ``or`` 而非 ``is not None``）——空串是
    「变量存在但没有值」的常见形态，把它当成「根目录 = 当前目录」会很意外。

    **不做 realpath 归一**：``.resolve()`` 在 macOS 上会把 ``/var`` 变成 ``/private/var``，
    让「我传的路径」与「回读的路径」字符串不等，测试里徒增困惑；这里也没有必须归一的
    理由（唯一消费者是同样的 ``Path`` 拼接）。
    """
    if explicit is not None:
        return explicit
    return Path(os.environ.get(ENV_HOME) or (Path.home() / DEFAULT_DIRNAME))


def runs_dir(home: Path) -> Path:
    """``<home>/runs``——每次运行一个子目录。"""
    return home / RUNS_DIRNAME


def run_dir(home: Path, run_id: str) -> Path:
    """某次运行的产物目录 ``<home>/runs/<run_id>``。"""
    return runs_dir(home) / run_id


def ledger_path(home: Path) -> Path:
    """账本文件 ``<home>/ledger.jsonl``。"""
    return home / LEDGER_FILENAME


def prices_path(home: Path) -> Path:
    """价格表 ``<home>/prices.json``（默认位置；``--prices`` 可指到别处）。"""
    return home / PRICES_FILENAME


def ensure_parent(path: Path) -> None:
    """确保 ``path`` 的父目录存在——**写入前**才调，不要在解析路径时调。"""
    path.parent.mkdir(parents=True, exist_ok=True)
