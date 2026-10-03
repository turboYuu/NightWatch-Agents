"""支持通过 `python -m nw_agent` 运行，等价于安装后的 `nw-agent` 命令。

这样在未执行 `pip install -e .` 的开发期也能直接跑 CLI：
    PYTHONPATH=src python -m nw_agent --help
"""

from nw_agent.cli.main import main

if __name__ == "__main__":
    # 用 main() 的返回值作为进程退出码，便于 CI 判断成功/失败。
    raise SystemExit(main())
