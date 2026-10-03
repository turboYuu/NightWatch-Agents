"""交给 Agent 的沙箱工具（全部在沙箱内执行）。

职责（见 doc/NightWatch产品说明.md 3.5）：
- 定义 read_file / write_file / execute_shell / ripgrep_search 等工具
- 路径与权限在**服务端**强制校验：路径不作为模型可传参数
  （例如 repo_path 用闭包绑定，write_file 校验目标属于 target_files）

边界：持有凭据的 ``fetch_issue`` / ``create_github_pr`` 属于「宿主机编排动作」，
不在本包，由 ``nw_agent.graph`` 的节点直接调用。

P0 尚未实现，仅占位以固定目录结构。
"""
