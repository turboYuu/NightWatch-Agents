# NightWatch-Agents

一个本地运行的 CLI Agent，接收 GitHub Issue 编号或自然语言任务描述，自动完成从“理解需求 → 定位代码 → 生成修复 → 沙箱验证 → 创建 PR”的全流程。

## 开发环境

依赖采用声明与锁定分离：`pyproject.toml` 声明，`conda-lock.yml` 锁定。安装步骤、新增依赖的流程及已知事项见 [doc/依赖管理.md](doc/依赖管理.md)。

```bash
conda run -n nightwatch_agents pytest          # 跑测试
conda run -n nightwatch_agents ruff check .    # lint
conda run -n nightwatch_agents mypy            # 类型检查
conda run -n nightwatch_agents nw-agent --help # CLI 入口
```
