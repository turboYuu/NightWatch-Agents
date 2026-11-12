# NightWatch-Agents

一个本地运行的 CLI Agent，接收 GitHub Issue 编号或自然语言任务描述，自动完成从“理解需求 → 定位代码 → 生成修复 → 沙箱验证 → 创建 PR”的全流程。

## 快速开始（clone 后在本地运行）

### 前置要求

- 已安装 [uv](https://docs.astral.sh/uv/)（macOS / Linux）：

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # 或 brew install uv
  uv --version
  ```

  不需要预装 Python：`.python-version` 固定了 3.14，`uv sync` 会自动下载对应解释器。

- 一个 [E2B](https://e2b.dev/) 账号的 API Key（真实跑任务需要创建远端沙箱）。
- 一个模型 provider 的 API Key。默认示例使用 DeepSeek，即 `DEEPSEEK_API_KEY`；
  换其他 provider 时，用 `uv add` 加入对应的 LangChain provider 包，并设置其 Key。

### 1. 克隆仓库

```bash
git clone https://github.com/turboYuu/NightWatch-Agents.git
cd NightWatch-Agents
```

### 2. 安装依赖

```bash
uv sync
```

该命令会在仓库根目录创建 `.venv`，并一次性装好运行时依赖、`dev` 组（pytest / ruff / mypy）
以及项目自身（可编辑模式），无需再执行 `pip install -e .`。

### 3. 配置凭据

复制模板并填入真实密钥（`.env` 已在 `.gitignore` 中，不会被提交）：

```bash
cp example.env .env
```

```dotenv
# 建远端沙箱用；--dry-run 不需要
E2B_API_KEY=你的 E2B Key

# 模型：provider:model 格式
NW_MODEL_MAIN=deepseek:deepseek-v4-flash
NW_MODEL_FIX=deepseek:deepseek-v4-pro

# 对应 provider 的 Key，由 init_chat_model 自己读取
DEEPSEEK_API_KEY=你的 DeepSeek Key
```

也可以直接使用同名环境变量而不建 `.env`。

### 4. 验证安装

```bash
uv run nw-agent --help
uv run pytest            # 默认只跑离线用例；真链路用例在无 Key 时自动跳过
```

> 注意计费：如果 shell 里已导出 `E2B_API_KEY` / `NW_MODEL_MAIN` 等变量，真链路用例会
> 实际创建 E2B 沙箱并调用模型，产生费用。只想跑离线用例时，先确认这些变量未设置。

### 5. 运行

先用 `--dry-run` 预演：只校验配置与将要上传的文件，**不建沙箱、不调模型、不花钱**。
示例用一个很小的本地仓库——本仓库自身文件较多，真跑的上传耗时不可控：

```bash
mkdir -p /tmp/demo-repo/tests
printf 'def add(a, b):\n    return a + b\n' > /tmp/demo-repo/calc.py
printf 'from calc import add\n\ndef test_add():\n    assert add(1, 2) == 3\n' > /tmp/demo-repo/tests/test_calc.py

uv run nw-agent --repo /tmp/demo-repo --task "给 calc.py 的 add() 补一个中文 docstring" --dry-run
```

确认无误后去掉 `--dry-run` 真跑：

```bash
uv run nw-agent --repo /tmp/demo-repo --task "给 calc.py 的 add() 补一个中文 docstring"
```

跑完会在终端打印宿主侧导出的补丁。

> 当前限制：`--repo` 只支持本地目录，`--issue`（GitHub 集成）尚未实现；执行路径与工具权限
> 白名单也仍在开发中，请不要把本工具指向不可信仓库。详见 [doc/开发路线图.md](doc/开发路线图.md)。

## 开发环境

依赖采用声明与锁定分离：`pyproject.toml` 声明，`uv.lock` 锁定；开发用 Python 版本由 `.python-version` 固定。安装步骤、增删依赖的流程及已知事项见 [doc/依赖管理.md](doc/依赖管理.md)。

```bash
uv sync                        # 安装 / 对齐环境（新 clone 后执行）
uv run pytest                  # 跑测试
uv run ruff check .            # lint
uv run mypy                    # 类型检查
uv run nw-agent --help         # CLI 入口
```
