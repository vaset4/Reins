# Reins

Reins 是面向 Windows 的本地优先 AI Agent 运行框架，提供终端聊天界面、工具执行、审批、会话恢复、记忆与技能管理，以及模型请求记录。运行数据保存在本机；使用远程模型时，请求内容仍会发送到你配置的模型服务。

当前版本为 `0.1.0`，使用 Python 3.11 或更新版本。正式交互入口为基于 Textual 的 `reins-tui`。

## 安装

在 Windows PowerShell 中运行（以下命令以已安装 Git 和 Python 3.11 为例）：

```powershell
git clone https://github.com/vaset4/Reins.git
cd Reins
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[tui]"
$ReinsPython = (Resolve-Path .\.venv\Scripts\python.exe).Path
$ReinsTui = (Resolve-Path .\.venv\Scripts\reins-tui.exe).Path
```

后续命令在同一个 PowerShell 窗口中使用这两个变量，因此切换到工作目录后也能启动已安装的程序。

## 配置模型

先保存模型服务的 API 密钥，按提示输入；输入内容不会显示在屏幕上：

```powershell
& $ReinsPython -m app.cli secret set my_model_key
```

密钥保存在用户目录的 `.reins/.env` 中，并由 Windows 文件访问权限检查保护。不要把密钥写进源码或提交到仓库。

模型配置使用两个文件，均位于 Windows 用户目录的 `.reins` 文件夹中，与启动时的工作目录无关：

- `%USERPROFILE%\.reins\models.json`：供应商、模型、协议和凭据名称
- `%USERPROFILE%\.reins\.env`：实际 API 密钥，由上面的 `secret set` 命令保存

用以下命令打开模型配置文件；已有配置时请保留其他供应商和模型：

```powershell
notepad "$env:USERPROFILE\.reins\models.json"
```

新建文件时可复制 [models.example.json](models.example.json)，也可使用以下 JSON，把 `YOUR_MODEL_ID` 替换为账户可用的实际模型名称，以 UTF-8 保存：

```json
{
  "active_provider": "openai",
  "active_model": "main",
  "providers": {
    "openai": {
      "model_provider": "openai",
      "base_url": "https://api.openai.com/v1",
      "credential": "my_model_key",
      "api_mode": "chat_completions",
      "models": {
        "main": {
          "model": "YOUR_MODEL_ID",
          "context_window": 30000,
          "max_output_tokens": 4096
        }
      }
    }
  }
}
```

`credential` 的值必须与 `secret set my_model_key` 中的名称一致，它不是密钥本身。`active_provider` 指向 `providers` 下的供应商名称，`active_model` 指向该供应商 `models` 下的模型别名。

这是 OpenAI Chat Completions 协议的示例。其他服务应填写对应的供应商、接口地址、模型 ID 和 `api_mode`；支持的协议值为 `chat_completions`、`responses`、`anthropic_messages`。示例额度不是模型的官方上限，请按所选模型实际支持的额度设置，输出额度不能超过窗口。

模型文件使用 `models.json`，不再读取旧的 `models.yaml`、项目中的 `llm.json` 或 `config.yaml` 模型配置。首次启动前先完成配置；F6 可以选择已配置的模型，不能创建供应商或录入密钥。

## 启动与使用

先进入希望 Agent 操作的工作目录，再启动界面：

```powershell
cd C:\work\my-project
& $ReinsTui
```

请将示例目录替换为实际工作目录。首次使用建议选择有备份的测试项目，提交任务前确认工作区和授权范围；工具可能修改文件或执行命令。

- `F1`：查看帮助
- `F6` 或 `/model`：选择已配置的模型及其支持的思考强度，应用到后续新运行
- `F4`：查看审批
- `F9`：查看模型请求记录
- `Ctrl+G`：停止当前运行
- `Ctrl+Q`：断开界面，后台继续运行

运行数据默认保存在用户目录的 `.reins/data`。可用独立目录隔离一次试用：

```powershell
& $ReinsTui --data-root C:\work\reins-data
```

关闭界面不等于关闭后台。查询或明确停止后台：

```powershell
& $ReinsPython -m app.cli background status
& $ReinsPython -m app.cli background stop
```

使用了自定义数据目录时，查询和停止也必须指定同一目录：

```powershell
& $ReinsPython -m app.cli background status --data-root C:\work\reins-data
& $ReinsPython -m app.cli background stop --data-root C:\work\reins-data
```

停止后台会保留运行记录，请以命令输出判断是否已退出。命令行的其他入口与参数可通过 `& $ReinsPython -m app.cli --help` 查看。

## 开发与验证

回到仓库目录安装开发依赖，并对受影响的测试分组验证：

```powershell
& $ReinsPython -m pip install -e ".[tui,dev]"
& $ReinsPython -m ruff check .
& $ReinsPython scripts/run_with_timeout.py --seconds 60 -- $ReinsPython -m pytest tests/test_startup_identity.py tests/test_model_profiles.py -q
```

每组后端测试限时 60 秒，避免未结束的测试持续占用进程。上面的命令是开发检查示例，不代表全量测试、真实模型服务和所有可选工具均已通过验证。浏览器、视觉、系统和 MCP 工具分别使用 `browser`、`vision`、`os`、`mcp` 可选依赖，部分功能还需要相应外部程序或服务。

## 许可证

采用 [MIT 许可证](LICENSE)。
