<p align="center">
  <img src="assets/logo-reimu-piggy-bank-v2.png" alt="Stock Robot Logo" width="192">
</p>

<h1 align="center">Stock Robot</h1>

<p align="center"><strong>股票、指数与黄金市场分析助手</strong></p>
<p align="center">Web · CLI · HTTP API · MCP · AstrBot</p>

<p align="center">
  <a href="#快速开始">快速开始</a> ·
  <a href="docs/使用指南.md">使用指南</a> ·
  <a href="docs/运行环境.md">运行环境</a> ·
  <a href="https://github.com/Arrhenius401/astrbot_plugin_stock_robot">AstrBot 插件</a>
</p>

数据采集与指标计算由代码完成。个股与指数报告可由 LLM 生成自然语言解读，黄金简报使用确定性数据与规则说明。

## 它能做什么

- **A 股 / 港股个股研报**：财务、技术、估值、行业、舆情与综合评分。
- **指数与 ETF 分析**：宽基、行业、海外及红利低波、自由现金流等策略指数。
- **黄金市场观察简报**：优先展示国内 Au99.99 与国际 XAU 价格、行情日期和涨跌，辅助呈现技术状态、宏观及需求数据；末尾汇总数据来源、观察日期和可用状态。
- **对话式投研与报告库**：用自然语言提问，浏览和下载历史报告。
- **配置雷达与订阅推送**：研究 ETF 标的池、比较历史表现，按需配置推送。
- **多端接入**：Web、CLI、HTTP API；MCP 见[使用指南](docs/使用指南.md#mcp-集成)，AstrBot 见[插件仓库](https://github.com/Arrhenius401/astrbot_plugin_stock_robot)。

## 快速开始

推荐使用 Web 界面。需要 Python 3.11 及以上，并在仓库根目录执行以下命令。

### 1. 安装

Windows PowerShell：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

Linux：

```sh
python3 -m venv .venv
./.venv/bin/python -m pip install -e .
```

普通用户安装核心依赖即可。需要运行开发或测试工具时使用 `".[dev]"`；需要知识库（RAG）时使用 `".[rag]"`。完整说明见[运行环境](docs/运行环境.md)。

### 2. 配置模型

```powershell
.\scripts\stock-robot.ps1 config set data.disclaimer_accepted true
.\scripts\stock-robot.ps1 config set llm.api_key "你的 API Key"
```

默认使用 OpenAI。使用 Claude、兼容接口或暂时不调用 LLM，见[配置 LLM](docs/使用指南.md#配置-llm)。Linux 将 `.\scripts\stock-robot.ps1` 替换为 `./scripts/stock-robot.sh`。

个股与指数的额外分析默认关闭。在 Web「配置 → 分析与数据 → 分析模式」开启，或运行 `.\scripts\stock-robot.ps1 config set analysis.extra_enabled true`：为个股和指数增加规则判断及 AI 解读；个股另提供评分与模型信号。黄金沿用原有设置。

### 3. 启动并生成第一份报告

```powershell
.\scripts\stock-robot.ps1 run
```

启动后打开 <http://127.0.0.1:25618>，输入 `600519` 并提交，即可生成第一份报告。端口可用 `run --port` 指定。

服务默认对外监听 `0.0.0.0`，外部访问需要 HTTP Basic Auth：在项目当前位置 `.stock_robot/config.yaml` 的 `api.auth` 设置 `username`/`password`，或使用环境变量 `STOCK_ROBOT_AUTH_USERNAME` / `STOCK_ROBOT_AUTH_PASSWORD`。本机 `127.0.0.1`/`::1` 请求（如 AstrBot 插件调用）免验证。

不想配置模型时，也可以先用 CLI 查看数据与指标：

```powershell
.\scripts\stock-robot.ps1 analyze 000001 --no-llm
```

## 常用操作

以下命令使用项目启动器；如果已将 `.venv/Scripts`（Linux 为 `.venv/bin`）加入 PATH，也可以直接使用 `stock-robot`。

```powershell
.\scripts\stock-robot.ps1 help
.\scripts\stock-robot.ps1 analyze 000001          # 个股研报
.\scripts\stock-robot.ps1 index 000300 000905    # 指数对比
.\scripts\stock-robot.ps1 index 515300           # ETF 会解析为跟踪指数
.\scripts\stock-robot.ps1 gold --no-llm           # 黄金市场观察简报，无需配置模型
.\scripts\stock-robot.ps1 gold --refresh -o markdown --output-dir reports  # 刷新并归档
.\scripts\stock-robot.ps1 chat                   # 交互式投研
```

股票报告保存到 `reports/stock/`，指数报告保存到 `reports/index/`，黄金报告保存到 `reports/gold/GOLD/`，回测产物保存到 `reports/backtests/`。配置与缓存位于当前目录的 `./.stock_robot/`。新个股与黄金报告同时保存 Markdown 和 `.report.json` 生成时快照，指数报告保存 Markdown；报告库可浏览和下载。`stock-robot run` 会按固定 7 天保留期自行清理个股、指数和黄金报告，可在 Web 配置页关闭；回测、雷达产物不受影响。完整命令与参数见[使用指南](docs/使用指南.md)。

Web 或聊天中输入“黄金”“国内黄金”“国际黄金”“Au99.99”或“XAU”可生成同一份黄金市场观察简报。黄金报告使用确定性说明，不提供买卖建议；来源缺失和过期会单独说明，历史价格分位只表示位置，不是估值或胜率。黄金不提供首饰金店牌价，也不分析单只黄金 ETF。

## 数据与使用边界

- 个股分析支持 A 股和港股；港股可输入 `00700`、`00700.HK` 或 `hk00700`。
- 输入已收录的 ETF 代码时会解析为跟踪指数；ETF 自身表现与指数表现分别展示。
- 公开数据缺失、历史不足或数据源失败时，报告会说明覆盖范围和缺失原因；公开官方 PB 日频历史尚未接入，PB 历史分位可能不可用。
- 指数价格收益不含分红再投资；历史表现不代表未来收益。

详细来源、估值口径和策略限制见[使用指南](docs/使用指南.md)。

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [运行环境](docs/运行环境.md) | Windows/Linux 安装、启动器与虚拟环境 |
| [使用指南](docs/使用指南.md) | CLI、配置、Web/API、RAG、MCP、回测及常见问题 |
| [开发与维护](docs/开发与维护.md) | 模型输出预算、RAG、开发验证与港股行业数据 |
| [升级与版本维护](docs/升级与版本维护.md) | 版本、依赖锁定与升级 |
| [配置雷达自动采集部署](docs/配置雷达自动采集部署.md) | 数据采集与部署 |
| [AGENTS.md](AGENTS.md) | 开发规范与验证要求 |

## 免责声明

本工具仅用于个人学习与研究。数据和观点不构成投资建议，数据准确性与时效性无法保证；使用者需自行判断并承担投资风险。
