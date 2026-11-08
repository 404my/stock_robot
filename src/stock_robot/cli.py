"""Stock Robot CLI — AI 驱动的股票分析研报助手"""
import logging
import os
from copy import deepcopy
from datetime import date, datetime, timedelta

os.environ["TQDM_DISABLE"] = "1"

import sys
from pathlib import Path
from typing import IO, Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)

from stock_robot.metadata import APP_NAME, VERSION
from utils.config import DEFAULT_CONFIG, Config
from utils.paths import resource_root, runtime_root

_RUNTIME_LOG_HANDLER_NAME = "stock-robot-runtime-log"


def _configure_runtime_log(config: Config) -> Path:
    """为 Web 服务增加可供界面读取的本地运行日志。"""
    root_logger = logging.getLogger()
    for existing_handler in root_logger.handlers:
        if existing_handler.get_name() != _RUNTIME_LOG_HANDLER_NAME:
            continue
        existing_path = getattr(existing_handler, "baseFilename", None)
        if isinstance(existing_path, str):
            return Path(existing_path)
    # 同机多个 Web 服务可并存，按进程隔离文件，避免日志页混入其他服务的记录。
    log_path = config.config_dir / f"runtime-{os.getpid()}.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.set_name(_RUNTIME_LOG_HANDLER_NAME)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"))
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    return log_path


def _create_cli_progress():
    """创建三条 CLI 命令共用的进度条。"""
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
        transient=True,
    )


def _get_registry():
    """构建默认注册表"""
    from analysis.financial import FinancialAnalyzer
    from analysis.industry import IndustryAnalyzer
    from analysis.sentiment import SentimentAnalyzer
    from analysis.technical import TechnicalAnalyzer
    from analysis.valuation import ValuationAnalyzer
    from core.registry import Registry
    from data.akshare import AkShareAdapter
    from data.hk_hsics_peers import HKIndustryAdapter

    reg = Registry()
    reg.register_data_source(AkShareAdapter())
    reg.register_data_source(HKIndustryAdapter())
    reg.register_analysis_module(FinancialAnalyzer())
    reg.register_analysis_module(TechnicalAnalyzer())
    reg.register_analysis_module(ValuationAnalyzer())
    reg.register_analysis_module(IndustryAnalyzer())
    reg.register_analysis_module(SentimentAnalyzer())
    return reg


def _register_llm(reg, config):
    """注册 LLM 后端（复用 bootstrap 构建逻辑，含重试/超时参数）"""
    from api.bootstrap import build_llm

    llm = build_llm(config)
    if llm is not None:
        reg.register_llm_backend(llm, provider=config.get("llm.provider", "openai"))


def _build_pipeline(llm_enabled=True):
    """构建管道"""
    from core.pipeline import Pipeline
    from utils.config import Config

    config = Config()
    reg = _get_registry()
    _register_llm(reg, config)
    return Pipeline(registry=reg, config=config, llm_enabled=llm_enabled)


def _get_cache():
    """获取缓存管理器"""
    from data.cache import CacheManager
    from utils.config import Config
    config = Config()
    return CacheManager(db_path=config.config_dir / "cache.db")


def _check_disclaimer(config):
    """检查免责声明是否已接受"""
    if not config.get("data.disclaimer_accepted", False):
        console.print(Panel.fit(
            "[bold yellow]免责声明[/bold yellow]\n\n"
            "本工具仅用于个人学习和研究目的。所有分析数据和观点不构成任何投资建议。\n"
            "股票投资有风险，入市需谨慎。数据来源的准确性和时效性无法保证，使用者需自行判断。\n\n"
            '输入 [bold]stock-robot config set data.disclaimer_accepted true[/bold] 确认已阅读。',
            title="首次使用"
        ))
        return False
    return True


def _convert_value(value: str):
    """将字符串转换为合适类型"""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _help_hint(ctx: click.Context) -> str:
    """根据实际命令上下文生成终端帮助入口。"""
    names = []
    while ctx.parent is not None:
        names.append(ctx.info_name or ctx.command.name or "")
        ctx = ctx.parent
    path = " ".join(reversed(names))
    return f"运行 stock-robot help{' ' + path if path else ''} 查看详细用法。"


class _HelpHintError(click.ClickException):
    def __init__(self, original: click.ClickException, ctx: click.Context):
        super().__init__(original.message)
        self.original, self.ctx = original, ctx

    def show(self, file: IO[str] | None = None) -> None:
        self.original.show(file)
        click.echo(_help_hint(self.ctx), file=file, err=file is None)


class _HelpHintCommand(click.Command):
    def get_usage(self, ctx: click.Context) -> str:
        return f"{super().get_usage(ctx)}\n\n{_help_hint(ctx)}"

    def invoke(self, ctx: click.Context) -> Any:
        try:
            return super().invoke(ctx)
        except (SystemExit, click.exceptions.Exit) as exc:
            code = exc.exit_code if isinstance(exc, click.exceptions.Exit) else exc.code
            if code not in (None, 0):
                click.echo(_help_hint(ctx), err=True)
            raise
        except click.ClickException as exc:
            if isinstance(exc, click.UsageError):
                raise
            # Click 的 message 与退出码只读，派生类保留原退出码而不改原异常。
            error_type = type("_HelpHintError", (_HelpHintError,), {"exit_code": exc.exit_code})
            raise error_type(exc, ctx) from exc


class _HelpHintGroup(click.Group):
    command_class = _HelpHintCommand
    # Click 使用 type 表示嵌套组沿用当前组类型。
    group_class = type

    def get_usage(self, ctx: click.Context) -> str:
        return f"{super().get_usage(ctx)}\n\n{_help_hint(ctx)}"


@click.group(cls=_HelpHintGroup, invoke_without_command=True, no_args_is_help=False,
             help=f"{APP_NAME} — AI 驱动的股票分析研报助手",
             epilog="运行 stock-robot help <命令> 查看详细用法。")
@click.version_option(version=VERSION)
@click.pass_context
def main(ctx: click.Context):
    """Stock Robot — AI 驱动的股票分析研报助手"""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command("help")
@click.argument("command_path", nargs=-1, metavar="[命令路径]...")
@click.pass_context
def help_command(ctx: click.Context, command_path: tuple[str, ...]) -> None:
    """查看总体帮助或指定命令的帮助，例如 help index、help config set。"""
    target_ctx = ctx.find_root()
    for position, name in enumerate(command_path):
        group = target_ctx.command
        if not isinstance(group, click.Group):
            parent_path = " ".join(command_path[:position])
            raise click.UsageError(f"命令 {parent_path!r} 没有子命令，无法查找 {name!r}", ctx)
        command = group.get_command(target_ctx, name)
        if command is None:
            path = " ".join(command_path[:position + 1])
            raise click.UsageError(f"没有命令：{path}", ctx)
        # 仅构建帮助上下文，不解析参数或执行各级命令回调。
        target_ctx = click.Context(command, info_name=name, parent=target_ctx)
    click.echo(target_ctx.get_help())


@main.command()
@click.argument("symbol")
@click.option("--dimension", "-d", help="指定分析维度 (financial/technical/valuation/industry/sentiment)")
@click.option("--refresh-cache", is_flag=True, help="强制刷新缓存")
@click.option("--no-llm", is_flag=True, help="仅输出数据，跳过 LLM 解读")
@click.option("--verbose", "-v", is_flag=True, help="显示采集和分析过程")
@click.option("--with-market", is_flag=True, help="在报告中嵌入大盘环境分析")
def analyze(symbol, dimension, refresh_cache, no_llm, verbose, with_market):
    """分析股票并生成研报"""
    from report.formatter import ReportFormatter
    from utils.config import Config
    from utils.symbols import (
        normalize_symbol,
        resolve_name,
        symbol_market,
        validate_symbol,
    )

    config = Config()
    if not _check_disclaimer(config):
        return

    if not validate_symbol(symbol):
        console.print(f"[red]无效的股票代码: {symbol}[/red]")
        console.print(
            "请输入 6 位数字代码（如 000001、600036），"
            "可选前缀 [bold]sh[/bold]（沪市）或 [bold]sz[/bold]（深市）"
        )
        sys.exit(1)

    symbol = normalize_symbol(symbol)
    market = symbol_market(symbol)

    llm_enabled = not no_llm and config.get("llm.enabled", True)
    pipeline = _build_pipeline(llm_enabled=llm_enabled)

    try:
        with _create_cli_progress() as progress:
            task_id = progress.add_task("正在查询股票名称...", total=None)

            name = resolve_name(symbol) or symbol

            if verbose:
                console.print(f"[dim]正在分析: {name} ({symbol})[/dim]")

            def on_progress(stage, current, total, label):
                progress.update(task_id, completed=current, total=total,
                               description=f"[{stage}] {label}")

            results, commentary, ctx = pipeline.run(
                symbol, name, market=market,
                dimension=dimension,
                refresh_cache=refresh_cache,
                on_progress=on_progress,
            )

            if not verbose:
                progress.update(task_id, visible=False)
    except Exception as e:  # noqa: BLE001 — CLI 顶层兜底，打印错误并退出
        console.print(f"[red]分析失败: {e}[/red]")
        sys.exit(1)

    # 大盘环境快照（可选，供报告中嵌入指数环境摘要）
    market_env = None
    if with_market:
        try:
            from index.pipeline import IndexPipeline
            index_pipeline = IndexPipeline()
            snapshot = index_pipeline.get_snapshot("000300")
            if snapshot:
                market_env = snapshot
        except Exception:  # noqa: BLE001 — 大盘快照为可选信息，失败静默跳过
            logger.debug("大盘快照获取失败，跳过")

    from report.payload import build_stock_payload
    from report.scoring import build_report
    from report.signal import load_signal_config
    signal_config = load_signal_config(config)
    report = build_report(symbol, name, results, commentary, ctx,
                          no_llm=not llm_enabled, market_env=market_env,
                          signal_cfg=signal_config)

    registry = getattr(pipeline, "_registry", None)
    llm_available = registry is not None and registry.get_llm_backend(
        config.get("llm.provider", "openai")) is not None
    payload = build_stock_payload(
        symbol, name, results, commentary, ctx, no_llm=not llm_enabled,
        signal_cfg=signal_config, llm_available=llm_available,
    )
    saved_path = ReportFormatter.save(report, symbol, category="stock", stock_payload=payload)
    console.print(ReportFormatter.to_rich_markdown(report))
    console.print(f"\n[dim]报告已保存至: {saved_path}[/dim]")


@main.command()
@click.option("--refresh", "--refresh-cache", "refresh_cache", is_flag=True,
              help="强制刷新黄金数据缓存")
@click.option("--no-llm", is_flag=True, help="兼容旧用法；黄金简报使用确定性解释")
@click.option("--output", "-o", type=click.Choice(["terminal", "markdown"]),
              default="terminal", show_default=True, help="输出格式；两种格式均归档完整报告")
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path),
              default=None, help="报告根目录（默认当前目录的 reports）")
def gold(refresh_cache: bool, no_llm: bool, output: str, output_dir: Path | None) -> None:
    """生成以国内与国际金价为核心的黄金市场观察简报。"""
    from gold.service import GoldService
    from report.formatter import ReportFormatter

    config = Config()
    if not _check_disclaimer(config):
        return
    report_root = output_dir or Path.cwd() / "reports"
    try:
        payload = GoldService(config=config, output_dir=report_root).run(
            refresh_cache=refresh_cache, no_llm=no_llm,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        logger.warning("黄金 CLI 生成报告失败：%s", exc)
        raise click.ClickException(f"黄金研报生成失败：{exc}") from exc
    if output == "terminal":
        console.print(ReportFormatter.to_rich_markdown(payload["markdown"]))
    click.echo("\n报告已保存至: " + str(report_root / payload["report_path"]))


def _render_index_report(report):
    """终端与下载报告复用同一渲染入口。"""
    from rich.markdown import Markdown

    from index.build_single import render_index_report_markdown

    return Markdown(render_index_report_markdown(report))


def _render_index_report_md(report) -> str:
    """渲染可保存的完整指数报告。"""
    from index.build_single import render_index_report_markdown

    return render_index_report_markdown(report)


def _render_compare_table(compare) -> Table | str:
    """渲染横向对比表格"""
    if not compare or not compare.rows:
        return ""
    from rich.table import Table
    # CompareTable 表头为中文、行内 key 为英文，需映射后再取值
    header_key_map = {
        "指数名称": "name", "最新点位": "latest", "涨跌幅": "change",
        "PE 分位": "pe_pct", "PB 分位": "pb_pct", "趋势": "trend",
        "估值": "valuation", "资金": "capital", "综合评级": "composite",
    }
    table = Table(title="指数横向对比", caption=getattr(compare, "valuation_cutoff", ""))
    for h in compare.headers:
        table.add_column(h)
    for row in compare.rows:
        table.add_row(*[str(row["capital_display"]["label"] if h == "资金" and row.get("capital_display")
                            else row.get(header_key_map.get(h, h), "")) for h in compare.headers])
    return table


@main.command()
@click.argument("symbols", nargs=-1, required=True)
@click.option("--style", "-s", type=click.Choice(["broad", "sector", "overseas", "strategy"]),
              help="指数类别（默认自动检测）")
@click.option("--output", "-o", type=click.Choice(["terminal", "markdown"]),
              default="terminal", help="输出格式")
@click.option("--compare-only", is_flag=True, help="仅输出横向对比表格")
def index(symbols, style, output, compare_only):
    """分析指数并生成报告"""
    from dataclasses import asdict

    from data.index_mapping import ETFIndexMapping, IndexMapping
    from data.schemas import AnalysisTarget
    from index.pipeline import IndexPipeline
    from utils.config import Config
    from utils.symbols import normalize_index_symbol, validate_index_symbol

    config = Config()
    if not _check_disclaimer(config):
        return

    mapping = IndexMapping()
    targets = []

    etf_mapping = ETFIndexMapping()
    for raw in symbols:
        etf = etf_mapping.lookup(raw)
        raw_index = etf.index_symbol if etf else raw
        try:
            entry = mapping.resolve(raw_index)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        if entry is None and ":" in raw_index:
            raise click.ClickException(f"指数 {raw_index} 不在已验证目录中，请先刷新目录")
        if entry is None and not validate_index_symbol(raw_index):
            raise click.ClickException(f"无效的指数代码或名称: {raw}")
        normalized = normalize_index_symbol(raw_index)
        if entry is None:
            if style is None:
                raise click.ClickException(
                    f"无法识别指数 {normalized}，请用 --style 指定类别 (broad/sector/overseas/strategy)")
            index_style, name, market = style, raw, "a-shares"
        else:
            if entry.market not in ("a-shares", "hk", "us"):
                raise click.ClickException(f"指数 {entry.symbol} 的市场尚未核验，暂不可分析")
            index_style, name, market = entry.index_style, entry.name, entry.market
            normalized = entry.symbol
        targets.append(AnalysisTarget(
            target_type="index", symbol=normalized, name=name,
            market=market, index_style=index_style,
            requested_instrument=asdict(etf) if etf else None,
            return_type=(entry.return_type or "price") if entry is not None else "unknown",
            official_index_code=entry.price_symbol if entry is not None and entry.provider == "hsi" else None,
        ))

    from api.bootstrap import build_llm

    pipeline = IndexPipeline(config=config, llm=build_llm(config))
    with _create_cli_progress() as progress:
        task_id = progress.add_task("正在分析指数...", total=None)

        def on_progress(stage, current, total, label):
            progress.update(task_id, completed=current, total=total,
                           description=f"[{stage}] {label}")

        result = pipeline.run(targets, on_progress=on_progress)
        progress.update(task_id, visible=False)


    if not compare_only:
        for report in result.reports:
            if output == "terminal":
                console.print(_render_index_report(report))
            elif output == "markdown":
                from index.archives import save_index_report
                saved = save_index_report(report)
                console.print(f"[green]报告已保存: {saved}[/green]")

    if result.compare is not None:
        console.print(_render_compare_table(result.compare))

    if result.errors:
        for err in result.errors:
            console.print(f"[yellow]警告: {err}[/yellow]")


@main.command()
@click.argument("symbol")
@click.option("--strategy", default=None, help="策略 ID（默认读配置 backtest.default_strategy）")
@click.option("--start", required=True, type=click.DateTime(formats=["%Y-%m-%d"]),
              help="回测开始日期（YYYY-MM-DD）")
@click.option("--end", required=True, type=click.DateTime(formats=["%Y-%m-%d"]),
              help="回测结束日期（YYYY-MM-DD）")
@click.option("--benchmark", default=None, help="基准 ID（默认读配置 backtest.default_benchmark）")
def backtest(symbol, strategy, start, end, benchmark):
    """对单只股票执行技术信号回测并输出可复现产物"""
    from backtest.artifacts import write_backtest_artifacts
    from backtest.data import BacktestDataError, HistoricalPriceProvider
    from backtest.models import BacktestRequest
    from backtest.runner import BacktestRunner
    from backtest.strategy import StrategyConfigError, StrategyRepository
    from data.akshare import AkShareAdapter
    from utils.symbols import normalize_symbol, validate_symbol

    config = Config()
    if not _check_disclaimer(config):
        return

    if not validate_symbol(symbol):
        console.print(f"[red]✗ 无效的股票代码: {symbol}[/red]")
        console.print(
            "请输入 6 位数字代码（如 000001、600036），"
            "可选前缀 [bold]sh[/bold]（沪市）或 [bold]sz[/bold]（深市）"
        )
        sys.exit(1)
    symbol = normalize_symbol(symbol)

    strategy_id = strategy or config.get("backtest.default_strategy")
    benchmark_id = benchmark or config.get("backtest.default_benchmark")

    # 基准 ID 必须存在于配置；不存在时提前给出可选列表，避免空跑取数
    benchmarks = config.get("backtest.benchmarks", {})
    if not isinstance(benchmarks, dict) or benchmark_id not in benchmarks:
        available = ", ".join(benchmarks.keys()) if isinstance(benchmarks, dict) else ""
        console.print(f"[red]✗ 未配置基准: {benchmark_id}（可选: {available}）[/red]")
        sys.exit(1)

    try:
        strategies_dir = resource_root() / "config" / "strategies"
        strategy_repo = StrategyRepository(strategies_dir)
        strategy_repo.get(strategy_id)  # 提前校验策略存在，失败时快速红字退出

        request = BacktestRequest(
            symbol=symbol,
            start_date=start.date(),
            end_date=end.date(),
            strategy_id=strategy_id,
            benchmark_id=benchmark_id,
            initial_cash=float(config.get("backtest.initial_cash", 100000.0)),
        )
        runner = BacktestRunner(
            provider=HistoricalPriceProvider(AkShareAdapter()),
            strategies=strategy_repo,
            config=config,
        )
        result = runner.run(request)
        output = write_backtest_artifacts(result)
    except (BacktestDataError, StrategyConfigError, ValueError) as e:
        console.print(f"[red]✗ 回测失败: {e}[/red]")
        sys.exit(1)
    except Exception as e:  # CLI 顶层兜底，数据源/仿真异常类型不可预测（logger.exception 豁免 BLE001）
        logger.exception("回测失败")
        console.print(f"[red]✗ 回测失败: {e}[/red]")
        sys.exit(1)

    table = Table(title=f"回测指标：{symbol}")
    table.add_column("指标", style="cyan")
    table.add_column("数值", justify="right", style="green")
    for key, value in result.metrics.items():
        table.add_row(key, f"{value:.4f}")
    console.print(table)
    for warning in result.warnings:
        console.print(f"[yellow]⚠ {warning}[/yellow]")
    console.print(f"[green]报告已保存: {output}[/green]")


@main.command("industry-mapping")
@click.argument("action_or_symbol", required=False)
@click.option("--resume", is_flag=True, help="从候选文件的断点继续全量构建")
@click.option("--delay", type=click.FloatRange(min=0.5, max=30.0), default=1.5,
              show_default=True, help="行业请求基础间隔（秒）；限流时建议 6 秒以上")
@click.option("--verbose", "-v", is_flag=True, help="显示抓取明细")
def industry_mapping(action_or_symbol, resume, delay, verbose):
    """重建/更新行业映射表。

    validate → 抽样校验上游页面与解析规则；
    rebuild → 构建候选文件（可配合 --resume 续跑）；
    publish → 校验后原子发布候选文件；
    migrate-placeholders → 将旧占位分类迁移为显式缺失；
    股票代码 → 单只秒级更新。
    """
    from data.collection_state import make_refresh_record
    from data.industry_mapping_builder import (
        IndustryMappingError,
        migrate_placeholder_rows,
        publish_candidate,
        rebuild_all,
        update_symbol,
        validate_sample,
    )
    from utils.config import Config
    from utils.symbols import normalize_symbol, validate_symbol

    config = Config()
    if not _check_disclaimer(config):
        return

    action = action_or_symbol or "rebuild"
    try:
        if action == "validate":
            result = validate_sample()
            console.print(
                f"[green]✓ 抽样校验通过：{result['stock_count']} 只股票，"
                f"有效分类率 {result['valid_classification_rate']}%[/green]"
            )
        elif action == "publish":
            result = publish_candidate()
            console.print(
                f"[green]✓ 候选映射已发布：{result['stock_count']} 只股票，"
                f"有效分类率 {result['valid_classification_rate']}%[/green]"
            )
            record = result.get("refresh_record")
            if record is not None:
                console.print(f"刷新记录: {record.source} {record.result_state} published={record.published} {record.error_message}")
        elif action == "migrate-placeholders":
            result = migrate_placeholder_rows()
            console.print(
                f"[green]✓ 已迁移 {result['migrated_count']} 条占位记录，"
                f"映射表共 {result['stock_count']} 条[/green]"
            )
        elif action != "rebuild":
            if not validate_symbol(action):
                console.print(f"[red]无效的股票代码或操作: {action}[/red]")
                sys.exit(1)
            symbol = normalize_symbol(action)
            result = update_symbol(symbol)
            console.print(
                f"[green]✓ {result['symbol']} → {result['sw_level1']}"
                f"/{result['sw_level2']}（{result['style_category']}）"
                f"[/green] [dim]({'更新' if result['action'] == 'updated' else '新增'})[/dim]"
            )
        else:
            from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                console=console,
                transient=True,
            ) as progress:
                task_id = progress.add_task("正在重建行业映射表", total=335)

                def on_progress(current, total, label):
                    progress.update(task_id, completed=current, total=total,
                                    description=f"[{current}/{total}] {label}")

                result = rebuild_all(on_progress=on_progress, resume=resume, delay=delay)
                progress.update(task_id, visible=False)

            coverage = result["coverage_pct"]
            console.print(
                f"[green]✓ 行业映射候选构建完成：{result['stock_count']} 只股票，"
                f"覆盖率 {coverage}%，有效分类率 {result['valid_classification_rate']}%[/green]"
            )
            console.print(f"[dim]候选文件：{result['candidate_path']}；确认后执行 industry-mapping publish[/dim]")
            record = result.get("refresh_record")
            if record is not None:
                console.print(f"刷新记录: {record.source} {record.result_state} published={record.published} {record.error_message}")
            if result["failed_industries"]:
                console.print(
                    f"[yellow]⚠ 失败行业 {len(result['failed_industries'])} 个: "
                    f"{', '.join(result['failed_industries'])}[/yellow]"
                )
            if verbose:
                console.print(f"[dim]遍历行业 {result['total_industries']} 个[/dim]")
    except IndustryMappingError as e:
        if action in ("rebuild", "publish"):
            record = make_refresh_record("乐咕申万", "failed", error_kind="refresh_failed",
                                         error_message="行业映射操作失败")
            console.print(f"刷新记录: {record.source} {record.result_state} published={record.published} {record.error_message}")
        console.print(f"[red]✗ 行业映射操作失败: {e}[/red]")
        sys.exit(1)
    except OSError:
        if action in ("rebuild", "publish"):
            record = make_refresh_record("乐咕申万", "failed", error_kind="refresh_failed",
                                         error_message="行业映射文件操作失败")
            console.print(f"刷新记录: {record.source} {record.result_state} published={record.published} {record.error_message}")
        raise


@main.command("hk-industry-refresh")
def hk_industry_refresh() -> None:
    """依次刷新港股通成员资格和恒生 HSICS 行业快照。"""
    from datetime import datetime

    from filelock import Timeout

    from data.collection_state import make_refresh_record
    from data.hk_hsics_industry import HsicsIndustryError, load_connect_symbols, refresh
    from data.hk_stock_connect import (
        HKStockConnectError,
        fetch_sse_snapshot,
        publish_snapshot,
    )
    from utils.paths import (
        hk_hsics_industry_path,
        hk_hsics_snapshot_path,
        hk_stock_connect_path,
    )

    attempted_at = datetime.now().astimezone()
    connect_path = hk_stock_connect_path(writable=True)
    try:
        connect_snapshot = fetch_sse_snapshot()
        publish_snapshot(connect_snapshot, connect_path)
    except (HKStockConnectError, OSError, ValueError, Timeout) as exc:
        record = make_refresh_record("上交所港股通+东财F10+HSCI", "failed", attempted_at=attempted_at,
                                     error_kind="fetch_failed", error_message="港股通刷新失败")
        click.echo(f"刷新记录: {record.source} {record.result_state} published={record.published}")
        raise click.ClickException(f"港股通成员快照刷新失败：{exc}") from exc

    try:
        symbols = load_connect_symbols()
        hsics_snapshot = refresh(
            hk_hsics_industry_path(writable=True),
            hk_hsics_snapshot_path(writable=True),
            connect_symbols=symbols,
        )
    except (HsicsIndustryError, OSError, ValueError, Timeout) as exc:
        record = make_refresh_record("上交所港股通+东财F10+HSCI", "partial", attempted_at=attempted_at,
                                     error_kind="fetch_failed", error_message="HSICS 刷新失败")
        click.echo(f"刷新记录: {record.source} {record.result_state} published={record.published}")
        raise click.ClickException(
            f"港股通成员快照已更新（{connect_snapshot.total} 只）；HSICS 行业快照刷新失败：{exc}"
        ) from exc

    click.echo(
        f"港股通成员 {connect_snapshot.total} 只；HSICS 总数 {hsics_snapshot.total}，"
        f"二级 {hsics_snapshot.sector_count}，一级 {hsics_snapshot.industry_count}，"
        f"数据日期 {hsics_snapshot.data_date.isoformat()}"
    )
    record = make_refresh_record("上交所港股通+东财F10+HSCI", "complete", attempted_at=attempted_at,
                                 published=True)
    click.echo(f"刷新记录: {record.source} {record.result_state} published={record.published}")


@main.group()
def config():
    """管理配置"""


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key, value):
    """设置配置项"""
    from utils.config import Config
    cfg = Config()
    converted = _convert_value(value)
    cfg.set(key, converted)
    console.print(f"[green]✓ {key} = {converted}[/green]")


@config.command("get")
@click.argument("key")
def config_get(key):
    """获取配置项"""
    from utils.config import Config
    cfg = Config()
    val = cfg.get(key)
    console.print(f"{key} = {val}")


@main.group()
def cache():
    """管理缓存"""


@cache.command("clear")
@click.option("--data-type", type=click.Choice(
    ["price", "financial", "valuation", "industry", "news"]),
    help="仅清空指定数据类型的缓存")
def cache_clear(data_type):
    """清空全部或指定类型的缓存"""
    c = _get_cache()
    if data_type:
        count = c.invalidate_data_type(data_type)
        console.print(f"[green]✓ 已清空 {data_type} 缓存（{count} 条）[/green]")
    else:
        c.clear()
        console.print("[green]✓ 缓存已清空[/green]")


@cache.command("status")
def cache_status():
    """查看缓存状态"""
    c = _get_cache()
    stats = c.stats()
    table = Table(title="缓存状态")
    table.add_column("指标", style="cyan")
    table.add_column("数值", style="green")
    table.add_row("缓存条目", str(stats["total_entries"]))
    table.add_row("数据库大小", f"{stats['db_size_bytes'] / 1024:.1f} KB")
    console.print(table)


def _resolve_api_bind(host: str | None, port: int | None, config: Config) -> tuple[str, int]:
    """解析 API 监听地址：CLI 显式参数 > 配置文件 > 默认值"""
    resolved_host = host or config.get("api.host", DEFAULT_CONFIG["api"]["host"])
    resolved_port = port if port is not None else config.get("api.port", DEFAULT_CONFIG["api"]["port"])
    return resolved_host, resolved_port


def _runtime_uvicorn_log_config(runtime_log_path: Path) -> dict:
    """构建同时写入控制台和本次服务日志的 Uvicorn 配置。"""
    from uvicorn.config import LOGGING_CONFIG

    log_config = deepcopy(LOGGING_CONFIG)
    log_config["formatters"]["runtime_default"] = {
        "()": "uvicorn.logging.DefaultFormatter",
        "fmt": "%(asctime)s %(levelprefix)s %(message)s",
        "use_colors": False,
    }
    log_config["formatters"]["runtime_access"] = {
        "()": "uvicorn.logging.AccessFormatter",
        "fmt": '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
        "use_colors": False,
    }
    log_config["handlers"]["runtime_default"] = {
        "class": "logging.FileHandler",
        "encoding": "utf-8",
        "filename": str(runtime_log_path),
        "formatter": "runtime_default",
    }
    log_config["handlers"]["runtime_access"] = {
        "class": "logging.FileHandler",
        "encoding": "utf-8",
        "filename": str(runtime_log_path),
        "formatter": "runtime_access",
    }
    log_config["loggers"]["uvicorn"]["handlers"].append("runtime_default")
    log_config["loggers"]["uvicorn.access"]["handlers"].append("runtime_access")
    return log_config


def _run_web_server(app, host: str, port: int, runtime_log_path: Path) -> None:
    """运行 Web 服务，并将访问日志写入本次服务的日志文件。"""
    import uvicorn

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_config=_runtime_uvicorn_log_config(runtime_log_path),
    )


@main.command("run")
@click.option("--host", default=None, help=f"监听地址（默认读配置 api.host，缺省 {DEFAULT_CONFIG['api']['host']}）")
@click.option("--port", default=None, type=int, help=f"监听端口（默认读配置 api.port，缺省 {DEFAULT_CONFIG['api']['port']}）")
def run(host, port):
    """启动 Web API 服务（含 Web UI）"""
    from api.app import create_app
    from api.bootstrap import build_agent_core
    from maintenance.runner import MaintenanceRunner

    config = Config()
    runtime_log_path = _configure_runtime_log(config)
    try:
        runtime_log_offset = runtime_log_path.stat().st_size
    except FileNotFoundError:
        runtime_log_offset = 0
    bind_host, bind_port = _resolve_api_bind(host, port, config)
    with _create_cli_progress() as progress:
        task_id = progress.add_task("正在构建 Agent 核心", total=3)
        core = build_agent_core(config)
        progress.update(task_id, completed=1, description="正在创建 Web 应用")
        app = create_app(
            core=core,
            log_start_offset=runtime_log_offset,
            runtime_log_path=runtime_log_path,
            maintenance=MaintenanceRunner(
                Path.cwd() / "reports", state_dir=config.config_dir
            ),
        )
        progress.update(task_id, completed=3, description="正在启动 HTTP 服务")

    if bind_host in {"0.0.0.0", "::"}:
        logger.info("%s API 已对外监听于 %s:%s", APP_NAME, bind_host, bind_port)
        console.print(f"[green]Web 服务已监听 {bind_host}:{bind_port}，请使用本机或服务器实际 IP 访问[/green]")
    else:
        web_url = f"http://{bind_host}:{bind_port}"
        logger.info("%s API 启动于 %s", APP_NAME, web_url)
        logger.info("Web 服务正在运行: %s", web_url)
        console.print(f"[green]Web 服务正在运行: {web_url}[/green]")
    from api.auth import auth_credentials

    if bool(config.get("api.auth.enabled", True)):
        _, auth_password = auth_credentials(config)
        console.print("[cyan]访问验证已启用：本机 127.0.0.1/::1 免验证，外部访问需输入账号密码[/cyan]")
        if not auth_password:
            warning = "未配置 api.auth.password 或 STOCK_ROBOT_AUTH_PASSWORD，外部访问将被拒绝"
            logger.warning(warning)
            console.print(f"[yellow]警告：{warning}[/yellow]")
    else:
        warning = "访问验证已关闭（api.auth.enabled=false），外部访问无任何保护"
        logger.warning(warning)
        console.print(f"[yellow]警告：{warning}[/yellow]")
    _run_web_server(app, bind_host, bind_port, runtime_log_path)


@main.command()
@click.option("--ask", "-a", default=None, help="单次对话（非交互式）")
@click.option("--verbose", "-v", is_flag=True, help="显示计划和工具调用细节")
def chat(ask, verbose):
    """进入 AI Agent 对话模式，支持复杂投研任务的自主拆解和分析"""
    from agent.chat import ChatResponder
    from agent.executor import Executor
    from agent.graph import DEFAULT_CHECKPOINT_DIR
    from agent.memory import Memory
    from agent.planner import Planner
    from api.bootstrap import build_agent_core
    from output.renderer import RichRenderer
    from utils.config import Config

    config = Config()
    renderer = RichRenderer(console=console)
    core = build_agent_core(config)

    memory = Memory()
    planner = Planner(llm=core.llm, registry=core.registry, memory=memory)
    executor = Executor(registry=core.registry, memory=memory, model=core.model,
                        persist_dir=str(DEFAULT_CHECKPOINT_DIR))
    chat_responder = ChatResponder(model=core.model)

    if ask:
        _run_agent_query(ask, planner, executor, memory, renderer, chat_responder)
        return

    _run_interactive_chat(planner, executor, memory, renderer, chat_responder)


def _run_agent_query(query, planner, executor, memory, renderer, chat_responder):
    """单次 Agent 查询"""
    plan = planner.plan(query)
    if plan.mode == "agent":
        # CLI 暂不接入自主循环：映射为单步 plan（与 planner 降级路径一致）
        from agent.memory import Plan, TaskStep

        plan = Plan(goal=plan.goal,
                    steps=[TaskStep(id="step-1", description=plan.goal)])
    if plan.mode == "chat":
        import asyncio
        # 与 API 路径对称：先写 user 再回复，保证 memory 有完整 user/assistant 轮次
        memory.add_message("user", query)
        reply = asyncio.run(chat_responder.reply(query, memory))
        memory.add_message("assistant", reply)
        console.print(reply)
        return
    console.print(renderer.render_plan(plan))

    import asyncio
    result = asyncio.run(executor.execute(plan))

    console.print(renderer.render_summary(result))


def _run_interactive_chat(planner, executor, memory, renderer, chat_responder):
    """交互式对话循环"""
    console.print(f"[bold]{APP_NAME} Agent[/bold] — AI 驱动的投资研究助手")
    console.print("输入你的投研问题；输入 /help 查看聊天命令，输入 /exit 退出。")
    console.print("[dim]终端命令帮助：stock-robot help[/dim]\n")

    while True:
        try:
            user_input = click.prompt("你", prompt_suffix="> ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n再见！")
            break

        if not user_input:
            continue

        # 处理快捷指令
        result = _handle_slash_command(user_input, memory, renderer)
        if result == "exit":
            break
        if result:
            continue

        plan = planner.plan(user_input)
        if plan.mode == "agent":
            # CLI 暂不接入自主循环：映射为单步 plan（与 planner 降级路径一致）
            from agent.memory import Plan, TaskStep

            plan = Plan(goal=plan.goal,
                        steps=[TaskStep(id="step-1", description=plan.goal)])
        if plan.mode == "chat":
            import asyncio
            # 与 API 路径对称：先写 user 再回复，保证 memory 有完整 user/assistant 轮次
            memory.add_message("user", user_input)
            reply = asyncio.run(chat_responder.reply(user_input, memory))
            memory.add_message("assistant", reply)
            console.print(reply)
            continue

        console.print(renderer.render_plan(plan))

        import asyncio
        result = asyncio.run(executor.execute(plan))

        console.print(renderer.render_summary(result))


def _handle_slash_command(text, memory, renderer):
    """处理 / 开头的快捷指令，返回 'exit' 表示退出，True 表示已处理"""
    cmd = text.strip().lower()

    if cmd == "/exit":
        console.print("再见！")
        return "exit"

    if cmd == "/help":
        console.print("""
[bold]可用快捷指令:[/bold]
  /help     - 显示此帮助
  /tools    - 列出可用工具
  /plan     - 显示最近一次执行计划
  /clear    - 清空当前会话上下文
  /verbose  - 切换详细输出模式
  /exit     - 退出对话模式
        """.strip())
        return True

    if cmd == "/tools":
        console.print("[dim]工具列表功能需要在上下文中访问 registry，暂时不可用[/dim]")
        return True

    if cmd == "/plan":
        last = memory.get_last_plan()
        if last is None:
            console.print("[dim]暂无执行计划[/dim]")
        else:
            console.print(renderer.render_plan(last))
        return True

    if cmd == "/clear":
        memory.clear_session()
        console.print("[dim]会话上下文已清空[/dim]")
        return True

    if cmd == "/verbose":
        console.print("[dim]详细模式已切换[/dim]")
        return True

    return False


# ---------------------------------------------------------------------------
# RAG 知识库管理命令组
# ---------------------------------------------------------------------------

@main.group()
def rag():
    """知识库管理 — 文档摄入、清理、统计"""


def _create_rag_engine():
    """创建引擎，仅将明确缺失的可选包转成安装提示。"""
    try:
        from rag.engine import RAGEngine

        return RAGEngine()
    except ModuleNotFoundError as exc:
        if exc.name not in {"chromadb", "sentence_transformers"}:
            raise
        raise click.ClickException(
            f'RAG 可选依赖未安装（{exc.name}）。安装：pip install -e ".[rag]"'
        ) from exc


@rag.command("ingest")
@click.argument("path")
@click.option("--source-type", "-s", required=True,
              type=click.Choice([
                  "research_reports", "financial_filings",
                  "policy_macro", "academic",
                  "history_reports", "system_rules",
              ]),
              help="知识库类型")
@click.option("--title", "-t", default="", help="文档标题")
@click.option("--date", "-d", default="", help="文档日期 (YYYY-MM-DD)")
@click.option("--symbol", multiple=True, help="关联股票代码（可多次指定）")
@click.option("--tag", multiple=True, help="内容标签（可多次指定）")
def rag_ingest(path, source_type, title, date, symbol, tag):
    """摄入文档或目录到知识库

    PATH 可以是单个文件或目录路径。
    """
    import os

    console.print("[bold]正在摄入知识库...[/bold]")
    console.print(f"  类型: {source_type}")
    console.print(f"  路径: {path}")

    engine = _create_rag_engine()
    symbols = list(symbol)
    tags = list(tag)

    if os.path.isdir(path):
        console.print("  模式: 目录批量导入")
        results = engine.ingest_directory(
            directory=path,
            source_type=source_type,
            title_prefix=title,
            symbols=symbols,
            tags=tags,
        )
        succeeded = sum(1 for r in results if r.get("status") == "success")
        skipped = sum(1 for r in results if r.get("status") == "skipped")
        failed = sum(1 for r in results if r.get("status") == "error")
        console.print(
            f"[green]✓ 成功: {succeeded}[/green]  "
            f"[yellow]跳过: {skipped}[/yellow]  "
            f"[red]失败: {failed}[/red]"
        )
        for r in results:
            if r.get("status") == "error":
                console.print(f"  [red]✗ {r.get('file_path', '?')}: {r.get('reason')}[/red]")
    else:
        result = engine.ingest_file(
            file_path=path,
            source_type=source_type,
            title=title,
            date=date,
            symbols=symbols,
            tags=tags,
        )
        if result["status"] == "success":
            console.print(
                f"[green]✓ 摄入成功: {result['chunks_count']} 个分块[/green]"
            )
            console.print(f"  哈希: {result.get('source_hash', '')[:16]}...")
        elif result["status"] == "skipped":
            console.print(f"[yellow]跳过: {result.get('reason', '?')}[/yellow]")
        else:
            console.print(f"[red]✗ 失败: {result.get('reason', '?')}[/red]")


@rag.command("clean")
@click.option("--source", "-s", "source_type",
              type=click.Choice([
                  "research_reports", "financial_filings",
                  "policy_macro", "academic",
                  "history_reports", "system_rules",
              ]),
              help="清除指定知识库类型")
@click.option("--before", "before_date", default="",
              help="清除指定日期前的文档 (YYYY-MM-DD)")
@click.option("--symbol", default="",
              help="清除指定股票关联的文档")
@click.option("--dry-run", is_flag=True, default=False,
              help="预览，不实际删除")
def rag_clean(source_type, before_date, symbol, dry_run):
    """清理知识库中的文档"""
    engine = _create_rag_engine()

    if dry_run:
        console.print("[bold yellow]DRY RUN 模式 — 仅预览，不实际删除[/bold yellow]\n")

    if source_type:
        if dry_run:
            stats = engine.collection_stats()
            for s in stats:
                if s["name"] == source_type:
                    console.print(
                        f"将清除 [bold]{source_type}[/bold] 全部 "
                        f"[bold]{s['count']}[/bold] 个文档"
                    )
                    break
        else:
            result = engine.delete_by_source_type(source_type)
            console.print(f"[green]✓ 已清除 {source_type} 知识库[/green]")

    elif before_date:
        if dry_run:
            console.print(f"将清除 [bold]{before_date}[/bold] 之前的文档")
        else:
            result = engine.delete_before_date(before_date)
            console.print(
                f"[green]✓ 已清除 {result['deleted']} 个文档"
                f"（{before_date} 之前）[/green]"
            )

    elif symbol:
        if dry_run:
            console.print(f"将清除与股票 [bold]{symbol}[/bold] 关联的文档")
        else:
            result = engine.delete_by_symbol(symbol)
            console.print(
                f"[green]✓ 已清除 {result['deleted']} 个与 {symbol} 关联的文档[/green]"
            )

    else:
        console.print(
            "[yellow]请指定清除条件: --source / --before / --symbol[/yellow]"
        )


@rag.command("stats")
def rag_stats():
    """查看知识库各 Collection 统计信息"""
    engine = _create_rag_engine()
    stats = engine.collection_stats()
    embedding_name = engine.embedding_name

    table = Table(title="RAG 知识库统计")
    table.add_column("Collection", style="cyan")
    table.add_column("文档数", justify="right")
    table.add_column("状态", style="green")

    total = 0
    for entry in stats:
        table.add_row(
            entry["name"],
            str(entry["count"]),
            "✓" if entry["count"] > 0 else "空",
        )
        total += entry["count"]

    table.add_section()
    table.add_row("[bold]合计[/bold]", f"[bold]{total}[/bold]", "")
    console.print(table)
    console.print(f"\n[dim]Embedding 模型: {embedding_name}[/dim]")


@main.group()
def subscribe():
    """管理每日邮箱推送订阅"""


@subscribe.command("add")
@click.option("--name", required=True, help="订阅名称")
@click.option("--symbols", required=True, help="标的代码，逗号/空格分隔")
@click.option("--channel", type=click.Choice(["email"]), default="email",
              help="推送渠道")
@click.option("--time", "push_time", required=True, help="每日推送时间 HH:MM")
@click.option("--kind", type=click.Choice(["auto", "stock", "index"]),
              default="auto", help="标的类型（默认 auto 自动判定）")
@click.option("--index-style", type=click.Choice(["broad", "sector", "overseas", "strategy"]),
              default=None, help="指数风格（kind=index 时使用）")
def subscribe_add(name, symbols, channel, push_time, kind, index_style):
    """创建订阅"""
    import re
    from datetime import datetime
    from typing import Any

    from push.models import Subscription
    from push.names import resolve_subscription
    from push.store import PushStore
    from utils.config import Config

    config = Config()
    store = PushStore(config.config_dir / "push.db")
    # 订阅级类型选项作为所有标的的默认值（API 支持 per-symbol 覆盖）；
    # pydantic before-validator 接受 dict 简写，标注 list[Any] 规避静态类型误报
    raw_symbols: list[Any] = [
        {"symbol": s, "kind": kind, "index_style": index_style}
        for s in re.split(r"[,，\s]+", symbols) if s
    ]
    sub = Subscription(
        name=name,
        symbols=raw_symbols,
        channel=channel,
        time=push_time,
        created_at=datetime.now().astimezone().isoformat(),
    )
    try:
        resolve_subscription(sub)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--symbols") from exc
    sub_id = store.create(sub)
    console.print(f"[green]已创建订阅 #{sub_id}: {name}（{channel} {push_time}）[/green]")
    console.print("[yellow]若 Web 服务正在运行，请重启服务以注册新的每日定时任务。[/yellow]")


@subscribe.command("list")
def subscribe_list():
    """列出全部订阅"""
    from push.store import PushStore
    from utils.config import Config

    store = PushStore(Config().config_dir / "push.db")
    table = Table(title="推送订阅")
    table.add_column("ID", style="cyan")
    table.add_column("名称", style="white")
    table.add_column("标的", style="yellow")
    table.add_column("渠道", style="green")
    table.add_column("时间", style="magenta")
    table.add_column("状态", style="green")
    for sub in store.list():
        last = store.last_run(sub.id) if sub.id else None
        status = "启用" if sub.enabled else "停用"
        if last:
            status += f"（上次 {last['ok']}/{last['total']} 成功）"
        table.add_row(str(sub.id), sub.name,
                      "、".join(s.symbol for s in sub.symbols),
                      sub.channel, sub.time, status)
    console.print(table)


@subscribe.command("remove")
@click.option("--id", "sub_id", type=int, required=True, help="订阅 ID")
def subscribe_remove(sub_id):
    """删除订阅"""
    from push.store import ActiveRunError, PushStore
    from utils.config import Config

    store = PushStore(Config().config_dir / "push.db")
    try:
        deleted = store.delete(sub_id)
    except ActiveRunError as exc:
        raise click.ClickException(str(exc)) from exc
    if deleted:
        console.print(f"[green]已删除订阅 #{sub_id}[/green]")
    else:
        console.print(f"[red]订阅 #{sub_id} 不存在[/red]")


@subscribe.command("enable")
@click.option("--id", "sub_id", type=int, required=True, help="订阅 ID")
def subscribe_enable(sub_id):
    """启用订阅"""
    _set_enabled(sub_id, True)


@subscribe.command("disable")
@click.option("--id", "sub_id", type=int, required=True, help="订阅 ID")
def subscribe_disable(sub_id):
    """停用订阅"""
    _set_enabled(sub_id, False)


def _set_enabled(sub_id: int, enabled: bool):
    from push.store import PushStore
    from utils.config import Config

    store = PushStore(Config().config_dir / "push.db")
    sub = store.get(sub_id)
    if sub is None:
        console.print(f"[red]订阅 #{sub_id} 不存在[/red]")
        return
    sub.enabled = enabled
    store.update(sub)
    console.print(f"[green]订阅 #{sub_id} 已{'启用' if enabled else '停用'}[/green]")
    if enabled:
        console.print("[yellow]若 Web 服务正在运行，请重启服务以注册该每日定时任务。[/yellow]")


@subscribe.command("run")
@click.option("--id", "sub_id", type=int, required=True, help="订阅 ID")
def subscribe_run(sub_id):
    """手动触发一次推送（同步执行，耗时取决于标的数）"""
    from push.executor import PushExecutor
    from push.store import ActiveRunError, MissingSubscriptionError, PushStore
    from utils.config import Config

    config = Config()
    store = PushStore(config.config_dir / "push.db")
    sub = store.get(sub_id)
    if sub is None:
        console.print(f"[red]订阅 #{sub_id} 不存在[/red]")
        return
    from api.bootstrap import build_agent_core
    core = build_agent_core(config)
    executor = PushExecutor(core, store, config)
    try:
        run_id = store.queue_run(sub, "manual")
    except (ActiveRunError, MissingSubscriptionError) as exc:
        raise click.ClickException(str(exc)) from exc
    with console.status("正在生成报告并推送..."):
        result = executor.run_subscription(sub, run_id)
    final_run = store.get_run(run_id)
    console.print(f"[green]运行 #{run_id}：{final_run['status'] if final_run else 'unknown'}，{result['ok']}/{result['total']} 成功[/green]")
    for failure in result["failures"]:
        console.print(f"[yellow]失败: {failure}[/yellow]")


def _radar_services():
    """构建配置雷达的本地服务。"""
    from radar.data import (
        AkShareETFDataProvider,
        FallbackETFDataProvider,
        LocalETFDataProvider,
        OfficialExchangeETFDataProvider,
        RequestPacer,
        SinaETFDataProvider,
        TencentETFDataProvider,
    )
    from radar.refresh import RadarRefresher
    from radar.store import RadarStore
    from radar.universe import UniverseRepository

    config = Config()
    root = resource_root()
    repository = UniverseRepository(root / "config" / "radar_universes")
    provider = FallbackETFDataProvider((
        LocalETFDataProvider(config.config_dir / "radar_local_data" / "etf"),
        AkShareETFDataProvider(pacer=RequestPacer(config.get("radar.minimum_interval_seconds", 1.0))),
        TencentETFDataProvider(),
        SinaETFDataProvider(),
        OfficialExchangeETFDataProvider(),
    ))
    store = RadarStore(config.config_dir / "radar.db")
    return config, repository, store, RadarRefresher(repository, provider, store)


@main.group()
def radar():
    """管理版本化 ETF/股票池的本地配置雷达。"""


@radar.group("universe")
def radar_universe():
    """查看已配置的标的池。"""


@radar_universe.command("list")
def radar_universe_list():
    """列出可刷新、不可混排的标的池。"""
    _, repository, _, _ = _radar_services()
    table = Table(title="配置雷达标的池")
    table.add_column("ID")
    table.add_column("名称")
    table.add_column("类型")
    table.add_column("版本", justify="right")
    for universe in repository.load_all():
        table.add_row(universe.id, universe.name, universe.asset_type, str(universe.version))
    console.print(table)


@radar.command("refresh")
@click.option("--universe", "universe_id", required=True, help="标的池 ID")
@click.option("--full", is_flag=True, help="扩展至完整历史窗口后重新生成快照")
@click.option("--as-of", type=click.DateTime(formats=["%Y-%m-%d"]), help="数据截至日期")
def radar_refresh(universe_id, full, as_of):
    """手动刷新一个标的池的本地快照。"""
    config, _, _, refresher = _radar_services()
    if not _check_disclaimer(config):
        return
    try:
        run_id = refresher.refresh(universe_id, full=full, as_of=as_of.date() if as_of else None)
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    console.print(f"[green]快照已完成: {run_id}[/green]")


@radar.command("collect")
@click.option("--hour", type=click.IntRange(0, 23), help="旧参数：时间改由持久设置管理")
@click.option("--minute", type=click.IntRange(0, 59), help="旧参数：时间改由持久设置管理")
@click.option("--once", is_flag=True, help="执行当前持久队列一次后退出")
@click.option("--universe", default=None, help="只采集指定启用池")
@click.option("--date", "target_date", type=click.DateTime(formats=["%Y-%m-%d"]), default=None)
def radar_collect(hour: int | None, minute: int | None, once: bool, universe: str | None, target_date: datetime | None):
    """兼容旧采集命令；单次与服务共用持久队列和执行锁。"""
    from radar.collector_service import build_collector

    if not once:
        # 旧启动器迁移完成前保留关闭标记，防止残留启动器唤起旧入口。
        if (Config().config_dir / "radar_collector.disabled").is_file():
            console.print("[yellow]自动采集已关闭，旧启动入口不启动；新服务请使用 radar daemon。[/yellow]")
            return
        if hour is not None or minute is not None:
            console.print("[yellow]采集时间改由持久设置管理，旧时间参数不覆盖已保存设置。[/yellow]")
        _serve_radar_collector()
        return
    worker = build_collector()
    try:
        runs = worker.enqueue_manual(datetime.now().astimezone(), universe, target_date.date() if target_date else None)
        while worker.execute_one():
            pass
        for run in runs:
            result = worker.store.get_run(run["id"])
            if result is not None:
                console.print(f"{run['universe_id']}: {result['status']} {result.get('snapshot_run_id') or ''}")
    except (ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc


def _serve_radar_collector() -> None:
    """退出信号交给执行器安全边界，自动关闭不退出此服务。"""
    import signal

    from filelock import Timeout

    from radar.collector_service import build_collector

    worker = build_collector()
    from radar.collector_startup import CollectorStartup

    def migrate() -> None:
        migration = CollectorStartup(Path.cwd()).migrate_legacy(state_dir=worker.state_dir)
        for error in migration.get("errors", []):
            worker.store.record_event("warning", str(error))
        if migration.get("unsafe_overlap"):
            raise click.ClickException("无法确认旧采集进程已退出，请先停止旧进程后重试")

    old_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, lambda *_args: worker.stop())
    console.print("[green]采集服务已启动，按持久计划执行；自动关闭时保持待命。[/green]")
    try:
        worker.serve(startup_hook=migrate)
    except Timeout as exc:
        raise click.ClickException("此项目已有采集服务运行") from exc
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


@radar.command("daemon")
def radar_daemon():
    """启动独立常驻采集服务，工作目录应固定为项目根。"""
    _serve_radar_collector()


@radar.command("collector-status")
def radar_collector_status():
    """显示持久计划及服务最近心跳，不启动采集。"""
    from radar.collector_service import build_collector

    worker = build_collector(refresh_calendar=False)
    settings = worker.store.settings()
    console.print(f"自动采集：{'启用' if settings['enabled'] else '关闭'} · {settings['hour']:02d}:{settings['minute']:02d}（北京时间）")
    console.print(worker.store.runtime())


@radar.command("service-unit")
@click.option("--user", required=True, help="Linux 普通服务账户")
@click.option("--output", type=click.Path(path_type=Path), default=None)
def radar_service_unit(user: str, output: Path | None):
    """生成 Linux systemd 模板，安装需部署人员执行。"""
    from radar.collector_startup import CollectorStartup

    startup = CollectorStartup(Path.cwd())
    try:
        unit = startup.linux_unit(user)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if output is None:
        console.print(unit, markup=False)
    else:
        output.write_text(unit, encoding="utf-8")
        console.print(f"已生成 {output}，安装为 /etc/systemd/system/{startup.unit_name} 后执行 systemctl daemon-reload 与 enable --now。")


@radar.command("import-data")
@click.option("--symbol", required=True, help="六位 ETF 代码")
@click.option("--file", "source_file", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="含日线的 CSV 文件")
def radar_import_data(symbol: str, source_file: Path):
    """导入本地 ETF 日线，供离线刷新与回测优先使用。"""
    from radar.data import LocalETFDataProvider, RadarDataError

    config = Config()
    provider = LocalETFDataProvider(config.config_dir / "radar_local_data" / "etf")
    try:
        rows, start, end = provider.import_csv(symbol, source_file)
    except RadarDataError as exc:
        raise click.ClickException(str(exc)) from exc
    console.print(f"[green]已导入 {symbol}: {rows} 行，覆盖 {start.isoformat()} 至 {end.isoformat()}[/green]")


@radar.command("import-benchmark")
@click.option("--benchmark", "benchmark_id", type=click.Choice(["money_fund", "csi_300", "csi_all_bond"]), required=True, help="雷达固定基准 ID")
@click.option("--file", "source_file", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="含日期和收盘价的 CSV 文件")
def radar_import_benchmark(benchmark_id: str, source_file: Path):
    """导入本地基准日线，供离线回测和标的比较优先使用。"""
    from radar.benchmarks import LocalBenchmarkStore, RadarBenchmarkError

    config = Config()
    store = LocalBenchmarkStore(config.config_dir / "radar_local_data" / "benchmarks")
    try:
        rows, start, end = store.import_csv(benchmark_id, source_file)
    except RadarBenchmarkError as exc:
        raise click.ClickException(str(exc)) from exc
    console.print(f"[green]已导入基准 {benchmark_id}: {rows} 行，覆盖 {start.isoformat()} 至 {end.isoformat()}[/green]")


@radar.command("status")
@click.option("--universe", "universe_id", required=True, help="标的池 ID")
def radar_status(universe_id):
    """显示最近完成快照的状态汇总。"""
    _, _, store, _ = _radar_services()
    summary = store.status_summary(universe_id)
    if summary is None:
        raise click.ClickException("尚无完成快照，请先执行 radar refresh")
    console.print(summary)


@radar.command("show")
@click.option("--universe", "universe_id", required=True, help="标的池 ID")
@click.option("--run-id", help="已完成快照 ID，省略时取最新")
@click.option("--category", help="仅显示指定类别")
def radar_show(universe_id, run_id, category):
    """显示最近完成快照的研究评分。"""
    _, _, store, _ = _radar_services()
    snapshot = store.get_snapshot(run_id) if run_id else store.latest_completed(universe_id)
    if snapshot is None:
        raise click.ClickException("尚无完成快照，请先执行 radar refresh")
    table = Table(title=f"配置雷达：{universe_id}（{snapshot['as_of_date']}）")
    for column in ("代码", "名称", "类别", "评分", "排名", "状态", "等级"):
        table.add_column(column)
    if snapshot["universe_id"] != universe_id:
        raise click.ClickException("快照不属于指定标的池")
    for item in snapshot["items"]:
        if category and item["category"] != category:
            continue
        score = "-" if item["score"] is None else f"{item['score']:.1f}"
        table.add_row(item["symbol"], item["name"], item["category"], score, str(item["rank"] or "-"), item["status"], item["grade"])
    console.print(table)


@radar.command("backtest")
@click.option("--universe", "universe_id", required=True, help="标的池 ID")
@click.option("--start", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--strategy", default="core_rotation_v1", show_default=True)
@click.option("--rerun", is_flag=True, help="忽略同参数的已完成产物并强制重新计算")
def radar_backtest(universe_id, start, end, strategy, rerun):
    """运行 ETF 月度轮动研究回测并写入可审计产物。"""
    import pandas as pd

    from radar.artifacts import (
        find_reusable_radar_backtest,
        write_radar_backtest_artifacts,
    )
    from radar.backtest import run_monthly_rotation
    from radar.benchmarks import (
        RadarBenchmarkError,
        fetch_benchmark_closes,
        load_benchmarks,
    )
    from radar.data import (
        AkShareETFDataProvider,
        FallbackETFDataProvider,
        LocalETFDataProvider,
        OfficialExchangeETFDataProvider,
        RadarDataError,
        RequestPacer,
        SinaETFDataProvider,
        TencentETFDataProvider,
    )
    from radar.strategy import StrategyConfigError, StrategyRepository
    from radar.universe import UniverseRepository

    config = Config()
    if not _check_disclaimer(config):
        return
    if start.date() >= end.date():
        raise click.ClickException("start 必须早于 end")
    root = resource_root()
    universe = UniverseRepository(root / "config" / "radar_universes").active_on(universe_id, end.date())
    try:
        selected_strategy = StrategyRepository(root / "config" / "radar_strategies").get(strategy)
    except StrategyConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    if selected_strategy.asset_type != universe.asset_type:
        raise click.ClickException("策略与标的池资产类型不一致")
    try:
        benchmark_catalog = load_benchmarks(root / "config" / "radar_benchmarks.yaml")
    except RadarBenchmarkError as exc:
        raise click.ClickException(str(exc)) from exc
    configured_profiles = config.get("radar.cost_profiles", {})
    raw_cost_profile = configured_profiles.get(selected_strategy.cost_profile) if isinstance(configured_profiles, dict) else None
    if not isinstance(raw_cost_profile, dict):
        raw_cost_profile = {"commission_rate": config.get("radar.etf_cost_rate", 0.0005), "slippage_rate": 0.0}
    try:
        commission_rate = float(raw_cost_profile["commission_rate"])
        slippage_rate = float(raw_cost_profile["slippage_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise click.ClickException(f"成本档 {selected_strategy.cost_profile} 配置无效") from exc
    if commission_rate < 0 or slippage_rate < 0:
        raise click.ClickException(f"成本档 {selected_strategy.cost_profile} 的费率不能为负")
    cost_profile = {
        "id": selected_strategy.cost_profile,
        "commission_rate": commission_rate,
        "slippage_rate": slippage_rate,
    }
    benchmark_specs = [item.model_dump() for item in benchmark_catalog.benchmarks]
    if not rerun:
        existing = find_reusable_radar_backtest(
            runtime_root() / "reports",
            universe_id=universe.id,
            strategy_id=strategy,
            strategy_fingerprint=selected_strategy.fingerprint,
            start_date=start.date(),
            end_date=end.date(),
            cost_profile=cost_profile,
            benchmarks=benchmark_specs,
        )
        if existing is not None:
            console.print(f"[green]复用已有回测产物: {existing}[/green]")
            return existing
    provider = FallbackETFDataProvider((
        LocalETFDataProvider(config.config_dir / "radar_local_data" / "etf"),
        AkShareETFDataProvider(pacer=RequestPacer(config.get("radar.minimum_interval_seconds", 1.0))),
        TencentETFDataProvider(),
        SinaETFDataProvider(),
        OfficialExchangeETFDataProvider(),
    ))
    data_start = start.date() - timedelta(days=500)
    try:
        from radar.calendar import TradingCalendar
        from radar.history import HistoryStore, HistorySynchronizer

        calendar = TradingCalendar(config.config_dir / "radar_collector.db")
        days = calendar.trading_days(data_start, end.date())
        if not days:
            raise RadarDataError("回测区间没有交易日")
        target = days[-1]
        cached_calendar = calendar.cached_data()
        all_days = calendar.trading_days(cached_calendar.coverage_start, target) if cached_calendar else days
        synchronizer = HistorySynchronizer(HistoryStore(config.config_dir / "radar.db"), provider)
        histories = {item.symbol: synchronizer.read_or_sync(item.symbol, item.market, data_start, target, trading_days=all_days)[0] for item in universe.instruments}
    except (RadarDataError, RuntimeError) as exc:
        raise click.ClickException(f"回测历史数据不可用: {exc}") from exc
    try:
        benchmarks = {
            item.id: fetch_benchmark_closes(
                item,
                start.date(),
                end.date(),
                local_directory=config.config_dir / "radar_local_data" / "benchmarks",
            )
            for item in benchmark_catalog.benchmarks
        }
    except RadarBenchmarkError as exc:
        raise click.ClickException(f"基准数据不可用: {exc}") from exc
    result = run_monthly_rotation(
        histories,
        {item.symbol: item.category for item in universe.instruments},
        start.date(),
        end.date(),
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        weights=selected_strategy.weights,
        benchmarks=benchmarks,
    )
    def coverage(frame):
        dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
        return {
            "start_date": dates.min().date().isoformat() if not dates.empty else None,
            "end_date": dates.max().date().isoformat() if not dates.empty else None,
            "rows": len(frame),
        }

    data_coverage = {
        "requested_history_start": data_start.isoformat(),
        "formal_start": start.date().isoformat(), "formal_end": end.date().isoformat(),
        "instruments": {symbol: coverage(frame) for symbol, frame in histories.items()},
        "benchmarks": {benchmark_id: {
            "start_date": str(values.index.min()),
            "end_date": str(values.index.max()), "rows": len(values),
        } for benchmark_id, values in benchmarks.items()},
    }
    output = write_radar_backtest_artifacts(
        result,
        universe_id=universe.id,
        universe_version=universe.version,
        strategy_id=strategy,
        strategy_fingerprint=selected_strategy.fingerprint,
        start_date=start.date(),
        end_date=end.date(),
        benchmarks=benchmark_specs,
        cost_rate=commission_rate,
        strategy_snapshot=selected_strategy.model_dump(mode="json"),
        cost_profile=cost_profile,
        data_coverage=data_coverage,
        data_providers={
            "etf": ["AkShareETFDataProvider", "TencentETFDataProvider", "SinaETFDataProvider", "OfficialExchangeETFDataProvider"],
            "benchmarks": ["AkShare CSI 指数日线"],
        },
    )
    console.print(f"[green]回测产物已保存: {output}[/green]")
    return output


@radar.command("backtest-walkforward")
@click.option("--universe", "universe_id", required=True, help="标的池 ID")
@click.option("--start", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--window-years", default=2, show_default=True, type=click.IntRange(min=1), help="固定且不重叠的窗口年数")
@click.option("--strategy", default="core_rotation_v1", show_default=True)
@click.option("--rerun", is_flag=True, help="忽略可复用的单段回测产物")
@click.pass_context
def radar_backtest_walkforward(ctx, universe_id, start, end, window_years, strategy, rerun):
    """按固定多窗口生成 ETF 策略稳健性报告。"""
    from radar.walkforward import (
        fixed_windows,
        summarize_windows,
        write_walkforward_artifacts,
    )

    config = Config()
    if not _check_disclaimer(config):
        return
    try:
        windows = fixed_windows(start.date(), end.date(), window_years)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    outputs: dict[tuple[date, date], Path] = {}
    failures: dict[tuple[date, date], str] = {}
    for window in windows:
        if not window.available:
            continue
        try:
            output = ctx.invoke(
                radar_backtest,
                universe_id=universe_id,
                start=start.replace(year=window.start_date.year, month=window.start_date.month, day=window.start_date.day),
                end=end.replace(year=window.end_date.year, month=window.end_date.month, day=window.end_date.day),
                strategy=strategy,
                rerun=rerun,
            )
            if isinstance(output, Path):
                outputs[(window.start_date, window.end_date)] = output
            else:
                failures[(window.start_date, window.end_date)] = "单段回测未生成产物"
        except click.ClickException as exc:
            failures[(window.start_date, window.end_date)] = exc.message
    summary = summarize_windows(windows, outputs, failures)
    output = write_walkforward_artifacts(
        summary,
        universe_id=universe_id,
        strategy_id=strategy,
        start_date=start.date(),
        end_date=end.date(),
        window_years=window_years,
        reports_dir=runtime_root() / "reports",
    )
    console.print(f"[green]稳健性报告已保存: {output}[/green]")
    return output


if __name__ == "__main__":
    main()
