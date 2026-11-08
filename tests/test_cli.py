from datetime import date

import pandas as pd
import pytest
from click.testing import CliRunner

from backtest.models import (
    BacktestRequest,
    BacktestResult,
    BacktestStrategy,
    BenchmarkSpec,
)
from stock_robot.cli import main


def test_hk_industry_refresh_runs_connect_before_hsics(mocker, tmp_path):
    from types import SimpleNamespace

    calls = []
    connect_snapshot = SimpleNamespace(total=666)
    hsics_snapshot = SimpleNamespace(total=666, sector_count=31, industry_count=12,
                                     data_date=date(2026, 10, 23))
    mocker.patch("data.hk_stock_connect.fetch_sse_snapshot", side_effect=lambda: calls.append("fetch") or connect_snapshot)
    mocker.patch("data.hk_stock_connect.publish_snapshot", side_effect=lambda *_: calls.append("publish"))
    mocker.patch("data.hk_hsics_industry.load_connect_symbols", side_effect=lambda: calls.append("load") or {"00700"})
    refresh = mocker.patch("data.hk_hsics_industry.refresh", side_effect=lambda *_, **__: calls.append("refresh") or hsics_snapshot)
    mocker.patch("utils.paths.hk_stock_connect_path", return_value=tmp_path / "connect.json")
    mocker.patch("utils.paths.hk_hsics_industry_path", return_value=tmp_path / "industry.csv")
    mocker.patch("utils.paths.hk_hsics_snapshot_path", return_value=tmp_path / "industry.json")

    result = CliRunner().invoke(main, ["hk-industry-refresh"])

    assert result.exit_code == 0, result.output
    assert calls == ["fetch", "publish", "load", "refresh"]
    assert refresh.call_args.kwargs["connect_symbols"] == {"00700"}
    assert "港股通成员 666 只" in result.output
    assert "HSICS 总数 666，二级 31，一级 12，数据日期 2026-10-23" in result.output
    assert "complete" in result.output and "published=True" in result.output


def test_hk_industry_refresh_reports_hsics_failure_without_rolling_back_connect(mocker, tmp_path):
    from types import SimpleNamespace

    from data.hk_hsics_industry import HsicsIndustryError

    published = []
    mocker.patch("data.hk_stock_connect.fetch_sse_snapshot", return_value=SimpleNamespace(total=666))
    mocker.patch("data.hk_stock_connect.publish_snapshot", side_effect=lambda *_: published.append(True))
    mocker.patch("data.hk_hsics_industry.load_connect_symbols", return_value={"00700"})
    mocker.patch("data.hk_hsics_industry.refresh", side_effect=HsicsIndustryError("上游失败"))
    mocker.patch("utils.paths.hk_stock_connect_path", return_value=tmp_path / "connect.json")
    mocker.patch("utils.paths.hk_hsics_industry_path", return_value=tmp_path / "industry.csv")
    mocker.patch("utils.paths.hk_hsics_snapshot_path", return_value=tmp_path / "industry.json")

    result = CliRunner().invoke(main, ["hk-industry-refresh"])

    assert result.exit_code == 1
    assert published == [True]
    assert "港股通成员快照已更新（666 只）" in result.output
    assert "partial" in result.output and "published=False" in result.output


@pytest.mark.parametrize("path", [[], ["index"], ["radar"], ["config", "set"]])
def test_help_command_matches_native_help(path):
    runner = CliRunner()
    result = runner.invoke(main, ["help", *path])
    expected = runner.invoke(main, [*path, "--help"])
    assert result.exit_code == 0, result.output
    assert result.output == expected.output


@pytest.mark.parametrize("path", [["missing"], ["radar", "missing"], ["index", "missing"]])
def test_help_command_rejects_unknown_or_leaf_paths(path):
    result = CliRunner().invoke(main, ["help", *path])
    assert result.exit_code == 2
    assert "missing" in result.output


def test_help_command_supports_deep_paths_without_running_callbacks(monkeypatch):
    import click

    callbacks = []
    def unexpected_callback(*_args, **_kwargs):
        callbacks.append(True)
    leaf = click.Command("leaf", callback=unexpected_callback,
                         params=[click.Argument(["required"])], help="测试深层帮助")
    nested = click.Group("nested", commands={"leaf": leaf}, callback=unexpected_callback)
    group = click.Group("help-test", commands={"nested": nested}, callback=unexpected_callback)
    monkeypatch.setitem(main.commands, "help-test", group)
    runner = CliRunner()
    result = runner.invoke(main, ["help", "help-test", "nested", "leaf"])
    assert result.exit_code == 0, result.output
    assert callbacks == []
    expected = runner.invoke(main, ["help-test", "nested", "leaf", "--help"])
    assert "REQUIRED" in result.output
    assert result.output == expected.output


def test_empty_invocation_shows_help_hint_without_business_calls(mocker):
    config = mocker.patch("stock_robot.cli.Config")
    pipeline = mocker.patch("stock_robot.cli._build_pipeline")
    result = CliRunner().invoke(main, [], prog_name="stock-robot")
    assert result.exit_code == 0, result.output
    assert "Commands:" in result.output
    assert "stock-robot help <命令>" in result.output
    config.assert_not_called()
    pipeline.assert_not_called()


@pytest.mark.parametrize("arguments,path", [
    (["missing"], ""),
    (["--missing-option"], ""),
    (["index"], "index"),
    (["index", "--missing-option"], "index"),
    (["radar", "missing"], "radar"),
    (["config", "set"], "config set"),
    (["run", "--port", "bad"], "run"),
])
def test_usage_error_points_to_matching_help(arguments, path):
    result = CliRunner().invoke(main, arguments, prog_name="stock-robot")
    assert result.exit_code == 2
    hint = f"运行 stock-robot help{' ' + path if path else ''} 查看详细用法。"
    assert hint in result.stderr
    assert "Error:" in result.stderr
    assert hint not in result.stdout


def test_business_input_error_keeps_reason_and_adds_help(mocker):
    mocker.patch("stock_robot.cli._check_disclaimer", return_value=True)
    result = CliRunner().invoke(main, ["analyze", "abc"])
    assert result.exit_code == 1
    assert "abc" in result.stdout
    assert "运行 stock-robot help analyze 查看详细用法。" in result.stderr


def test_create_cli_progress_uses_shared_transient_columns():
    from stock_robot.cli import _create_cli_progress

    progress = _create_cli_progress()

    assert progress.live.transient is True
    assert len(progress.columns) == 3
    assert all("task.completed" not in str(column) for column in progress.columns)


def test_radar_import_data_writes_local_etf_history(mocker, tmp_path):
    from utils.config import Config

    source = tmp_path / "etf.csv"
    pd.DataFrame({
        "日期": ["2024-01-02"], "开盘": [3.7], "最高": [3.9], "最低": [3.6],
        "收盘": [3.8], "成交量": [100], "成交额": [380],
    }).to_csv(source, index=False, encoding="utf-8-sig")
    config = Config(config_dir=tmp_path / "config")
    mocker.patch("stock_robot.cli.Config", return_value=config)

    result = CliRunner().invoke(main, ["radar", "import-data", "--symbol", "510300", "--file", str(source)])

    assert result.exit_code == 0
    assert "已导入 510300" in result.output
    assert (config.config_dir / "radar_local_data" / "etf" / "510300.csv").exists()


def test_radar_import_benchmark_writes_local_history(mocker, tmp_path):
    from utils.config import Config

    source = tmp_path / "benchmark.csv"
    pd.DataFrame({"日期": ["2024-01-02"], "收盘": [3500.0]}).to_csv(source, index=False, encoding="utf-8-sig")
    config = Config(config_dir=tmp_path / "config")
    mocker.patch("stock_robot.cli.Config", return_value=config)

    result = CliRunner().invoke(main, ["radar", "import-benchmark", "--benchmark", "csi_300", "--file", str(source)])

    assert result.exit_code == 0
    assert "已导入基准 csi_300" in result.output
    assert (config.config_dir / "radar_local_data" / "benchmarks" / "csi_300.csv").exists()


def test_radar_collect_once_prints_each_pool(mocker, tmp_path):
    worker = mocker.patch("radar.collector_service.build_collector").return_value
    worker.enqueue_manual.return_value = [{"id": "task", "universe_id": "cn_hk_etf"}]
    worker.execute_one.return_value = False
    worker.store.get_run.return_value = {"status": "completed", "snapshot_run_id": "run-cn_hk_etf"}
    result = CliRunner().invoke(main, ["radar", "collect", "--once"])
    assert result.exit_code == 0
    assert "run-cn_hk_etf" in result.output


def test_radar_collect_daemon_exits_when_autostart_is_disabled(mocker, tmp_path):
    from utils.config import Config

    config = Config(config_dir=tmp_path / "config")
    (config.config_dir / "radar_collector.disabled").write_text("disabled\n")
    services = mocker.patch("stock_robot.cli._radar_services")
    mocker.patch("stock_robot.cli.Config", return_value=config)

    result = CliRunner().invoke(main, ["radar", "collect"])

    assert result.exit_code == 0
    assert "自动采集已关闭" in result.output
    services.assert_not_called()


class TestCLI:
    def test_analyze_without_symbol_shows_error(self):
        runner = CliRunner()
        result = runner.invoke(main, ["analyze"])
        assert result.exit_code != 0

    def test_analyze_with_invalid_symbol_shows_error(self, mocker):
        mocker.patch("stock_robot.cli._check_disclaimer", return_value=True)
        runner = CliRunner()
        result = runner.invoke(main, ["analyze", "abc"])
        assert result.exit_code != 0

    def test_analyze_with_valid_symbol(self, mocker, tmp_path):
        mocker.patch("stock_robot.cli._check_disclaimer", return_value=True)
        mocker.patch("utils.symbols.resolve_name", return_value="平安银行")
        mock_pipeline = mocker.patch("stock_robot.cli._build_pipeline")
        mock_save = mocker.patch(
            "report.formatter.ReportFormatter.save",
            return_value=tmp_path / "stock" / "000001" / "2026-08" / "000001.md",
        )
        mock_instance = mock_pipeline.return_value
        from data.schemas import AnalysisContext, AnalysisResult
        ctx = AnalysisContext(symbol="000001", name="平安银行")
        def run_with_progress(*args, **kwargs):
            kwargs["on_progress"]("data", 1, 3, "读取")
            return (
                [AnalysisResult(dimension="financial", status="ok", summary="OK", metrics={"roe": 0.12})],
                {"bulk": "综合解读"},
                ctx,
            )

        mock_instance.run.side_effect = run_with_progress

        runner = CliRunner()
        result = runner.invoke(main, ["analyze", "000001", "--no-llm"])
        assert result.exit_code == 0
        assert "查看详细用法" not in result.output
        assert mock_save.call_args.kwargs["category"] == "stock"

    def test_config_set_and_get(self):
        runner = CliRunner()
        result = runner.invoke(main, ["config", "set", "llm.provider", "claude"])
        assert result.exit_code == 0
        result = runner.invoke(main, ["config", "get", "llm.provider"])
        assert result.exit_code == 0
        assert "claude" in result.output

    def test_cache_clear(self, mocker):
        mock_cache = mocker.patch("stock_robot.cli._get_cache")
        runner = CliRunner()
        result = runner.invoke(main, ["cache", "clear"])
        assert result.exit_code == 0
        mock_cache.return_value.clear.assert_called_once()

    def test_analyze_no_llm_flag(self, mocker):
        mocker.patch("stock_robot.cli._check_disclaimer", return_value=True)
        mocker.patch("utils.symbols.resolve_name", return_value="平安银行")
        mock_pipeline = mocker.patch("stock_robot.cli._build_pipeline")
        mock_instance = mock_pipeline.return_value
        from data.schemas import AnalysisContext, AnalysisResult
        ctx = AnalysisContext(symbol="000001", name="平安银行")
        mock_instance.run.return_value = (
            [AnalysisResult(dimension="financial", status="ok", summary="OK", metrics={})],
            {},
            ctx,
        )
        runner = CliRunner()
        result = runner.invoke(main, ["analyze", "000001", "--no-llm"])
        assert result.exit_code == 0


class TestBacktest:
    """回测 CLI 命令测试。"""

    @staticmethod
    def _fake_result() -> BacktestResult:
        """构造可完整通过产物写入的最小回测结果。"""
        idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="trade_date")
        return BacktestResult(
            equity_curve=pd.DataFrame(
                {
                    "策略净值": [1.0, 1.01],
                    "基准净值": [1.0, 1.0],
                    "目标仓位": [0.0, 0.7],
                    "当日信号": ["defend", "attack"],
                },
                index=idx,
            ),
            trades=pd.DataFrame(
                {
                    "signal_date": [date(2024, 1, 2)],
                    "trade_date": [date(2024, 1, 3)],
                    "reason": ["attack"],
                    "direction": ["buy"],
                    "price": [10.0],
                    "quantity": [7000],
                    "fees": [12.0],
                    "notional": [70000.0],
                }
            ),
            metrics={
                "累计收益": 0.01,
                "年化收益": 0.2,
                "最大回撤": -0.01,
                "年化波动率": 0.1,
                "夏普比率": 1.0,
                "交易次数": 1.0,
                "超额收益": 0.01,
            },
            warnings=[],
            request=BacktestRequest(
                symbol="000001",
                start_date=date(2024, 1, 2),
                end_date=date(2024, 12, 31),
                strategy_id="report_technical",
                benchmark_id="money_fund",
                initial_cash=100000.0,
            ),
            strategy=BacktestStrategy(
                id="report_technical",
                name="报告技术信号策略",
                version=1,
                signal_source="technical_score",
                thresholds={"attack": 7, "watch": 4},
                target_positions={"attack": 0.7, "watch": 0.4, "defend": 0.0},
                execution="next_open",
                warmup_days=90,
                cost_profile="a_share_default",
            ),
            benchmark=BenchmarkSpec(id="money_fund", name="中证货币型基金指数", symbol="H11025"),
            costs={
                "commission_rate": 0.0003,
                "minimum_commission": 5.0,
                "stamp_duty_rate": 0.0005,
                "transfer_fee_rate": 0.00001,
                "slippage_rate": 0.001,
            },
            data_start=date(2023, 9, 1),
            data_end=date(2024, 12, 31),
        )

    def test_backtest_cli_forwards_benchmark_override(self, mocker, tmp_path):
        mocker.patch("stock_robot.cli._check_disclaimer", return_value=True)
        fake_result = self._fake_result()
        mock_run = mocker.patch("backtest.runner.BacktestRunner.run", return_value=fake_result)
        mock_write = mocker.patch(
            "backtest.artifacts.write_backtest_artifacts",
            return_value=tmp_path / "reports" / "run_dir",
        )

        result = CliRunner().invoke(
            main,
            [
                "backtest", "000001",
                "--start", "2024-01-01",
                "--end", "2024-12-31",
                "--benchmark", "csi_300",
            ],
        )

        assert result.exit_code == 0
        assert mock_run.call_args.args[0].benchmark_id == "csi_300"
        mock_write.assert_called_once_with(fake_result)

    def test_backtest_cli_invalid_date_exits_without_output(self, mocker, tmp_path):
        mocker.patch("stock_robot.cli._check_disclaimer", return_value=True)

        result = CliRunner().invoke(
            main, ["backtest", "000001", "--start", "bad", "--end", "2024-12-31"]
        )

        assert result.exit_code != 0

    def test_backtest_cli_invalid_symbol_exits(self, mocker):
        mocker.patch("stock_robot.cli._check_disclaimer", return_value=True)

        result = CliRunner().invoke(
            main, ["backtest", "abc", "--start", "2024-01-01", "--end", "2024-12-31"]
        )

        assert result.exit_code != 0

    def test_backtest_cli_unaccepted_disclaimer_returns(self, mocker):
        mocker.patch("stock_robot.cli._check_disclaimer", return_value=False)
        mock_run = mocker.patch("backtest.runner.BacktestRunner.run")

        result = CliRunner().invoke(
            main, ["backtest", "000001", "--start", "2024-01-01", "--end", "2024-12-31"]
        )

        assert result.exit_code == 0
        mock_run.assert_not_called()


def test_run_command_help_is_available_and_api_is_unknown():
    """run 是唯一的 Web 服务启动命令。"""
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "启动 Web API 服务" in result.output
    assert runner.invoke(main, ["api", "--help"]).exit_code != 0


def test_run_builds_server_and_reports_ready_url(mocker):
    """run 显示启动进度并将解析后的地址交给服务运行器。"""
    mock_core = mocker.patch("api.bootstrap.build_agent_core", return_value=object())
    mock_app = mocker.patch("api.app.create_app", return_value=object())
    mock_server = mocker.patch("stock_robot.cli._run_web_server")

    result = CliRunner().invoke(main, ["run", "--port", "8000"])

    assert result.exit_code == 0
    assert "正在启动 HTTP 服务" not in result.output
    assert "Web 服务已监听 0.0.0.0:8000" in result.output
    mock_core.assert_called_once()
    server_args = mock_server.call_args.args
    assert server_args[:3] == (mock_app.return_value, "0.0.0.0", 8000)
    assert server_args[3].name.startswith("runtime-")


def test_run_web_server_adds_runtime_log_handlers(mocker, tmp_path):
    """Uvicorn 日志已经覆盖运行状态，命令层不再输出 spinner。"""
    mock_run = mocker.patch("uvicorn.run")
    status = mocker.patch("stock_robot.cli.console.status")

    from stock_robot.cli import _run_web_server
    log_path = tmp_path / "runtime.log"
    _run_web_server(object(), "127.0.0.1", 8000, log_path)

    status.assert_not_called()
    log_config = mock_run.call_args.kwargs["log_config"]
    assert log_config["handlers"]["runtime_access"]["filename"] == str(log_path)
    assert "runtime_default" in log_config["loggers"]["uvicorn"]["handlers"]
    assert "runtime_access" in log_config["loggers"]["uvicorn.access"]["handlers"]


def test_api_bind_resolution_uses_config_defaults(tmp_path):
    """未传 host/port 时读配置 api.host/api.port（默认对外监听 0.0.0.0:25618）"""
    from stock_robot.cli import _resolve_api_bind
    from utils.config import Config

    cfg = Config(config_dir=tmp_path)
    assert _resolve_api_bind(None, None, cfg) == ("0.0.0.0", 25618)
    assert _resolve_api_bind("127.0.0.1", None, cfg) == ("127.0.0.1", 25618)
    assert _resolve_api_bind(None, 9000, cfg) == ("0.0.0.0", 9000)


def test_api_bind_resolution_uses_configured_values(tmp_path):
    """配置自定义 api.port 后未传参时生效"""
    from stock_robot.cli import _resolve_api_bind
    from utils.config import Config

    cfg = Config(config_dir=tmp_path)
    cfg.set("api.port", 9000)
    cfg.set("api.host", "0.0.0.0")
    assert _resolve_api_bind(None, None, cfg) == ("0.0.0.0", 9000)
    assert _resolve_api_bind(None, 8000, cfg) == ("0.0.0.0", 8000)  # CLI 优先


def test_register_llm_keyless_config_registers_nothing(tmp_path):
    """无 api_key 的默认配置下 _register_llm 不应崩溃、不应注册后端"""
    from core.registry import Registry
    from stock_robot.cli import _register_llm
    from utils.config import Config

    config = Config(config_dir=tmp_path)
    reg = Registry()
    _register_llm(reg, config)  # 不应抛异常
    assert reg.get_llm_backend(config.get("llm.provider", "openai")) is None


class TestSubscribe:
    def test_add_creates_subscription(self, mocker, tmp_path):
        mock_store = mocker.patch("push.store.PushStore")
        instance = mock_store.return_value
        instance.create.return_value = 7
        runner = CliRunner()
        result = runner.invoke(main, [
            "subscribe", "add",
            "--name", "自选池",
            "--symbols", "600519,000300",
            "--channel", "email",
            "--time", "08:30",
        ])
        assert result.exit_code == 0
        created = instance.create.call_args.args[0]
        assert created.name == "自选池"
        assert [s.symbol for s in created.symbols] == ["600519", "000300"]
        assert created.channel == "email"

    def test_add_explicit_kind(self, mocker, tmp_path):
        mock_store = mocker.patch("push.store.PushStore")
        instance = mock_store.return_value
        instance.create.return_value = 8
        runner = CliRunner()
        result = runner.invoke(main, [
            "subscribe", "add",
            "--name", "指数池",
            "--symbols", "000001",
            "--kind", "index",
            "--index-style", "broad",
            "--channel", "email",
            "--time", "08:30",
        ])
        assert result.exit_code == 0
        created = instance.create.call_args.args[0]
        assert created.symbols[0].kind == "index"
        assert created.symbols[0].index_style == "broad"

    def test_add_invalid_channel(self):
        runner = CliRunner()
        result = runner.invoke(main, [
            "subscribe", "add",
            "--name", "t", "--symbols", "600519",
            "--channel", "sms", "--time", "08:30",
        ])
        assert result.exit_code != 0

    def test_list_prints_table(self, mocker, tmp_path):
        mock_store = mocker.patch("push.store.PushStore")
        from push.models import Subscription, SubscriptionSymbol
        mock_store.return_value.list.return_value = [
            Subscription(id=1, name="自选池",
                         symbols=[SubscriptionSymbol(symbol="600519")],
                         channel="email", time="08:30")]
        runner = CliRunner()
        result = runner.invoke(main, ["subscribe", "list"])
        assert result.exit_code == 0
        assert "自选池" in result.output

    def test_remove(self, mocker, tmp_path):
        mock_store = mocker.patch("push.store.PushStore")
        mock_store.return_value.delete.return_value = True
        runner = CliRunner()
        result = runner.invoke(main, ["subscribe", "remove", "--id", "3"])
        assert result.exit_code == 0
        mock_store.return_value.delete.assert_called_once_with(3)

    def test_run_triggers_executor(self, mocker, tmp_path):
        mock_store = mocker.patch("push.store.PushStore")
        mocker.patch("api.bootstrap.build_agent_core")  # 避免真实构建 AgentCore
        from push.models import Subscription, SubscriptionSymbol
        sub = Subscription(id=1, name="自选池",
                           symbols=[SubscriptionSymbol(symbol="600519")],
                           channel="email", time="08:30")
        mock_store.return_value.get.return_value = sub
        mock_executor = mocker.patch("push.executor.PushExecutor")
        mock_executor.return_value.run_subscription.return_value = {
            "total": 1, "ok": 1, "failures": []}
        runner = CliRunner()
        result = runner.invoke(main, ["subscribe", "run", "--id", "1"])
        assert result.exit_code == 0
        mock_executor.return_value.run_subscription.assert_called_once_with(
            sub, mock_store.return_value.queue_run.return_value,
        )
        assert "1/1" in result.output


class TestIndustryMappingCommand:
    @pytest.mark.parametrize("action, function", [("rebuild", "rebuild_all"), ("publish", "publish_candidate")])
    def test_file_error_records_failure_and_preserves_exception(self, mocker, tmp_path, action, function):
        mocker.patch("stock_robot.cli._check_disclaimer", return_value=True)
        error = OSError(f"临时文件失败: {tmp_path}")
        mocker.patch(f"data.industry_mapping_builder.{function}", side_effect=error)
        result = CliRunner().invoke(main, ["industry-mapping", action])
        assert result.exception is error
        assert result.exit_code != 0
        assert "failed" in result.output and "published=False" in result.output

    def test_refresh_record_summary_and_failure(self, mocker):
        from data.collection_state import make_refresh_record
        from data.industry_mapping_builder import IndustryMappingError

        mocker.patch("stock_robot.cli._check_disclaimer", return_value=True)
        publish = mocker.patch("data.industry_mapping_builder.publish_candidate", return_value={
            "stock_count": 1, "valid_classification_rate": 100,
            "refresh_record": make_refresh_record("乐咕申万", "complete", published=True),
        })
        result = CliRunner().invoke(main, ["industry-mapping", "publish"])
        assert result.exit_code == 0
        assert "乐咕申万" in result.output and "complete" in result.output and "published=True" in result.output
        publish.side_effect = IndustryMappingError("验证失败")
        result = CliRunner().invoke(main, ["industry-mapping", "publish"])
        assert result.exit_code == 1
        assert "failed" in result.output and "published=False" in result.output

    def test_rebuild_refresh_record_summary(self, mocker):
        from data.collection_state import make_refresh_record

        mocker.patch("stock_robot.cli._check_disclaimer", return_value=True)
        mocker.patch("data.industry_mapping_builder.rebuild_all", return_value={
            "total_industries": 1, "failed_industries": [], "stock_count": 1,
            "coverage_pct": 100, "valid_classification_rate": 100, "candidate_path": "candidate.csv",
            "refresh_record": make_refresh_record("乐咕申万", "partial", published=False),
        })
        result = CliRunner().invoke(main, ["industry-mapping", "rebuild"])
        assert result.exit_code == 0
        assert "partial" in result.output and "published=False" in result.output

    def test_update_symbol(self, mocker):
        from click.testing import CliRunner

        from stock_robot.cli import main

        mocker.patch("stock_robot.cli._check_disclaimer", return_value=True)
        mocker.patch("data.industry_mapping_builder.update_symbol",
                     return_value={"symbol": "600097", "sw_level1": "农林牧渔",
                                   "sw_level2": "渔业", "style_category": "必选消费",
                                   "action": "updated"})
        result = CliRunner().invoke(main, ["industry-mapping", "600097"])
        assert result.exit_code == 0
        assert "600097" in result.output
        assert "农林牧渔" in result.output

    def test_rebuild_all(self, mocker):
        from click.testing import CliRunner

        from stock_robot.cli import main

        mocker.patch("stock_robot.cli._check_disclaimer", return_value=True)
        mocker.patch("data.industry_mapping_builder.rebuild_all",
                      return_value={"total_industries": 335, "failed_industries": [],
                                    "stock_count": 5534, "coverage_pct": 98.2,
                                    "valid_classification_rate": 99.1,
                                    "candidate_path": "candidate.csv"})
        result = CliRunner().invoke(main, ["industry-mapping"])
        assert result.exit_code == 0
        assert "5534" in result.output
        assert "98.2%" in result.output

    def test_rebuild_forwards_custom_delay_and_resume(self, mocker):
        from click.testing import CliRunner

        from stock_robot.cli import main

        mocker.patch("stock_robot.cli._check_disclaimer", return_value=True)
        rebuild = mocker.patch("data.industry_mapping_builder.rebuild_all", return_value={
            "total_industries": 335, "failed_industries": [], "stock_count": 5534,
            "coverage_pct": 98.2, "valid_classification_rate": 99.1,
            "candidate_path": "candidate.csv",
        })
        result = CliRunner().invoke(main, ["industry-mapping", "rebuild", "--resume", "--delay", "6"])
        assert result.exit_code == 0
        assert rebuild.call_args.kwargs["resume"] is True
        assert rebuild.call_args.kwargs["delay"] == 6.0

    def test_publish_candidate(self, mocker):
        from click.testing import CliRunner

        from stock_robot.cli import main

        mocker.patch("stock_robot.cli._check_disclaimer", return_value=True)
        mocker.patch("data.industry_mapping_builder.publish_candidate",
                     return_value={"stock_count": 5534, "valid_classification_rate": 99.1})
        result = CliRunner().invoke(main, ["industry-mapping", "publish"])
        assert result.exit_code == 0
        assert "候选映射已发布" in result.output

    def test_migrate_placeholders(self, mocker):
        from click.testing import CliRunner

        from stock_robot.cli import main

        mocker.patch("stock_robot.cli._check_disclaimer", return_value=True)
        migrate = mocker.patch("data.industry_mapping_builder.migrate_placeholder_rows",
                               return_value={"stock_count": 5534, "migrated_count": 5512})
        result = CliRunner().invoke(main, ["industry-mapping", "migrate-placeholders"])
        assert result.exit_code == 0
        assert "5512" in result.output
        migrate.assert_called_once()

    def test_validate_sample(self, mocker):
        from click.testing import CliRunner

        from stock_robot.cli import main

        mocker.patch("stock_robot.cli._check_disclaimer", return_value=True)
        mocker.patch("data.industry_mapping_builder.validate_sample",
                     return_value={"stock_count": 21, "valid_classification_rate": 100.0})
        result = CliRunner().invoke(main, ["industry-mapping", "validate"])
        assert result.exit_code == 0
        assert "抽样校验通过" in result.output

    def test_invalid_symbol(self, mocker):
        from click.testing import CliRunner

        from stock_robot.cli import main

        mocker.patch("stock_robot.cli._check_disclaimer", return_value=True)
        result = CliRunner().invoke(main, ["industry-mapping", "abc"])
        assert result.exit_code == 1
        assert "无效" in result.output
