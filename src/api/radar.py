"""配置雷达的快照读取与手动刷新 API。"""

from __future__ import annotations

import json
import logging
import sqlite3
import subprocess
import sys
import threading
import time as clock
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from api.auth import is_trusted_client
from api.report_library import ReportLibraryError, get_report_detail, list_reports
from radar.backtest_cache import BacktestCacheError, derive_backtest_window
from radar.benchmarks import (
    RadarBenchmarkError,
    fetch_benchmark_closes,
    load_benchmarks,
)
from radar.calendar import CalendarUnavailable, TradingCalendar
from radar.collector_startup import CollectorStartup, StartupError
from radar.collector_store import CollectorRevisionConflict, CollectorStore
from radar.collector_worker import CollectorWorker
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
from radar.history import HistoryStore, HistorySynchronizer
from radar.overseas import OverseasService
from radar.performance import calculate_instrument_performance
from radar.refresh import RadarRefresher
from radar.score_profile import ScoreProfileConfigError, ScoreProfileRepository
from radar.store import RadarStore
from radar.universe import UniverseConfigError, UniverseRepository
from radar.update_service import UpdateService, UpdateStore
from utils.config import Config
from utils.paths import resource_root, runtime_root


class RefreshRequest(BaseModel):
    """启动一次指定池刷新的请求体。"""

    universe_id: str
    full: bool = Field(default=False, strict=True)
    as_of: date | None = None


class BacktestRequest(BaseModel):
    """启动一段 ETF 池回测的请求体。"""

    universe_id: str
    start_date: date
    end_date: date
    strategy: str = "core_rotation_v1"


class CollectorAutostartRequest(BaseModel):
    """更新本机采集守护启动项的请求体。"""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = Field(strict=True)


class CollectorConfigRequest(BaseModel):
    """修订号防止两个页面相互覆盖，拒绝字符串和布尔冒充时间整数。"""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = Field(strict=True)
    hour: int = Field(strict=True, ge=0, le=23)
    minute: int = Field(strict=True, ge=0, le=59)
    revision: str = Field(strict=True, min_length=1)


class CollectorRunRequest(BaseModel):
    """人工请求可选择一个池，留空按最近目标日的启用池入队。"""

    model_config = ConfigDict(extra="forbid")
    universe_id: str | None = Field(default=None, min_length=1)
    as_of: date | None = None


_STARTUP_HEARTBEAT_WAIT_SECONDS = 20.0
_RESEARCH_NOTICE = "研究评分，不构成投资建议。"
logger = logging.getLogger(__name__)


def _collector_now() -> datetime:
    """控制接口统一按北京时间读取已完成交易日。"""
    return datetime.now(ZoneInfo("Asia/Shanghai"))


def _collector_runtime(store: CollectorStore, now: datetime | None = None) -> dict[str, Any]:
    """心跳是在线事实；旧守护无实例身份和异常时间戳均不能表示新服务在线。"""
    runtime = store.runtime()
    online = False
    try:
        heartbeat = datetime.fromisoformat(str(runtime["heartbeat_at"]))
        if heartbeat.tzinfo is not None and runtime.get("worker_id"):
            age = ((now or _collector_now()) - heartbeat).total_seconds()
            online = 0 <= age < 60
    except (KeyError, ValueError, TypeError):
        online = False
    return {**runtime, "status": "online" if online else "never" if runtime.get("status") == "never" else "offline", "online": online,
            "phase": runtime.get("phase") if online else "offline", "last_error": runtime.get("last_error")}


def _require_collector_online(store: CollectorStore) -> None:
    """离线时不无限堆积人工请求，返回安装和启动入口。"""
    if not _collector_runtime(store)["online"]:
        raise HTTPException(status_code=503, detail={
            "message": "采集服务未在线，请启动独立采集服务后重试。Windows 可启用登录启动；Linux 请启动部署的 systemd 服务。",
            "startup_endpoint": "/api/v1/radar/collector/startup",
            "command": "python -m stock_robot.cli radar daemon",
        })


def create_radar_router() -> APIRouter:
    """查询只读本地数据；生命周期拥有一个人工更新线程。"""
    manual_service: UpdateService | None = None

    @asynccontextmanager
    async def lifespan(_app):
        service = update_services()
        service.start()
        try:
            yield
        finally:
            service.stop()

    router = APIRouter(prefix="/api/v1/radar", tags=["radar"], lifespan=lifespan)
    tasks: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()

    def services() -> tuple[UniverseRepository, RadarStore, RadarRefresher]:
        config = Config()
        root = resource_root()
        repository = UniverseRepository(root / "config" / "radar_universes")
        store = RadarStore(config.config_dir / "radar.db")
        provider = FallbackETFDataProvider((
            LocalETFDataProvider(config.config_dir / "radar_local_data" / "etf"),
            AkShareETFDataProvider(pacer=RequestPacer(config.get("radar.minimum_interval_seconds", 1.0))),
            TencentETFDataProvider(),
            SinaETFDataProvider(),
            OfficialExchangeETFDataProvider(),
        ))
        return repository, store, RadarRefresher(repository, provider, store)

    def collector_services(*, refresh_calendar: bool = False) -> CollectorWorker:
        """复用本地依赖，日历实例严格只读，网页不发起行情采集。"""
        config = Config()
        repository, _, refresher = services()
        status_store = CollectorStore(config.config_dir / "radar_collector.db")
        calendar = TradingCalendar(status_store.db_path, refresh_enabled=refresh_calendar)
        return CollectorWorker(status_store, repository, calendar, refresher.refresh, state_dir=config.config_dir)

    def update_services() -> UpdateService:
        nonlocal manual_service
        with lock:
            if manual_service is None:
                worker = collector_services(refresh_calendar=True)
                manual_service = UpdateService(worker, RadarStore(worker.state_dir / "radar.db"),
                                               OverseasService(worker.state_dir / "radar.db"))
            return manual_service

    def settings_payload(status_store: CollectorStore) -> dict[str, Any]:
        return {**status_store.settings(), "timezone": "Asia/Shanghai"}

    def enqueue_manual(body: CollectorRunRequest) -> dict[str, Any]:
        service = update_services()
        try:
            identifiers = [body.universe_id] if body.universe_id else [pool.id for pool in service.worker.repository.enabled_on(body.as_of or _collector_now().date())]
            runs = [service.enqueue(identifier, body.as_of) for identifier in identifiers]
        except (ValueError, UniverseConfigError, ScoreProfileConfigError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not runs:
            raise HTTPException(status_code=422, detail="目标日期没有启用的标的池")
        return {**runs[0], "runs": runs}

    @router.get("/updates/latest")
    def latest_update(universe_id: str):
        return UpdateStore(Config().config_dir / "radar_collector.db").latest(universe_id)

    @router.get("/overseas")
    def overseas_status(universe_id: str, snapshot_run_id: str | None = None):
        repository, store, _ = services()
        snapshot = store.get_snapshot(snapshot_run_id) if snapshot_run_id else store.latest_completed(universe_id)
        if snapshot_run_id and (snapshot is None or snapshot["universe_id"] != universe_id):
            raise HTTPException(status_code=404, detail="指定池的ETF快照不存在")
        try:
            universe = repository.active_on(universe_id, _collector_now().date())
            return OverseasService(Config().config_dir / "radar.db").view(universe, snapshot, _collector_now())
        except (ValueError, UniverseConfigError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def snapshot_status_summary(snapshot: dict[str, Any]) -> dict[str, int]:
        """从指定快照计算状态，避免历史快照误用最新一次刷新的汇总。"""
        counts = {"fresh": 0, "stale": 0, "failed": 0}
        for item in cast(list[dict[str, Any]], snapshot["items"]):
            status = item.get("status")
            if status in counts:
                counts[status] += 1
        return counts

    def snapshot_payload(store: RadarStore, snapshot: dict[str, Any]) -> dict[str, Any]:
        """补齐快照的研究提示与最近一次未发布刷新信息。"""
        return {
            **snapshot,
            "research_notice": _RESEARCH_NOTICE,
            "status_summary": snapshot_status_summary(snapshot),
            "last_refresh_failure": store.latest_failure_since(
                snapshot["universe_id"], snapshot.get("completed_at"),
            ),
        }

    def backtest_payload(root: Path, report: Any) -> dict[str, Any]:
        """补充池级产物的成本与警告，供前端明确标注其策略属性。"""
        try:
            detail = get_report_detail(root, report.id, equity_curve_max_rows=None)
            manifest_path = root / Path(report.path).parent / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ReportLibraryError) as exc:
            raise HTTPException(status_code=422, detail="回测产物已损坏或不完整") from exc
        return {
            **detail.to_dict(),
            "research_notice": _RESEARCH_NOTICE,
            "cost_profile": manifest.get("cost_profile", {}),
            "warnings": detail.summary.get("warnings", []) if detail.summary else [],
        }

    @router.get("/universes")
    def list_universes():
        repository, _, _ = services()
        return [item.model_dump(mode="json") for item in repository.load_all()]

    @router.get("/collector/config")
    def collector_config():
        return settings_payload(CollectorStore(Config().config_dir / "radar_collector.db"))

    @router.put("/collector/config")
    def update_collector_config(body: CollectorConfigRequest):
        status_store = CollectorStore(Config().config_dir / "radar_collector.db")
        try:
            status_store.update_settings(**body.model_dump())
        except CollectorRevisionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return settings_payload(status_store)

    @router.get("/collector/status")
    def collector_status():
        """读取真实心跳、持久任务和缓存日历，不按工作日猜下一次时间。"""
        try:
            worker = collector_services()
            settings = settings_payload(worker.store)
            config_error = None
            now = _collector_now()
            try:
                target_date = worker.calendar.latest_completed(now)
            except CalendarUnavailable:
                # 无缓存时仅展示当日生效范围，计划错误在下方单独返回。
                target_date = now.date()
            try:
                universe_ids = tuple(item.id for item in worker.repository.enabled_on(target_date))
            except UniverseConfigError as exc:
                config_error = str(exc)
                universe_ids = ()
            runtime = _collector_runtime(worker.store)
            next_scheduled_at = None
            next_overseas_check_at = None
            schedule_error = None
            if settings["enabled"]:
                overseas_check = now.replace(hour=settings["hour"], minute=settings["minute"], second=0, microsecond=0)
                if overseas_check <= now:
                    overseas_check += timedelta(days=1)
                next_overseas_check_at = overseas_check.isoformat()
                try:
                    next_scheduled_at = worker.calendar.next_scheduled(_collector_now(), settings["hour"], settings["minute"]).isoformat()
                except CalendarUnavailable as exc:
                    schedule_error = str(exc)
            return {
                "settings": settings, "schedule": settings,
                "items": worker.store.latest(universe_ids), "runtime": runtime,
                "service_online": runtime["online"], "recent_events": worker.store.recent_events(),
                "runs": worker.store.list_runs(), "next_scheduled_at": next_scheduled_at,
                "next_overseas_check_at": next_overseas_check_at,
                "manual_update_available": True,
                "schedule_error": schedule_error, "config_error": config_error,
            }
        except (OSError, sqlite3.Error) as exc:
            raise HTTPException(status_code=503, detail="采集状态暂不可读取") from exc

    @router.get("/collector/startup")
    @router.get("/collector/autostart")
    def collector_startup_status():
        """系统注册与服务在线分别返回，Linux 网页只查询。"""
        runtime = _collector_runtime(CollectorStore(Config().config_dir / "radar_collector.db"))
        return {**CollectorStartup(runtime_root()).status(), "service_online": runtime["online"], "runtime": runtime}

    @router.put("/collector/startup")
    @router.put("/collector/autostart")
    def update_collector_startup(body: CollectorAutostartRequest, request: Request):
        if not is_trusted_client(request):
            raise HTTPException(status_code=403, detail="仅允许本机或已登录的同源页面管理采集启动项")
        adapter = CollectorStartup(runtime_root())
        if adapter.platform != "win32":
            raise HTTPException(status_code=403, detail="服务器启动设置由部署命令管理，网页只读")
        status_store = CollectorStore(Config().config_dir / "radar_collector.db")
        previous = status_store.runtime()
        try:
            if body.enabled:
                migration = adapter.migrate_legacy(state_dir=status_store.db_path.parent)
                if migration.get("unsafe_overlap"):
                    message = "旧采集入口尚未安全停止，请处理迁移错误后再启用登录启动。"
                    status_store.record_event("error", message)
                    raise HTTPException(status_code=409, detail={"message": message, "migration": migration})
                if migration.get("cleanup_pending"):
                    status_store.record_event("warning", f"旧采集启动项清理待完成：{migration.get('errors', [])}")
            startup = adapter.set_enabled(body.enabled)
        except StartupError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if body.enabled:
            deadline = clock.monotonic() + _STARTUP_HEARTBEAT_WAIT_SECONDS
            while True:
                runtime = _collector_runtime(status_store)
                changed = runtime.get("heartbeat_at") != previous.get("heartbeat_at") or runtime.get("worker_id") != previous.get("worker_id")
                if runtime["online"] and changed:
                    return {**startup, "service_online": True, "runtime": runtime}
                if clock.monotonic() >= deadline:
                    raise HTTPException(status_code=503, detail={"message": "登录启动已配置，但尚未收到采集服务的新鲜心跳，请查看服务日志。", "startup": startup, "service_online": False, "runtime": runtime})
                clock.sleep(0.2)
        runtime = _collector_runtime(status_store)
        return {**startup, "service_online": runtime["online"], "runtime": runtime}

    @router.post("/collector/runs", status_code=202)
    def create_collector_run(body: CollectorRunRequest):
        return enqueue_manual(body)

    @router.get("/collector/runs")
    def list_collector_runs(limit: int = Query(default=20, ge=1, le=200)):
        path = Config().config_dir / "radar_collector.db"
        runs = UpdateStore(path).list_runs(limit) + CollectorStore(path).list_runs(limit)
        return sorted(runs, key=lambda run: run["created_at"], reverse=True)[:limit]

    @router.get("/collector/runs/{run_id}")
    def get_collector_run(run_id: str):
        try:
            return UpdateStore(Config().config_dir / "radar_collector.db").get(run_id)
        except ValueError:
            # 旧ETF任务继续沿用原查询路径。
            logger.debug("组合记录中未找到 %s，查询旧ETF任务", run_id)
        try:
            return CollectorStore(Config().config_dir / "radar_collector.db").get_run(run_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/collector/runs/{run_id}/retry", status_code=202)
    def retry_collector_run(run_id: str):
        status_store = CollectorStore(Config().config_dir / "radar_collector.db")
        try:
            update = UpdateStore(status_store.db_path).get(run_id)
        except ValueError:
            update = None
        if update is not None:
            target = date.fromisoformat(update["target_date"]) if update["target_date"] else None
            return update_services().enqueue(update["universe_id"], target)
        try:
            run = status_store.get_run(run_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if run["status"] in {"queued", "running", "retry_wait"}:
            return status_store.promote_manual(run_id) if run["source"] != "manual" else run
        try:
            return status_store.retry(run_id, manual=True)
        except ValueError as exc:
            # 两个重试请求并发时，先入队者的产物也是后到请求的结果。
            current = status_store.get_run(run_id)
            if current["status"] in {"queued", "running", "retry_wait"}:
                return current
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/score-profiles/{profile_id}")
    def get_score_profile(profile_id: str):
        """暴露版本化评分依据，供 Web 详情页解释当前分数。"""
        root = resource_root()
        try:
            profile = ScoreProfileRepository(root / "config" / "radar_score_profiles").get(profile_id)
        except ScoreProfileConfigError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return profile.model_dump(mode="json")

    @router.get("/snapshots/latest")
    def latest_snapshot(universe_id: str):
        _, store, _ = services()
        snapshot = store.latest_completed(universe_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="尚无完成快照")
        return snapshot_payload(store, snapshot)

    @router.get("/snapshots/{run_id}")
    def get_snapshot(run_id: str):
        _, store, _ = services()
        snapshot = store.get_snapshot(run_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="快照不存在或尚未完成")
        return snapshot_payload(store, snapshot)

    @router.get("/backtests/latest")
    def latest_backtest(universe_id: str):
        """读取指定标的池最近一次完整回测产物，作为“成立以来”的缓存源。"""
        root = runtime_root() / "reports"
        reports = list_reports(root, report_type="backtest", query=universe_id)
        radar_reports = [
            report for report in reports
            if report.path.startswith("radar_backtests/") and report.symbol == universe_id
        ]
        if not radar_reports:
            raise HTTPException(status_code=404, detail="尚无该标的池的回测产物")
        try:
            return derive_backtest_window(backtest_payload(root, radar_reports[0]), None)
        except BacktestCacheError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.get("/backtests")
    def get_backtest(universe_id: str, end_date: date, start_date: date | None = None):
        """从覆盖所选区间的完整产物派生窗口，不重复运行策略。"""
        root = runtime_root() / "reports"
        reports = list_reports(root, report_type="backtest", query=universe_id)
        matching = [
            report for report in reports
            if report.path.startswith("radar_backtests/")
            and report.symbol == universe_id
            and report.end_date == end_date.isoformat()
            and (start_date is None or (report.start_date is not None and report.start_date <= start_date.isoformat()))
        ]
        if not matching:
            raise HTTPException(status_code=404, detail="尚无覆盖该区间的回测产物")
        matching.sort(key=lambda report: report.start_date or "9999-12-31")
        for report in matching:
            try:
                result = derive_backtest_window(backtest_payload(root, report), start_date)
            except BacktestCacheError:
                continue
            result["selection"] = {
                "requested_start_date": start_date.isoformat() if start_date else None,
                "source_start_date": report.start_date,
                "source_end_date": report.end_date,
            }
            return result
        raise HTTPException(status_code=404, detail="尚无覆盖该区间的有效回测缓存")

    @router.get("/performance")
    def instrument_performance(universe_id: str, symbol: str, end_date: date, start_date: date | None = None):
        """读取当前 ETF 自身历史表现；不运行或复用池级轮动策略。"""
        if start_date is not None and start_date >= end_date:
            raise HTTPException(status_code=422, detail="表现开始日必须早于结束日")
        repository, market_store, _ = services()
        try:
            universe = repository.active_on(universe_id, end_date)
        except UniverseConfigError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        instrument = next((item for item in universe.instruments if item.symbol == symbol), None)
        if instrument is None:
            raise HTTPException(status_code=404, detail="标的不属于指定 ETF 池")
        effective_start = start_date or date(2005, 1, 1)
        config = Config()
        provider = FallbackETFDataProvider((
            LocalETFDataProvider(config.config_dir / "radar_local_data" / "etf"),
            AkShareETFDataProvider(pacer=RequestPacer(config.get("radar.minimum_interval_seconds", 1.0))),
            TencentETFDataProvider(), SinaETFDataProvider(), OfficialExchangeETFDataProvider(),
        ))
        root = resource_root()
        try:
            calendar = TradingCalendar(config.config_dir / "radar_collector.db", refresh_enabled=False)
            metadata = calendar.cached_data()
            if metadata is None:
                raise RadarDataError("交易日历缓存不可用，请启动采集服务初始化日历")
            known_days = calendar.trading_days(metadata.coverage_start, end_date)
            if not known_days or known_days[-1] < effective_start:
                raise RadarDataError("请求区间没有已验证交易日")
            target_date = known_days[-1]
            history, _ = HistorySynchronizer(HistoryStore(market_store.db_path), provider).read_or_sync(symbol, instrument.market, effective_start, target_date, trading_days=known_days)
            actual_start = max(effective_start, history.iloc[0]["date"])
            catalog = load_benchmarks(root / "config" / "radar_benchmarks.yaml")
            benchmarks = {
                item.id: fetch_benchmark_closes(
                    item,
                    actual_start,
                    target_date,
                    local_directory=config.config_dir / "radar_local_data" / "benchmarks",
                )
                for item in catalog.benchmarks
            }
            result = calculate_instrument_performance(history, benchmarks)
        except (RadarDataError, RadarBenchmarkError, CalendarUnavailable, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"标的历史表现不可用: {exc}") from exc
        return {
            "symbol": symbol,
            "start_date": str(result.equity_curve.index.min()),
            "end_date": str(result.equity_curve.index.max()),
            "metrics": result.metrics,
            "benchmarks": result.benchmark_metrics,
            "equity_curve": {
                "columns": ["date", *result.equity_curve.columns],
                "rows": [{"date": pd.Timestamp(cast(Any, index)).date().isoformat(), **{key: float(value) for key, value in row.items()}} for index, row in result.equity_curve.iterrows()],
            },
            "research_notice": _RESEARCH_NOTICE,
        }

    @router.post("/backtests", status_code=202)
    def start_backtest(body: BacktestRequest):
        """后台运行指定区间回测；相同完整产物由 CLI 自行复用。"""
        if body.start_date >= body.end_date:
            raise HTTPException(status_code=422, detail="回测开始日必须早于结束日")
        with lock:
            if any(item["status"] == "running" for item in tasks.values()):
                raise HTTPException(status_code=409, detail="已有雷达任务在运行")
            task_id = uuid.uuid4().hex
            tasks[task_id] = {"status": "running", "universe_id": body.universe_id}

        def run() -> None:
            root = runtime_root()
            command = [
                sys.executable, "-m", "stock_robot.cli", "radar", "backtest",
                "--universe", body.universe_id, "--start", body.start_date.isoformat(),
                "--end", body.end_date.isoformat(), "--strategy", body.strategy,
            ]
            try:
                completed = subprocess.run(
                    command, cwd=root, capture_output=True, text=True, timeout=600, check=False,
                )
                if completed.returncode:
                    detail = completed.stderr.strip() or completed.stdout.strip() or "回测执行失败"
                    tasks[task_id] = {"status": "failed", "universe_id": body.universe_id, "error": detail}
                else:
                    tasks[task_id] = {"status": "completed", "universe_id": body.universe_id}
            except (OSError, subprocess.TimeoutExpired) as exc:
                tasks[task_id] = {"status": "failed", "universe_id": body.universe_id, "error": str(exc)}

        threading.Thread(target=run, daemon=True).start()
        return {"task_id": task_id, "status": "running"}

    @router.get("/backtests/tasks/{task_id}")
    def backtest_status(task_id: str):
        task = tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="回测任务不存在")
        return task

    @router.post("/refresh", status_code=202)
    def refresh(body: RefreshRequest):
        """立即保存组合请求，人工更新不要求独立服务在线。"""
        if body.full:
            raise HTTPException(status_code=422, detail="持久采集按所需窗口补齐，暂不支持旧 full 参数；历史表现查询可按所需区间扩展行情")
        return enqueue_manual(CollectorRunRequest(universe_id=body.universe_id, as_of=body.as_of))

    @router.get("/refresh/{task_id}")
    def refresh_status(task_id: str):
        try:
            return UpdateStore(Config().config_dir / "radar_collector.db").get(task_id)
        except ValueError:
            # 兼容升级前的持久ETF任务ID。
            logger.debug("组合记录中未找到 %s，查询旧ETF任务", task_id)
        try:
            run = CollectorStore(Config().config_dir / "radar_collector.db").get_run(task_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="刷新任务不存在") from exc
        return {**run, "task_id": run["id"], "run_id": run["snapshot_run_id"], "error": run["error_summary"]}

    return router
