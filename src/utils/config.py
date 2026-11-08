import copy
import hashlib
import os
import sqlite3
import tempfile
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import yaml
from filelock import FileLock

from utils.paths import project_state_dir

DEFAULT_CONFIG = {
    "analysis": {"extra_enabled": False},
    "llm": {
        "provider": "openai",
        "model": "gpt-4o",
        "enabled": True,
        "api_key": "",
        "base_url": "",
        "temperature": 0.3,
        "max_tokens": None,
        "retry_times": 2,
        "timeout_seconds": 60,
    },
    "data": {
        "cache_ttl": {
            "daily": 86400,
            "quarterly": 604800,
            "news": 21600,
        },
        "disclaimer_accepted": False,
    },
    "maintenance": {
        "report_cleanup_enabled": True,
        "sweep_interval_minutes": 360,
    },
    "api": {
        # 默认对外监听；非本机请求由 api.auth 的 Basic Auth 拦截，未配置密码时将拒绝外部访问。
        "host": "0.0.0.0",
        "port": 25618,
        "auth": {
            "enabled": True,
            "username": "admin",
            "password": "",
        },
    },
    "push": {
        "enabled": True,
        "max_symbols_per_subscription": 20,
        "email": {
            "smtp_host": "smtp.qq.com",
            "smtp_port": 465,
            "smtp_user": "",
            "smtp_password": "",
            "to_addr": "",
        },
    },
    "signal": {
        "thresholds": {
            "attack": 7,
            "watch": 4,
        },
        "actions": {
            "attack": {"action": "可考虑建仓/加仓", "position": "60%-80%"},
            "watch": {"action": "持有观察，等待明确方向", "position": "30%-50%"},
            "defend": {"action": "减仓或回避", "position": "0%-20%"},
        },
    },
    "backtest": {
        "default_strategy": "report_technical",
        "default_benchmark": "money_fund",
        "initial_cash": 100000.0,
        "cost_profiles": {
            "a_share_default": {
                "commission_rate": 0.0003,
                "minimum_commission": 5.0,
                "stamp_duty_rate": 0.0005,
                "transfer_fee_rate": 0.00001,
                "slippage_rate": 0.001,
            },
        },
        "benchmarks": {
            "money_fund": {"name": "中证货币型基金指数", "symbol": "H11025"},
            "csi_300": {"name": "沪深300", "symbol": "000300"},
            "csi_all_bond": {"name": "中证全债指数", "symbol": "H11001"},
        },
    },
    "radar": {
        "collector": {"enabled": False, "hour": 18, "minute": 30},
        "history_days": 400,
        "minimum_interval_seconds": 1.0,
        "etf_cost_rate": 0.0005,
        "cost_profiles": {
            "etf_default": {
                "commission_rate": 0.0003,
                "slippage_rate": 0.0002,
            },
        },
    },
}


class ConfigRevisionConflict(ValueError):
    """配置文件已被其他进程修改。"""


class Config:
    def __init__(self, config_dir: Path | None = None, *, collector_db_path: Path | None = None):
        if config_dir is None:
            config_dir = project_state_dir()
        self._config_dir = Path(config_dir)
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._config_path = self._config_dir / "config.yaml"
        self._write_lock = FileLock(str(self._config_path) + ".lock", timeout=30)
        self._collector_db_path = collector_db_path or self._config_dir / "radar_collector.db"
        self.data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        with self.write_lock():
            return self._load_unlocked()

    def _load_unlocked(self) -> dict[str, Any]:
        if not self._config_path.exists():
            self._write_default()
        with open(self._config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
        legacy = self._legacy_collector()
        if legacy and isinstance(user_config.get("radar", {}), dict):
            collector = user_config.get("radar", {}).get("collector", {})
            if isinstance(collector, dict) and set(legacy) - collector.keys():
                with self.write_lock():
                    user_config = yaml.safe_load(self.source) or {}
                    collector = user_config.setdefault("radar", {}).setdefault("collector", {})
                    for key, value in legacy.items():
                        collector.setdefault(key, value)
                    self._atomic_write(yaml.safe_dump(user_config, allow_unicode=True, sort_keys=False))
        if legacy:
            self._mark_collector_migrated()
        merged = self._deep_copy(DEFAULT_CONFIG)
        self._merge(merged, user_config)
        # 旧企微凭据不载入运行时；原文件仍保留未展示字段。
        merged["push"].pop("wecom", None)
        return merged

    def migrate_collector(self, legacy: dict[str, Any]) -> None:
        """只补原文件缺失的采集字段，明确 YAML 值始终优先。"""
        with self.write_lock():
            if not self._legacy_collector():
                self.data = self._load()
                return
            raw = yaml.safe_load(self.source) or {}
            collector = raw.setdefault("radar", {}).setdefault("collector", {})
            missing = {key: legacy[key] for key in ("enabled", "hour", "minute") if key not in collector}
            if missing:
                collector.update(missing)
                self._atomic_write(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False))
            self._mark_collector_migrated()
            self.data = self._load()

    def _write_default(self):
        with self.write_lock():
            if self._config_path.exists():
                return
            data = self._deep_copy(DEFAULT_CONFIG)
            legacy = self._legacy_collector()
            if legacy:
                data["radar"]["collector"] = legacy
            self._atomic_write(yaml.safe_dump(data, allow_unicode=True))
            if legacy:
                self._mark_collector_migrated()

    def _persist(self):
        # 运行时移除的弃用字段仍保留在原文件中，表单保存不删除未展示内容。
        raw = yaml.safe_load(self.source) or {}
        self._merge(raw, self.data)
        self._atomic_write(yaml.safe_dump(raw, allow_unicode=True))

    def _legacy_collector(self) -> dict[str, Any]:
        if not self._collector_db_path.exists():
            return {}
        with closing(sqlite3.connect(self._collector_db_path)) as connection:
            if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='config_migrations'").fetchone() and connection.execute("SELECT 1 FROM config_migrations WHERE name='collector-yaml'").fetchone():
                return {}
            if not connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='collector_settings'").fetchone():
                return {}
            row = connection.execute("SELECT enabled,hour,minute FROM collector_settings WHERE singleton=1").fetchone()
            return {"enabled": bool(row[0]), "hour": row[1], "minute": row[2]} if row else {}

    def _mark_collector_migrated(self) -> None:
        """迁移标记防止用户删除 YAML 字段后旧数据库设置复活。"""
        with closing(sqlite3.connect(self._collector_db_path)) as connection, connection:
            connection.execute("CREATE TABLE IF NOT EXISTS config_migrations(name TEXT PRIMARY KEY)")
            connection.execute("INSERT OR IGNORE INTO config_migrations VALUES('collector-yaml')")

    def _atomic_write(self, source: str) -> None:
        """同目录临时文件经同步后原子替换，失败保留原配置。"""
        descriptor, name = tempfile.mkstemp(dir=self._config_dir, suffix=".yaml.tmp")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
                stream.write(source)
                stream.flush()
                os.fsync(stream.fileno())
            for attempt in range(3):
                try:
                    os.replace(name, self._config_path)
                    break
                except PermissionError:
                    # Windows 上病毒扫描或短暂句柄占用会让原子替换瞬时失败。
                    if attempt == 2:
                        raise
                    time.sleep(0.05)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def write_lock(self):
        """复用项目文件锁，串行化跨进程保存和迁移。"""
        return self._write_lock

    @property
    def source(self) -> str:
        return self._config_path.read_text(encoding="utf-8")

    @property
    def revision(self) -> str:
        return hashlib.sha256(self._config_path.read_bytes()).hexdigest()

    def replace_source(self, source: str, data: dict[str, Any], revision: str) -> str:
        with self.write_lock():
            if revision != self.revision:
                raise ConfigRevisionConflict("配置已更改，请重新加载后再保存")
            self._atomic_write(source)
            merged = self._deep_copy(DEFAULT_CONFIG)
            self._merge(merged, data)
            merged["push"].pop("wecom", None)
            self.data = merged
            return self.revision

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        node = self.data
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node

    def set(self, key: str, value):
        keys = key.split(".")
        update: dict[str, Any] = {}
        target = update
        for part in keys[:-1]:
            target[part] = {}
            target = target[part]
        target[keys[-1]] = value
        self.update(update)

    def update(self, values: dict[str, Any], revision: str | None = None) -> str:
        """深度合并一组配置，并仅执行一次持久化。"""
        with self.write_lock():
            if revision is not None and revision != self.revision:
                raise ConfigRevisionConflict("配置已更改，请重新加载后再保存")
            self.data = self._load()
            before = self._deep_copy(self.data)
            self._merge(self.data, self._deep_copy(values))
            try:
                self._persist()
            except OSError:
                self.data = before
                raise
            return self.revision

    def reload(self) -> None:
        """调用方可在文件锁内读取一致的配置及版本快照。"""
        self.data = self._load()

    def get_llm_config(self) -> dict[str, Any]:
        return {k: v for k, v in self.data["llm"].items() if k != "api_key"}

    @property
    def config_dir(self) -> Path:
        return self._config_dir

    @staticmethod
    def _deep_copy(d: dict) -> dict:
        return copy.deepcopy(d)

    @staticmethod
    def _merge(base: dict, override: dict):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                Config._merge(base[key], value)
            else:
                base[key] = value
