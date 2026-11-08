from utils.config import Config


class TestConfig:
    def test_default_config_dir_is_under_current_directory(self, monkeypatch, tmp_path):
        """未显式传入目录时使用项目内状态目录。"""
        monkeypatch.chdir(tmp_path)

        assert Config().config_dir == tmp_path / ".stock_robot"

    def test_default_config_has_required_sections(self, tmp_path):
        cfg = Config(config_dir=tmp_path)
        assert "llm" in cfg.data
        assert "data" in cfg.data
        assert cfg.data["llm"]["enabled"] is True
        assert cfg.data["llm"]["provider"] == "openai"

    def test_load_from_yaml_file(self, tmp_path):
        yaml_content = """
llm:
  provider: claude
  model: claude-opus-4-7
  enabled: false
data:
  cache_ttl: 3600
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)
        cfg = Config(config_dir=tmp_path)
        assert cfg.data["llm"]["provider"] == "claude"
        assert cfg.data["llm"]["model"] == "claude-opus-4-7"
        assert cfg.data["data"]["cache_ttl"] == 3600

    def test_get_returns_nested_value(self, tmp_path):
        cfg = Config(config_dir=tmp_path)
        assert cfg.get("llm.provider") == "openai"
        assert cfg.get("nonexistent.key", "fallback") == "fallback"

    def test_set_updates_value_and_persists(self, tmp_path):
        cfg = Config(config_dir=tmp_path)
        cfg.set("llm.provider", "claude")
        assert cfg.get("llm.provider") == "claude"
        cfg2 = Config(config_dir=tmp_path)
        assert cfg2.get("llm.provider") == "claude"

    def test_first_run_creates_default_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        assert not config_file.exists()
        Config(config_dir=tmp_path)
        assert config_file.exists()

    def test_disclaimer_flag_defaults_to_false(self, tmp_path):
        cfg = Config(config_dir=tmp_path)
        assert cfg.data["data"]["disclaimer_accepted"] is False

    def test_default_config_has_base_url(self, tmp_path):
        cfg = Config(config_dir=tmp_path)
        assert cfg.data["llm"]["base_url"] == ""

    def test_default_config_has_api_and_signal_sections(self, tmp_path):
        cfg = Config(config_dir=tmp_path)
        assert cfg.data["api"] == {
            "host": "0.0.0.0",
            "port": 25618,
            "auth": {"enabled": True, "username": "admin", "password": ""},
        }
        assert cfg.data["signal"]["thresholds"] == {"attack": 7, "watch": 4}
        assert set(cfg.data["signal"]["actions"]) == {"attack", "watch", "defend"}
        assert cfg.data["signal"]["actions"]["attack"] == {"action": "可考虑建仓/加仓", "position": "60%-80%"}
        assert cfg.data["signal"]["actions"]["watch"] == {"action": "持有观察，等待明确方向", "position": "30%-50%"}
        assert cfg.data["signal"]["actions"]["defend"] == {"action": "减仓或回避", "position": "0%-20%"}

    def test_default_config_has_backtest_section(self, tmp_path):
        cfg = Config(config_dir=tmp_path)
        assert cfg.get("backtest.default_strategy") == "report_technical"
        assert cfg.get("backtest.default_benchmark") == "money_fund"
        assert cfg.get("backtest.initial_cash") == 100000.0
        assert cfg.get("backtest.cost_profiles.a_share_default") == {
            "commission_rate": 0.0003,
            "minimum_commission": 5.0,
            "stamp_duty_rate": 0.0005,
            "transfer_fee_rate": 0.00001,
            "slippage_rate": 0.001,
        }
        assert cfg.get("backtest.benchmarks.money_fund.symbol") == "H11025"
        assert cfg.get("backtest.benchmarks.csi_300.symbol") == "000300"
        assert cfg.get("backtest.benchmarks.csi_all_bond.symbol") == "H11001"


class TestPushConfig:
    def test_legacy_wecom_credentials_not_loaded(self, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "push:\n  wecom:\n    secret: old-secret\n  email:\n    smtp_user: user@example.com\n",
            encoding="utf-8",
        )
        config = Config(config_dir=tmp_path)
        assert config.get("push.wecom") is None
        assert config.get("push.email.smtp_user") == "user@example.com"

    def test_default_push_section(self, tmp_path):
        cfg = Config(config_dir=tmp_path)
        assert cfg.get("push.enabled") is True
        assert cfg.get("push.max_symbols_per_subscription") == 20
        assert cfg.get("push.email.smtp_host") == "smtp.qq.com"
        assert cfg.get("push.email.smtp_port") == 465
        assert cfg.get("push.email.smtp_user") == ""
        assert cfg.get("push.wecom") is None


def test_output_budget_defaults_to_auto_without_overwriting_existing_integer(tmp_path):
    config = Config(config_dir=tmp_path)
    assert config.get("llm.max_tokens") is None
    config.set("llm.max_tokens", 2000)
    assert Config(config_dir=tmp_path).get("llm.max_tokens") == 2000
    config.set("llm.max_tokens", None)
    assert Config(config_dir=tmp_path).get("llm.max_tokens") is None
