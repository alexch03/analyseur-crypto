"""Tests routeur d'exécution paper (backends + métadonnées export Bitget)."""

from __future__ import annotations

import pandas as pd

from app.paper.engine_replay import replay_engine_from_bt_cfg
from app.schemas.domain import Side, TradeSetupDTO
from app.services import paper_execution


def test_resolve_backend_unknown_defaults():
    assert paper_execution.resolve_paper_execution_backend({}) == "sim_replay"
    assert paper_execution.resolve_paper_execution_backend({"paper_execution_backend": "bitget_futures_sim"}) == "bitget_futures_sim"
    assert paper_execution.resolve_paper_execution_backend({"paper_execution_backend": "unknown_x"}) == "sim_replay"


def test_resolve_ohlcv_exchange_fallback(monkeypatch):
    monkeypatch.setattr(paper_execution.settings, "exchange_id", "binance")
    assert paper_execution.resolve_paper_ohlcv_exchange_id({}) == "binance"
    assert paper_execution.resolve_paper_ohlcv_exchange_id({"paper_ohlcv_exchange_id": "bitget"}) == "bitget"


def test_run_paper_execution_step_bitget_hook_metadata():
    eng = replay_engine_from_bt_cfg(
        {
            "warmup_bars": 2,
            "max_holding_bars": 80,
            "max_setups_per_bar": 1,
            "unit_size": 1.0,
            "entry_fee_rate": 0.0004,
            "exit_fee_rate": 0.0004,
            "funding_rate_8h": 0.0,
        }
    )
    top = TradeSetupDTO(
        symbol="BTC/USDT",
        timeframe="1h",
        side=Side.LONG,
        entry=100.0,
        stop_loss=95.0,
        take_profits=[108.0],
        risk_reward=2.0,
        confidence=0.8,
        setup_type="TEST",
        timestamp=pd.Timestamp("2026-01-01T02:00:00+00:00"),
    )
    ts3 = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    df3 = pd.DataFrame(
        {
            "timestamp": ts3,
            "open": [100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0],
            "volume": [1.0, 1.0, 1.0],
        }
    )
    eff = {"paper_execution_backend": "bitget_futures_sim", "paper_ohlcv_exchange_id": "bitget"}
    patch = paper_execution.run_paper_execution_step(
        df3,
        symbol="BTC/USDT",
        timeframe="1h",
        tick_wall_utc_iso="2026-01-01T03:00:00+00:00",
        top=top,
        prev_pl={},
        engine=eng,
        eff=eff,
    )
    assert patch.get("paper_execution_backend_resolved") == "bitget_futures_sim"
    assert patch.get("paper_ohlcv_exchange_id_resolved") == "bitget"
    assert patch.get("sim_broker_plan_fr") and "Bitget" in patch["sim_broker_plan_fr"]
