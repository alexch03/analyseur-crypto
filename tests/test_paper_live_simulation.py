"""Tests simulation paper live alignée replay."""

from __future__ import annotations

import pandas as pd

from app.paper.engine_replay import replay_engine_from_bt_cfg
from app.schemas.domain import Side, TradeSetupDTO
from app.services.paper_live_simulation import step_paper_simulation


def _engine():
    return replay_engine_from_bt_cfg(
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


def _top_long() -> TradeSetupDTO:
    return TradeSetupDTO(
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


def test_pending_then_entry_then_tp():
    eng = _engine()
    top = _top_long()
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
    r1 = step_paper_simulation(
        df3,
        symbol="BTC/USDT",
        timeframe="1h",
        tick_wall_utc_iso="2026-01-01T03:00:00+00:00",
        top=top,
        prev_pl={},
        engine=eng,
    )
    assert r1["sim_replay_pending"] is not None
    assert r1["sim_replay_open"] is None

    ts4 = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    df4 = pd.DataFrame(
        {
            "timestamp": ts4,
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 110.0],
            "low": [99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 105.0],
            "volume": [1.0, 1.0, 1.0, 1.0],
        }
    )
    prev = {
        "trade_log": [],
        "sim_cumulative_net_pnl_quote": 0.0,
        "sim_replay_last_bar_ts": r1.get("sim_replay_last_bar_ts"),
        "sim_replay_pending": r1["sim_replay_pending"],
        "sim_replay_open": r1.get("sim_replay_open"),
    }
    r2 = step_paper_simulation(
        df4,
        symbol="BTC/USDT",
        timeframe="1h",
        tick_wall_utc_iso="2026-01-01T04:00:00+00:00",
        top=top,
        prev_pl=prev,
        engine=eng,
    )
    assert r2["sim_replay_pending"] is None
    if r2["sim_replay_open"] is not None:
        assert r2["trade_log"] == []
    else:
        assert len(r2["trade_log"]) >= 1
        assert r2["trade_log"][0]["outcome"] == "TP"

    df5 = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC"),
            "open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 110.0, 115.0],
            "low": [99.0, 99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 105.0, 110.0],
            "volume": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    prev2 = {
        "trade_log": list(r2.get("trade_log") or []),
        "sim_cumulative_net_pnl_quote": float(r2.get("sim_cumulative_net_pnl_quote") or 0),
        "sim_skip_entry_until_bar_ts": r2.get("sim_skip_entry_until_bar_ts"),
        "sim_replay_last_bar_ts": r2.get("sim_replay_last_bar_ts"),
        "sim_replay_pending": r2.get("sim_replay_pending"),
        "sim_replay_open": r2.get("sim_replay_open"),
    }
    r3 = step_paper_simulation(
        df5,
        symbol="BTC/USDT",
        timeframe="1h",
        tick_wall_utc_iso="2026-01-01T05:00:00+00:00",
        top=top,
        prev_pl=prev2,
        engine=eng,
    )
    assert r3["sim_replay_open"] is None
    assert len(r3["trade_log"]) >= 1
    assert r3["trade_log"][0]["outcome"] == "TP"
    assert float(r3["trade_log"][0]["net_pnl_quote"]) > 0
    assert float(r3["sim_cumulative_net_pnl_quote"]) > 0


def test_skip_blocks_new_pending_same_bar():
    eng = _engine()
    top = _top_long()
    ts4 = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    last_key = pd.Timestamp(ts4[-1]).isoformat()
    df4 = pd.DataFrame(
        {
            "timestamp": ts4,
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 110.0],
            "low": [99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 105.0],
            "volume": [1.0, 1.0, 1.0, 1.0],
        }
    )
    r = step_paper_simulation(
        df4,
        symbol="BTC/USDT",
        timeframe="1h",
        tick_wall_utc_iso="2026-01-01T04:00:00+00:00",
        top=top,
        prev_pl={
            "trade_log": [],
            "sim_cumulative_net_pnl_quote": 0.0,
            "sim_replay_last_bar_ts": last_key,
            "sim_skip_entry_until_bar_ts": last_key,
        },
        engine=eng,
    )
    assert r["sim_replay_pending"] is None
    assert r["sim_replay_open"] is None
