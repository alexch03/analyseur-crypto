"""Walk-forward OOS: train segments then test on next slice."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from app.services.walk_forward_out_of_sample import run_walk_forward_oos


def _make_df(n: int = 2000) -> pd.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    price = 100.0
    for i in range(n):
        price += 0.05 if i % 50 < 25 else -0.04
        o = price - 0.3
        c = price + 0.3
        h = max(o, c) + 0.5
        l = min(o, c) - 0.5
        rows.append(
            {
                "timestamp": start + timedelta(hours=i),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": 1000.0,
            }
        )
    return pd.DataFrame(rows)


def test_walk_forward_oos_returns_splits():
    df = _make_df(2000)
    bt = {
        "warmup_bars": 120,
        "max_holding_bars": 80,
        "max_setups_per_bar": 1,
        "unit_size": 1.0,
        "entry_fee_rate": 0.0004,
        "exit_fee_rate": 0.0004,
        "funding_rate_8h": 0.0,
    }
    out = run_walk_forward_oos(
        df,
        symbol="BTC/USDT",
        timeframe="1h",
        objective="net_pnl_quote",
        backtest_config=bt,
        rr_min_values=[2.0],
        fvg_proximity_values=[0.005],
        ob_proximity_values=[0.005],
        swing_left_values=[2],
        swing_right_values=[2],
        max_setups_values=[5],
        n_splits=2,
    )
    assert out["ok"] is True
    assert len(out["splits"]) >= 1
    assert "in_sample" in out["splits"][0]
    assert "out_of_sample" in out["splits"][0]
    assert "oos_summary" in out
