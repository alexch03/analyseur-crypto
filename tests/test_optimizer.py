from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from app.paper.engine_replay import BacktestReport
from app.services.optimizer import (
    _objective_score,
    optimize_setup_parameters,
    report_optimization_export_metrics,
)


def _minimal_report(**overrides) -> BacktestReport:
    d = {
        "total_trades": 10,
        "wins": 6,
        "losses": 4,
        "win_rate": 0.6,
        "profit_factor": 1.5,
        "expectancy_r": 0.2,
        "net_r": 2.0,
        "max_drawdown_r": 0.5,
        "gross_pnl_quote": 100.0,
        "net_pnl_quote": 80.0,
        "total_fees_quote": 5.0,
        "total_funding_quote": 0.0,
        "realized_gains_quote": 100.0,
        "realized_losses_quote": 20.0,
        "avg_trade_duration_bars": 3.0,
        "avg_time_in_negative_pct": 0.2,
        "max_drawdown_quote": 30.0,
        "trades": [],
    }
    d.update(overrides)
    return BacktestReport(**d)


def _make_df(n: int = 380) -> pd.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    price = 200.0
    for i in range(n):
        wave = 2.5 if i % 30 < 15 else -2.0
        price += wave * 0.2
        o = price - 1.0
        c = price + 1.0
        h = max(o, c) + 1.5
        l = min(o, c) - 1.5
        rows.append(
            {
                "timestamp": start + timedelta(hours=i),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": 2000 + i,
            }
        )
    return pd.DataFrame(rows)


def test_optimizer_returns_ranked_results_by_default_objective():
    df = _make_df()
    res = optimize_setup_parameters(
        df,
        symbol="ETH/USDT",
        timeframe="4h",
        rr_min_values=[1.5, 2.0],
        fvg_proximity_values=[0.003],
        ob_proximity_values=[0.003],
        swing_left_values=[2],
        swing_right_values=[2],
        max_setups_values=[3],
    )
    assert len(res) == 2
    assert res[0].report.net_pnl_quote >= res[1].report.net_pnl_quote
    for r in res:
        assert r.report.realized_gains_quote >= 0
        assert r.report.realized_losses_quote >= 0
        assert abs(r.report.net_pnl_quote - (r.report.realized_gains_quote - r.report.realized_losses_quote)) < 1e-6


def test_optimizer_random_caps_trials():
    df = _make_df()
    res = optimize_setup_parameters(
        df,
        symbol="ETH/USDT",
        timeframe="4h",
        rr_min_values=[1.5, 2.0, 2.5],
        fvg_proximity_values=[0.003, 0.005],
        ob_proximity_values=[0.003],
        swing_left_values=[2, 3],
        swing_right_values=[2],
        max_setups_values=[3, 5],
        strategy="random",
        max_trials=5,
    )
    assert len(res) == 5


def test_optimizer_composite_runs():
    df = _make_df()
    res = optimize_setup_parameters(
        df,
        symbol="ETH/USDT",
        timeframe="4h",
        objective="composite",
        rr_min_values=[1.5, 2.0],
        fvg_proximity_values=[0.003],
        ob_proximity_values=[0.003],
        swing_left_values=[2],
        swing_right_values=[2],
        max_setups_values=[3],
    )
    assert len(res) == 2


def test_optimizer_penalized_pnl_runs():
    df = _make_df()
    res = optimize_setup_parameters(
        df,
        symbol="ETH/USDT",
        timeframe="4h",
        objective="penalized_pnl_quote",
        rr_min_values=[1.5, 2.0],
        fvg_proximity_values=[0.003],
        ob_proximity_values=[0.003],
        swing_left_values=[2],
        swing_right_values=[2],
        max_setups_values=[3],
    )
    assert len(res) == 2


def test_penalized_objective_prefers_lower_drawdown_at_equal_pnl():
    hi_dd = _minimal_report(net_pnl_quote=500.0, max_drawdown_quote=400.0, total_trades=10, profit_factor=1.2)
    lo_dd = _minimal_report(net_pnl_quote=500.0, max_drawdown_quote=50.0, total_trades=10, profit_factor=1.2)
    assert _objective_score(lo_dd, "penalized_pnl_quote") > _objective_score(hi_dd, "penalized_pnl_quote")


def test_penalized_alias_matches_canonical():
    r = _minimal_report()
    assert _objective_score(r, "penalized_net_pnl") == _objective_score(r, "penalized_pnl_quote")


def test_report_optimization_export_metrics_includes_penalized_score():
    r = _minimal_report()
    m = report_optimization_export_metrics(r, objective="penalized_pnl_quote")
    assert m["optimization_objective_used"] == "penalized_pnl_quote"
    assert "penalized_pnl_adjusted_score" in m
    m2 = report_optimization_export_metrics(r, objective="net_pnl_quote")
    assert "penalized_pnl_adjusted_score" not in m2


def test_optimizer_coordinate_descent_returns_fewer_than_full_grid():
    df = _make_df()
    full = optimize_setup_parameters(
        df,
        symbol="ETH/USDT",
        timeframe="4h",
        rr_min_values=[1.5, 2.0],
        fvg_proximity_values=[0.003, 0.005],
        ob_proximity_values=[0.003],
        swing_left_values=[2, 3],
        swing_right_values=[2],
        max_setups_values=[3, 5],
        strategy="exhaustive",
    )
    cd = optimize_setup_parameters(
        df,
        symbol="ETH/USDT",
        timeframe="4h",
        rr_min_values=[1.5, 2.0],
        fvg_proximity_values=[0.003, 0.005],
        ob_proximity_values=[0.003],
        swing_left_values=[2, 3],
        swing_right_values=[2],
        max_setups_values=[3, 5],
        strategy="coordinate_descent",
    )
    assert len(full) == 2 * 2 * 1 * 2 * 1 * 2  # 16
    assert len(cd) < len(full)
    assert len(cd) >= 1
