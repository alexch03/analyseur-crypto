from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from app.paper.engine_replay import (
    ReplayBacktestEngine,
    _atr_at_bar,
    _resolve_intrabar_long,
    _resolve_intrabar_short,
    replay_engine_from_bt_cfg,
)


def _make_df(n: int = 350) -> pd.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    price = 100.0
    for i in range(n):
        drift = 0.25 if i % 40 < 20 else -0.18
        price += drift
        o = price - 0.6
        c = price + 0.6
        h = max(o, c) + 1.2
        l = min(o, c) - 1.2
        rows.append(
            {
                "timestamp": start + timedelta(hours=i),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": 1000 + i,
            }
        )
    return pd.DataFrame(rows)


def test_backtest_runs_and_returns_metrics():
    df = _make_df()
    report = ReplayBacktestEngine().run_walkforward(df, symbol="BTC/USDT", timeframe="1h")
    assert report.total_trades >= 0
    assert 0.0 <= report.win_rate <= 1.0
    assert report.max_drawdown_r >= 0.0


def test_backtest_empty_when_not_enough_data():
    df = _make_df(50)
    report = ReplayBacktestEngine().run_walkforward(df, symbol="BTC/USDT", timeframe="1h")
    assert report.total_trades == 0
    assert report.net_r == 0.0
    assert report.realized_gains_quote == 0.0
    assert report.realized_losses_quote == 0.0


def test_atr_at_bar_positive():
    df = _make_df(80)
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    atr = _atr_at_bar(h, l, c, 40, period=10)
    assert atr > 0


def test_intrabar_conflict_conservative_sl_on_bullish_bar_long():
    # Convention CONSERVATRICE (worst-case) : quand SL et TP sont tous deux dans
    # la bougie, on resout le cote adverse (SL) en premier, meme si la bougie
    # cloture haussiere. (Avant : biais optimiste 'bullish -> TP'.)
    oc, px = _resolve_intrabar_long(
        lo=99.0,
        hi=110.0,
        o=100.0,
        c=108.0,
        effective_sl=98.0,
        tp=109.0,
        entry=100.0,
        trail_armed=False,
    )
    assert oc == "SL"
    assert px == 98.0


def test_intrabar_gap_open_above_tp_long_allows_tp():
    # Exception gap : si l'open est deja >= TP, le 1er prix tradable est le TP.
    oc, px = _resolve_intrabar_long(
        lo=99.0,
        hi=112.0,
        o=110.0,      # open deja au-dessus du TP=109
        c=108.0,
        effective_sl=98.0,
        tp=109.0,
        entry=100.0,
        trail_armed=False,
    )
    assert oc == "TP"
    assert px == 109.0


def test_intrabar_conflict_prefers_sl_on_bearish_bar_long():
    oc, px = _resolve_intrabar_long(
        lo=97.0,
        hi=105.0,
        o=104.0,
        c=98.0,
        effective_sl=97.5,
        tp=110.0,
        entry=100.0,
        trail_armed=False,
    )
    assert oc == "SL"
    assert px == 97.5


def test_intrabar_short_conservative_sl_on_bearish_bar():
    # Convention CONSERVATRICE pour le SHORT : SL adverse resolu en premier meme
    # sur bougie baissiere. (Avant : biais optimiste 'bearish -> TP short'.)
    oc, px = _resolve_intrabar_short(
        lo=90.0,
        hi=101.0,
        o=100.0,
        c=91.0,
        effective_sl=102.0,
        tp=88.0,
        entry=100.0,
        trail_armed=False,
    )
    assert oc == "SL"
    assert px == 102.0


def test_replay_engine_from_bt_cfg_disables_trail_with_zero():
    eng = replay_engine_from_bt_cfg(
        {
            "warmup_bars": 120,
            "max_holding_bars": 120,
            "replay_trail_atr_mult": 0,
            "replay_trail_after_r": 1.0,
        }
    )
    assert eng.trail_atr_mult is None
