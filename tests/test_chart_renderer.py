"""Graphiques SMC : PNG valide avec zoom / mode compact."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from app.chart.renderer import render_chart
from app.services.analysis_pipeline import build_context_and_setups


def _ohlcv_df(n: int = 80) -> pd.DataFrame:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows = []
    base = 100.0
    for i in range(n):
        o = base + i * 0.15 + (i % 5) * 0.1
        c = o + 0.2
        h = max(o, c) + 0.5
        l = min(o, c) - 0.5
        rows.append(
            {
                "timestamp": start + timedelta(hours=i),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": 1000.0 + i,
            }
        )
    return pd.DataFrame(rows)


def test_render_chart_png_default_focus() -> None:
    df = _ohlcv_df(90)
    ctx, setups = build_context_and_setups(
        ohlcv_df=df,
        symbol="BTC/USDT",
        timeframe="1h",
        swing_left=2,
        swing_right=2,
    )
    png = render_chart(ctx, setups, title="test")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 2000


def test_render_chart_full_width_no_focus() -> None:
    df = _ohlcv_df(60)
    ctx, setups = build_context_and_setups(
        ohlcv_df=df,
        symbol="ETH/USDT",
        timeframe="4h",
        swing_left=2,
        swing_right=2,
    )
    png = render_chart(ctx, setups, focus_last_bars=None, compact_overlays=False)
    assert png[:4] == b"\x89PNG"
    assert len(png) > 2000


def test_render_chart_tight_focus() -> None:
    df = _ohlcv_df(100)
    ctx, setups = build_context_and_setups(
        ohlcv_df=df,
        symbol="ETH/USDT",
        timeframe="5m",
        swing_left=2,
        swing_right=2,
    )
    png = render_chart(ctx, setups, focus_last_bars=25, compact_overlays=True)
    assert png[:4] == b"\x89PNG"
    assert len(png) > 2000
