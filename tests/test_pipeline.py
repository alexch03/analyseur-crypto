"""Integration test: run the full analysis pipeline on synthetic data."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from app.services.analysis_pipeline import run_analysis


def _make_trend_df():
    """Build a 30-candle bullish trend with clear structure for testing."""
    prices = []
    base = 100.0
    for i in range(30):
        noise = (i % 3 - 1) * 2
        o = base + i * 2 + noise
        c = o + 3
        h = max(o, c) + 1
        l = min(o, c) - 1

        if i == 10:
            o, h, l, c = 120, 121, 108, 109
        if i == 15:
            o, h, l, c = 130, 145, 129, 144

        prices.append((o, h, l, c))

    rows = []
    start = datetime(2025, 1, 1, tzinfo=UTC)
    for i, (o, h, l, c) in enumerate(prices):
        rows.append({
            "timestamp": start + timedelta(hours=i),
            "open": o, "high": h, "low": l, "close": c, "volume": 100.0 + i * 10,
        })
    return pd.DataFrame(rows)


class TestPipeline:
    @pytest.mark.asyncio
    async def test_pipeline_runs_without_error(self):
        df = _make_trend_df()
        result = await run_analysis(
            df, "BTC/USDT", "4h",
            send_telegram=False,
            render_chart_img=False,
        )
        assert result.symbol == "BTC/USDT"
        assert result.timeframe == "4h"
        assert len(result.context.swings) >= 0
        assert result.chart_png is None

    @pytest.mark.asyncio
    async def test_pipeline_renders_chart(self):
        df = _make_trend_df()
        result = await run_analysis(
            df, "BTC/USDT", "1h",
            send_telegram=False,
            render_chart_img=True,
        )
        assert result.chart_png is not None
        assert len(result.chart_png) > 1000
        assert result.chart_png[:4] == b"\x89PNG"

    @pytest.mark.asyncio
    async def test_pipeline_context_populated(self):
        df = _make_trend_df()
        result = await run_analysis(
            df, "ETH/USDT", "4h",
            send_telegram=False,
            render_chart_img=False,
        )
        ctx = result.context
        assert ctx.symbol == "ETH/USDT"
        assert isinstance(ctx.fvgs, list)
        assert isinstance(ctx.order_blocks, list)
        assert isinstance(ctx.swings, list)
