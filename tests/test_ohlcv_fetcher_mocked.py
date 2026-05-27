"""Tests for the OHLCV fetcher using a mocked ccxt exchange — no network calls."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from app.ingestion.ccxt_fetcher import CCXTFetcher


@pytest.fixture()
def mock_exchange():
    exchange = AsyncMock()
    exchange.fetch_ohlcv = AsyncMock(
        return_value=[
            [1704067200000, 42000.0, 42500.0, 41800.0, 42300.0, 150.5],
            [1704070800000, 42300.0, 42700.0, 42100.0, 42600.0, 200.3],
            [1704074400000, 42600.0, 43000.0, 42400.0, 42900.0, 180.1],
        ]
    )
    exchange.close = AsyncMock()
    return exchange


class TestCCXTFetcher:
    @pytest.mark.asyncio
    async def test_fetch_ohlcv_returns_candle_rows(self, mock_exchange):
        with patch("app.ingestion.ccxt_fetcher.ccxt_async") as mock_ccxt:
            mock_ccxt.binance.return_value = mock_exchange
            fetcher = CCXTFetcher("binance")
            fetcher._exchange = mock_exchange

            rows = await fetcher.fetch_ohlcv("BTC/USDT", "1h", limit=3)
            await fetcher.close()

        assert len(rows) == 3
        assert rows[0].open == 42000.0
        assert rows[0].high == 42500.0
        assert rows[0].low == 41800.0
        assert rows[0].close == 42300.0
        assert rows[0].volume == 150.5
        assert rows[0].ts_open.tzinfo is not None

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_timestamps_are_utc(self, mock_exchange):
        with patch("app.ingestion.ccxt_fetcher.ccxt_async") as mock_ccxt:
            mock_ccxt.binance.return_value = mock_exchange
            fetcher = CCXTFetcher("binance")
            fetcher._exchange = mock_exchange

            rows = await fetcher.fetch_ohlcv("BTC/USDT", "1h")
            await fetcher.close()

        for row in rows:
            assert row.ts_open.tzinfo == UTC

    @pytest.mark.asyncio
    async def test_fetch_empty_returns_empty(self):
        exchange = AsyncMock()
        exchange.fetch_ohlcv = AsyncMock(return_value=[])
        exchange.close = AsyncMock()

        fetcher = CCXTFetcher("binance")
        fetcher._exchange = exchange

        rows = await fetcher.fetch_ohlcv("BTC/USDT", "1h")
        await fetcher.close()

        assert rows == []
