"""OHLCV fetcher backed by ccxt async API."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import ccxt.async_support as ccxt_async

from app.ingestion.interfaces import CandleRow
from app.services.period_utils import timeframe_bar_seconds


class CCXTFetcher:
    """Wraps a ccxt async exchange to fetch OHLCV candles with basic rate-limiting."""

    def __init__(self, exchange_id: str = "binance") -> None:
        exchange_cls = getattr(ccxt_async, exchange_id)
        self._exchange: ccxt_async.Exchange = exchange_cls({"enableRateLimit": True})

    async def close(self) -> None:
        await self._exchange.close()

    def _ohlcv_chunk_limit(self) -> int:
        """Plafond typique par requête (Binance ~1000). CCXT expose parfois ``fetchOHLCVLimit``."""
        ex = self._exchange
        if isinstance(ex.options, dict):
            for key in ("fetchOHLCVLimit", "ohlcvLimit"):
                raw = ex.options.get(key)
                if isinstance(raw, (int, float)) and int(raw) >= 50:
                    return min(1500, int(raw))
        return 1000

    @staticmethod
    def _raw_to_candles(raw: list[list]) -> list[CandleRow]:
        return [
            CandleRow(
                ts_open=datetime.fromtimestamp(row[0] / 1000, tz=UTC),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
            for row in raw
        ]

    async def _fetch_ohlcv_paginated_forward_meta(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int,
        want: int,
        chunk: int,
    ) -> tuple[list[CandleRow], dict[str, Any]]:
        """Enchaîne plusieurs ``fetchOHLCV`` (``since`` croissant) jusqu’à ``want`` bougies ou fin de série."""
        all_raw: list[list] = []
        cursor = int(since_ms)
        http_requests = 0
        max_iters = max(5, (want + chunk - 1) // chunk + 25)
        for _ in range(max_iters):
            if len(all_raw) >= want:
                break
            need = want - len(all_raw)
            req_limit = min(chunk, max(need, 1))
            raw = await self._exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=cursor, limit=req_limit
            )
            http_requests += 1
            if not raw:
                break
            all_raw.extend(raw)
            cursor = int(raw[-1][0]) + 1
            if len(raw) < req_limit:
                break

        by_ts: dict[int, list] = {}
        for row in all_raw:
            by_ts[int(row[0])] = row
        ordered = [by_ts[k] for k in sorted(by_ts.keys())]
        if len(ordered) > want:
            ordered = ordered[-want:]
        meta: dict[str, Any] = {
            "http_requests": http_requests,
            "chunk_size": chunk,
            "candles_requested": want,
            "candles_returned": len(ordered),
            "pagination": "forward_since",
        }
        return self._raw_to_candles(ordered), meta

    async def fetch_ohlcv_with_meta(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int | None = None,
        limit: int = 500,
    ) -> tuple[list[CandleRow], dict[str, Any]]:
        """Comme ``fetch_ohlcv`` mais retourne des métadonnées (requêtes HTTP, taille de page)."""
        want = max(1, int(limit))
        chunk = self._ohlcv_chunk_limit()

        if since_ms is not None:
            if want <= chunk:
                raw = await self._exchange.fetch_ohlcv(
                    symbol, timeframe=timeframe, since=int(since_ms), limit=want
                )
                meta = {
                    "http_requests": 1,
                    "chunk_size": chunk,
                    "candles_requested": want,
                    "candles_returned": len(raw),
                    "pagination": "single",
                }
                return self._raw_to_candles(raw), meta
            return await self._fetch_ohlcv_paginated_forward_meta(
                symbol, timeframe, int(since_ms), want, chunk
            )

        if want <= chunk:
            raw = await self._exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=None, limit=want
            )
            meta = {
                "http_requests": 1,
                "chunk_size": chunk,
                "candles_requested": want,
                "candles_returned": len(raw),
                "pagination": "single",
            }
            return self._raw_to_candles(raw), meta

        bar_ms = max(1, int(timeframe_bar_seconds(timeframe) * 1000))
        end_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
        approx_since = end_ms - want * bar_ms
        rows, meta = await self._fetch_ohlcv_paginated_forward_meta(
            symbol, timeframe, approx_since, want, chunk
        )
        meta["pagination"] = "multi_forward_recent_window"
        return rows, meta

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int | None = None,
        limit: int = 500,
    ) -> list[CandleRow]:
        candles, _meta = await self.fetch_ohlcv_with_meta(
            symbol, timeframe, since_ms=since_ms, limit=limit
        )
        return candles

    async def fetch_all_timeframes(
        self,
        symbol: str,
        timeframes: list[str],
        limit: int = 500,
    ) -> dict[str, list[CandleRow]]:
        results: dict[str, list[CandleRow]] = {}
        for tf in timeframes:
            results[tf] = await self.fetch_ohlcv(symbol, tf, limit=limit)
            await asyncio.sleep(0.1)
        return results
