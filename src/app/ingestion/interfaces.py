"""Protocol interfaces for the data ingestion layer."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True, slots=True)
class CandleRow:
    ts_open: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class OHLCVFetcher(Protocol):
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int | None = None,
        limit: int = 500,
    ) -> list[CandleRow]: ...


class CandleRepository(Protocol):
    async def upsert_candles(
        self, symbol_id: int, timeframe_id: int, rows: Iterable[CandleRow]
    ) -> int: ...
