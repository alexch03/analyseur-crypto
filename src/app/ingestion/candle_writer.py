"""Upsert candle rows into PostgreSQL using ON CONFLICT."""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.interfaces import CandleRow


class CandleWriter:
    """Batch upsert candles with ON CONFLICT DO UPDATE."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert_candles(
        self, symbol_id: int, timeframe_id: int, rows: Iterable[CandleRow]
    ) -> int:
        rows_list = list(rows)
        if not rows_list:
            return 0

        stmt = text("""
            INSERT INTO candles (symbol_id, timeframe_id, ts_open, "open", high, low, close, volume)
            VALUES (:sid, :tid, :ts, :o, :h, :l, :c, :v)
            ON CONFLICT ON CONSTRAINT uq_candle
            DO UPDATE SET
                "open" = EXCLUDED."open",
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
        """)

        params = [
            {
                "sid": symbol_id,
                "tid": timeframe_id,
                "ts": r.ts_open,
                "o": r.open,
                "h": r.high,
                "l": r.low,
                "c": r.close,
                "v": r.volume,
            }
            for r in rows_list
        ]

        await self._session.execute(stmt, params)
        await self._session.commit()
        return len(rows_list)
