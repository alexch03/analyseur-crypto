"""Protected ingestion endpoint: triggers OHLCV fetch + upsert."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.deps import SessionDep
from app.config import settings
from app.ingestion.candle_metadata import ensure_symbol_timeframe_ids
from app.ingestion.candle_writer import CandleWriter
from app.ingestion.ccxt_fetcher import CCXTFetcher

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


@router.post("/run")
async def run_ingestion(
    session: SessionDep,
    symbols: list[str] | None = Query(None),
    timeframes: list[str] | None = Query(None),
    limit: int = Query(500, ge=1, le=1500),
):
    symbols_list = symbols or settings.symbols
    tf_list = timeframes or settings.timeframes

    fetcher = CCXTFetcher(settings.exchange_id)
    writer = CandleWriter(session)
    results: list[dict] = []

    try:
        for sym in symbols_list:
            for tf in tf_list:
                symbol_id, tf_id = await ensure_symbol_timeframe_ids(
                    session, settings.exchange_id, sym, tf
                )
                rows = await fetcher.fetch_ohlcv(sym, tf, limit=limit)
                count = await writer.upsert_candles(symbol_id, tf_id, rows)
                results.append({"symbol": sym, "tf": tf, "upserted": count})
    finally:
        await fetcher.close()

    return {"ingested": results}
