"""Chart-pack endpoint: OHLCV + swings + S/R levels + structure events in one JSON payload."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Query
from sqlalchemy import select

from app.api.deps import SessionDep
from app.db.models.candle import Candle, MarketStructureEvent, SRLevel, SwingPoint, Symbol, Timeframe

router = APIRouter(prefix="/chart-pack", tags=["chart"])


@router.get("")
async def chart_pack(
    session: SessionDep,
    symbol: str = Query(...),
    tf: str = Query(...),
    start: datetime | None = Query(None),
    end: datetime | None = Query(None),
    limit: int = Query(500, ge=1, le=5000),
):
    base, quote = symbol.upper().split("/")

    sym = (
        await session.execute(select(Symbol).where(Symbol.base == base, Symbol.quote == quote))
    ).scalar_one_or_none()
    if sym is None:
        return {"candles": [], "swings": [], "sr_levels": [], "structure_events": []}

    tf_obj = (
        await session.execute(select(Timeframe).where(Timeframe.code == tf))
    ).scalar_one_or_none()
    if tf_obj is None:
        return {"candles": [], "swings": [], "sr_levels": [], "structure_events": []}

    filt = [Candle.symbol_id == sym.id, Candle.timeframe_id == tf_obj.id]
    if start:
        filt.append(Candle.ts_open >= start)
    if end:
        filt.append(Candle.ts_open <= end)

    candles = (
        await session.execute(
            select(Candle).where(*filt).order_by(Candle.ts_open.desc()).limit(limit)
        )
    ).scalars().all()

    swings = (
        await session.execute(
            select(SwingPoint).where(
                SwingPoint.symbol_id == sym.id, SwingPoint.timeframe_id == tf_obj.id
            )
        )
    ).scalars().all()

    sr = (
        await session.execute(
            select(SRLevel).where(
                SRLevel.symbol_id == sym.id, SRLevel.timeframe_id == tf_obj.id
            )
        )
    ).scalars().all()

    events = (
        await session.execute(
            select(MarketStructureEvent).where(
                MarketStructureEvent.symbol_id == sym.id,
                MarketStructureEvent.timeframe_id == tf_obj.id,
            )
        )
    ).scalars().all()

    return {
        "candles": [
            {"ts": c.ts_open.isoformat(), "o": c.open, "h": c.high, "l": c.low, "c": c.close, "v": c.volume}
            for c in reversed(candles)
        ],
        "swings": [
            {"ts": s.ts.isoformat(), "price": s.price, "kind": s.kind}
            for s in swings
        ],
        "sr_levels": [
            {"price": lv.price, "width": lv.width, "touches": lv.touches, "role": lv.role}
            for lv in sr
        ],
        "structure_events": [
            {"ts": e.ts.isoformat(), "type": e.event_type, "direction": e.direction}
            for e in events
        ],
    }
