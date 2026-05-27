from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Query
from sqlalchemy import select

from app.api.deps import SessionDep
from app.db.models import Candle, Symbol, Timeframe

router = APIRouter(prefix="/candles", tags=["candles"])


@router.get("")
async def get_candles(
    session: SessionDep,
    symbol: str = Query(..., description="e.g. BTC/USDT"),
    tf: str = Query(..., description="e.g. 1h"),
    start: datetime | None = Query(None),
    end: datetime | None = Query(None),
    limit: int = Query(500, ge=1, le=5000),
):
    base, quote = symbol.upper().split("/")

    sym_q = select(Symbol).where(Symbol.base == base, Symbol.quote == quote)
    sym = (await session.execute(sym_q)).scalar_one_or_none()
    if sym is None:
        return []

    tf_q = select(Timeframe).where(Timeframe.code == tf)
    tf_obj = (await session.execute(tf_q)).scalar_one_or_none()
    if tf_obj is None:
        return []

    q = (
        select(Candle)
        .where(Candle.symbol_id == sym.id, Candle.timeframe_id == tf_obj.id)
        .order_by(Candle.ts_open.desc())
        .limit(limit)
    )
    if start:
        q = q.where(Candle.ts_open >= start)
    if end:
        q = q.where(Candle.ts_open <= end)

    rows = (await session.execute(q)).scalars().all()
    return [
        {
            "ts": r.ts_open.isoformat(),
            "o": r.open,
            "h": r.high,
            "l": r.low,
            "c": r.close,
            "v": r.volume,
        }
        for r in reversed(rows)
    ]
