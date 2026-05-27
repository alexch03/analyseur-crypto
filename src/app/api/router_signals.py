from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Query
from sqlalchemy import select

from app.api.deps import SessionDep
from app.db.models.candle import Signal

router = APIRouter(prefix="/signals", tags=["signals"])


@router.get("")
async def get_signals(
    session: SessionDep,
    symbol: str | None = Query(None),
    since: datetime | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
):
    q = select(Signal).order_by(Signal.ts_generated.desc()).limit(limit)
    if since:
        q = q.where(Signal.ts_generated >= since)
    rows = (await session.execute(q)).scalars().all()
    return [
        {
            "id": r.id,
            "setup_type": r.setup_type,
            "side": r.side,
            "entry": r.entry,
            "stop_loss": r.stop_loss,
            "tp1": r.take_profit_1,
            "rr": r.risk_reward,
            "confidence": r.confidence,
            "status": r.status,
            "ts": r.ts_generated.isoformat(),
        }
        for r in rows
    ]
