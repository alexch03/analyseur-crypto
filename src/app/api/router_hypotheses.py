"""API routes pour les hypothèses de patterns chartistes."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import desc, select

from app.api.deps import SessionDep
from app.db.models import Hypothesis, Symbol, Timeframe

router = APIRouter(prefix="/hypotheses", tags=["hypotheses"])


@router.get("")
async def list_hypotheses(
    session: SessionDep,
    state: str | None = Query(None, description="Filtre état: FORMING, ARMED, TRIGGERED, ..."),
    symbol: str | None = Query(None, description="ex: BTC/USDT"),
    timeframe: str | None = Query(None),
    pattern_kind: str | None = Query(None),
    since: datetime | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
) -> list[dict[str, Any]]:
    q = (
        select(Hypothesis, Symbol, Timeframe)
        .join(Symbol, Symbol.id == Hypothesis.symbol_id)
        .join(Timeframe, Timeframe.id == Hypothesis.timeframe_id)
        .order_by(desc(Hypothesis.updated_at))
        .limit(limit)
    )
    if state:
        q = q.where(Hypothesis.state == state.upper())
    if symbol:
        base, quote = symbol.split("/")
        q = q.where(Symbol.base == base, Symbol.quote == quote)
    if timeframe:
        q = q.where(Timeframe.code == timeframe)
    if pattern_kind:
        q = q.where(Hypothesis.pattern_kind == pattern_kind.upper())
    if since:
        q = q.where(Hypothesis.updated_at >= since)

    rows = (await session.execute(q)).all()
    return [_hypothesis_summary(h, s, t) for h, s, t in rows]


@router.get("/{hypothesis_id}")
async def get_hypothesis(hypothesis_id: str, session: SessionDep) -> dict[str, Any]:
    q = (
        select(Hypothesis, Symbol, Timeframe)
        .join(Symbol, Symbol.id == Hypothesis.symbol_id)
        .join(Timeframe, Timeframe.id == Hypothesis.timeframe_id)
        .where(Hypothesis.id == hypothesis_id)
    )
    row = (await session.execute(q)).first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Hypothesis {hypothesis_id} not found")
    h, s, t = row
    out = _hypothesis_summary(h, s, t)
    out["pattern_snapshot"] = h.pattern_snapshot
    out["transitions"] = h.transitions
    return out


def _hypothesis_summary(h: Hypothesis, s: Symbol, t: Timeframe) -> dict[str, Any]:
    return {
        "id": h.id,
        "symbol": f"{s.base}/{s.quote}",
        "timeframe": t.code,
        "pattern_kind": h.pattern_kind,
        "side": h.side,
        "state": h.state,
        "entry_price": h.entry_price,
        "target_price": h.target_price,
        "invalidation_price": h.invalidation_price,
        "triggered_price": h.triggered_price,
        "outcome_price": h.outcome_price,
        "confluence_score": h.confluence_score,
        "confluence_tags": h.confluence_tags or [],
        "created_at": h.created_at.isoformat() if h.created_at else None,
        "updated_at": h.updated_at.isoformat() if h.updated_at else None,
        "triggered_at": h.triggered_at.isoformat() if h.triggered_at else None,
        "closed_at": h.closed_at.isoformat() if h.closed_at else None,
    }
