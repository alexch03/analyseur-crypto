"""API routes for the unit-based paper tracker (cumulative %)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Query
from sqlalchemy import desc, select

from app.api.deps import SessionDep
from app.db.models import Symbol, Timeframe, UnitTrade
from app.paper.unit_tracker import UnitTracker, UnitTradeDTO
from app.schemas.domain import Side
from app.schemas.patterns import PatternKind

router = APIRouter(prefix="/unit_paper", tags=["unit_paper"])


@router.get("/trades")
async def list_trades(
    session: SessionDep,
    symbol: str | None = Query(None),
    timeframe: str | None = Query(None),
    pattern_kind: str | None = Query(None),
    outcome: str | None = Query(None),
    open_only: bool = Query(False),
    since: datetime | None = Query(None),
    limit: int = Query(200, ge=1, le=1000),
) -> list[dict[str, Any]]:
    q = (
        select(UnitTrade, Symbol, Timeframe)
        .join(Symbol, Symbol.id == UnitTrade.symbol_id)
        .join(Timeframe, Timeframe.id == UnitTrade.timeframe_id)
        .order_by(desc(UnitTrade.entry_timestamp))
        .limit(limit)
    )
    if symbol:
        base, quote = symbol.split("/")
        q = q.where(Symbol.base == base, Symbol.quote == quote)
    if timeframe:
        q = q.where(Timeframe.code == timeframe)
    if pattern_kind:
        q = q.where(UnitTrade.pattern_kind == pattern_kind.upper())
    if outcome:
        q = q.where(UnitTrade.outcome == outcome.upper())
    if open_only:
        q = q.where(UnitTrade.exit_price.is_(None))
    if since:
        q = q.where(UnitTrade.entry_timestamp >= since)

    rows = (await session.execute(q)).all()
    return [_trade_summary(t, s, tf) for t, s, tf in rows]


@router.get("/stats")
async def get_stats(
    session: SessionDep,
    symbol: str | None = Query(None),
    timeframe: str | None = Query(None),
    pattern_kind: str | None = Query(None),
    since: datetime | None = Query(None),
) -> dict[str, Any]:
    q = (
        select(UnitTrade, Symbol, Timeframe)
        .join(Symbol, Symbol.id == UnitTrade.symbol_id)
        .join(Timeframe, Timeframe.id == UnitTrade.timeframe_id)
    )
    if symbol:
        base, quote = symbol.split("/")
        q = q.where(Symbol.base == base, Symbol.quote == quote)
    if timeframe:
        q = q.where(Timeframe.code == timeframe)
    if pattern_kind:
        q = q.where(UnitTrade.pattern_kind == pattern_kind.upper())
    if since:
        q = q.where(UnitTrade.entry_timestamp >= since)

    rows = (await session.execute(q)).all()
    dtos = [_row_to_dto(t, s, tf) for t, s, tf in rows]
    stats = UnitTracker.compute_cumulative(dtos)
    return {
        "total_trades": stats.total_trades,
        "closed_trades": stats.closed_trades,
        "open_trades": stats.open_trades,
        "win_count": stats.win_count,
        "loss_count": stats.loss_count,
        "breakeven_count": stats.breakeven_count,
        "win_rate_pct": stats.win_rate * 100.0,
        "avg_pct_gain": stats.avg_pct_gain,
        "cumulative_simple_pct": stats.cumulative_simple_pct,
        "cumulative_compound_pct": stats.cumulative_compound_pct,
        "best_pct": stats.best_pct,
        "worst_pct": stats.worst_pct,
        "expectancy_pct": stats.expectancy_pct,
    }


def _trade_summary(t: UnitTrade, s: Symbol, tf: Timeframe) -> dict[str, Any]:
    return {
        "id": t.id,
        "hypothesis_id": t.hypothesis_id,
        "symbol": f"{s.base}/{s.quote}",
        "timeframe": tf.code,
        "side": t.side,
        "pattern_kind": t.pattern_kind,
        "entry_price": t.entry_price,
        "entry_timestamp": t.entry_timestamp.isoformat(),
        "exit_price": t.exit_price,
        "exit_timestamp": t.exit_timestamp.isoformat() if t.exit_timestamp else None,
        "pct_gain": t.pct_gain,
        "outcome": t.outcome,
        "confluence_score": t.confluence_score,
        "confluence_tags": t.confluence_tags or [],
    }


def _row_to_dto(t: UnitTrade, s: Symbol, tf: Timeframe) -> UnitTradeDTO:
    return UnitTradeDTO(
        id=str(t.id),
        hypothesis_id=str(t.hypothesis_id),
        symbol=f"{s.base}/{s.quote}",
        timeframe=tf.code,
        side=Side(t.side),
        pattern_kind=PatternKind(t.pattern_kind),
        entry_price=float(t.entry_price),
        entry_timestamp=t.entry_timestamp,
        exit_price=(float(t.exit_price) if t.exit_price is not None else None),
        exit_timestamp=t.exit_timestamp,
        pct_gain=(float(t.pct_gain) if t.pct_gain is not None else None),
        outcome=t.outcome,
        confluence_score=float(t.confluence_score),
        confluence_tags=tuple(t.confluence_tags or []),
    )
