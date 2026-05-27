"""Persistance des hypothèses, unit trades et scan runs (Postgres / asyncpg).

Bridge entre les DTOs frozen (schemas/) et les ORM rows (db/models/hypothesis).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Hypothesis, ScanRun, Symbol, Timeframe, UnitTrade
from app.paper.unit_tracker import UnitTradeDTO
from app.schemas.domain import Side
from app.schemas.hypothesis import (
    HypothesisDTO,
    HypothesisState,
    StateTransition,
)
from app.schemas.patterns import (
    BreakoutDirection,
    ChartPatternDTO,
    PatternKind,
)


def _serialize_pattern(p: ChartPatternDTO) -> dict:
    return {
        "kind": p.kind.value,
        "symbol": p.symbol,
        "timeframe": p.timeframe,
        "start_index": p.start_index,
        "end_index": p.end_index,
        "start_timestamp": p.start_timestamp.isoformat(),
        "end_timestamp": p.end_timestamp.isoformat(),
        "breakout_level": p.breakout_level,
        "invalidation_level": p.invalidation_level,
        "breakout_direction": p.breakout_direction.value,
        "height": p.height,
        "target": p.target,
        "confidence": p.confidence,
        "payload": p.payload,
        "upper_line": _serialize_line(p.upper_line),
        "lower_line": _serialize_line(p.lower_line),
    }


def _serialize_line(line) -> dict | None:
    if line is None:
        return None
    return {
        "slope": line.slope,
        "intercept": line.intercept,
        "indices_used": list(line.indices_used),
        "r_squared": line.r_squared,
    }


def _deserialize_pattern(d: dict) -> ChartPatternDTO:
    from app.schemas.patterns import TrendLine

    def _line(raw):
        if raw is None:
            return None
        return TrendLine(
            slope=float(raw["slope"]),
            intercept=float(raw["intercept"]),
            indices_used=tuple(raw["indices_used"]),
            r_squared=float(raw["r_squared"]),
        )

    return ChartPatternDTO(
        kind=PatternKind(d["kind"]),
        symbol=str(d["symbol"]),
        timeframe=str(d["timeframe"]),
        start_index=int(d["start_index"]),
        end_index=int(d["end_index"]),
        start_timestamp=datetime.fromisoformat(d["start_timestamp"]),
        end_timestamp=datetime.fromisoformat(d["end_timestamp"]),
        breakout_level=float(d["breakout_level"]),
        invalidation_level=float(d["invalidation_level"]),
        breakout_direction=BreakoutDirection(d["breakout_direction"]),
        height=float(d["height"]),
        target=(float(d["target"]) if d.get("target") is not None else None),
        confidence=float(d.get("confidence", 0.0)),
        payload=d.get("payload") or {},
        upper_line=_line(d.get("upper_line")),
        lower_line=_line(d.get("lower_line")),
    )


def _serialize_transitions(transitions: tuple[StateTransition, ...]) -> list[dict]:
    return [
        {
            "from_state": t.from_state.value,
            "to_state": t.to_state.value,
            "timestamp": t.timestamp.isoformat(),
            "price": t.price,
            "reason": t.reason,
        }
        for t in transitions
    ]


def _deserialize_transitions(raw: list | None) -> tuple[StateTransition, ...]:
    if not raw:
        return tuple()
    return tuple(
        StateTransition(
            from_state=HypothesisState(item["from_state"]),
            to_state=HypothesisState(item["to_state"]),
            timestamp=datetime.fromisoformat(item["timestamp"]),
            price=float(item["price"]),
            reason=str(item.get("reason", "")),
        )
        for item in raw
    )


async def ensure_symbol_id(session: AsyncSession, exchange_id: int, symbol: str) -> int:
    base, quote = symbol.split("/", 1)
    res = await session.execute(
        select(Symbol.id).where(
            Symbol.exchange_id == exchange_id,
            Symbol.base == base,
            Symbol.quote == quote,
        )
    )
    sid = res.scalar_one_or_none()
    if sid is not None:
        return int(sid)
    inserted = Symbol(exchange_id=exchange_id, base=base, quote=quote, active=True)
    session.add(inserted)
    await session.flush()
    return int(inserted.id)


async def ensure_timeframe_id(session: AsyncSession, code: str) -> int:
    res = await session.execute(select(Timeframe.id).where(Timeframe.code == code))
    tid = res.scalar_one_or_none()
    if tid is not None:
        return int(tid)
    inserted = Timeframe(code=code)
    session.add(inserted)
    await session.flush()
    return int(inserted.id)


async def load_active_hypotheses(
    session: AsyncSession, symbol_id: int, timeframe_id: int
) -> list[HypothesisDTO]:
    res = await session.execute(
        select(Hypothesis).where(
            Hypothesis.symbol_id == symbol_id,
            Hypothesis.timeframe_id == timeframe_id,
            Hypothesis.state.in_([
                HypothesisState.FORMING.value,
                HypothesisState.ARMED.value,
                HypothesisState.TRIGGERED.value,
            ]),
        )
    )
    rows = res.scalars().all()
    return [_row_to_dto(r) for r in rows]


def _row_to_dto(r: Hypothesis) -> HypothesisDTO:
    pattern = _deserialize_pattern(r.pattern_snapshot or {})
    return HypothesisDTO(
        id=str(r.id),
        pattern=pattern,
        symbol=pattern.symbol,
        timeframe=pattern.timeframe,
        side=Side(r.side),
        entry_price=float(r.entry_price),
        target_price=float(r.target_price),
        invalidation_price=float(r.invalidation_price),
        state=HypothesisState(r.state),
        created_at=r.created_at,
        updated_at=r.updated_at,
        arm_proximity_pct=float(r.arm_proximity_pct),
        expiry_bars=int(r.expiry_bars),
        triggered_at=r.triggered_at,
        triggered_price=(float(r.triggered_price) if r.triggered_price is not None else None),
        closed_at=r.closed_at,
        outcome_price=(float(r.outcome_price) if r.outcome_price is not None else None),
        confluence_score=float(r.confluence_score),
        confluence_tags=tuple(r.confluence_tags or []),
        transitions=_deserialize_transitions(r.transitions),
    )


async def upsert_hypothesis(
    session: AsyncSession,
    dto: HypothesisDTO,
    *,
    symbol_id: int,
    timeframe_id: int,
) -> None:
    """Insert ou update portable (SQLite + Postgres) sans dialecte specifique."""
    existing = await session.get(Hypothesis, dto.id)
    if existing is None:
        session.add(Hypothesis(
            id=dto.id,
            symbol_id=symbol_id,
            timeframe_id=timeframe_id,
            pattern_kind=dto.pattern.kind.value,
            side=dto.side.value,
            state=dto.state.value,
            entry_price=float(dto.entry_price),
            target_price=float(dto.target_price),
            invalidation_price=float(dto.invalidation_price),
            triggered_price=dto.triggered_price,
            outcome_price=dto.outcome_price,
            confluence_score=float(dto.confluence_score),
            arm_proximity_pct=float(dto.arm_proximity_pct),
            expiry_bars=int(dto.expiry_bars),
            created_at=dto.created_at,
            updated_at=dto.updated_at,
            triggered_at=dto.triggered_at,
            closed_at=dto.closed_at,
            pattern_snapshot=_serialize_pattern(dto.pattern),
            transitions=_serialize_transitions(dto.transitions),
            confluence_tags=list(dto.confluence_tags),
        ))
        return

    existing.state = dto.state.value
    existing.pattern_kind = dto.pattern.kind.value
    existing.side = dto.side.value
    existing.entry_price = float(dto.entry_price)
    existing.target_price = float(dto.target_price)
    existing.invalidation_price = float(dto.invalidation_price)
    existing.triggered_price = dto.triggered_price
    existing.outcome_price = dto.outcome_price
    existing.confluence_score = float(dto.confluence_score)
    existing.updated_at = dto.updated_at
    existing.triggered_at = dto.triggered_at
    existing.closed_at = dto.closed_at
    existing.pattern_snapshot = _serialize_pattern(dto.pattern)
    existing.transitions = _serialize_transitions(dto.transitions)
    existing.confluence_tags = list(dto.confluence_tags)


async def insert_unit_trade(
    session: AsyncSession,
    trade: UnitTradeDTO,
    *,
    symbol_id: int,
    timeframe_id: int,
) -> None:
    existing = await session.get(UnitTrade, trade.id)
    if existing is None:
        session.add(UnitTrade(
            id=trade.id,
            hypothesis_id=trade.hypothesis_id,
            symbol_id=symbol_id,
            timeframe_id=timeframe_id,
            side=trade.side.value,
            pattern_kind=trade.pattern_kind.value,
            entry_price=float(trade.entry_price),
            entry_timestamp=trade.entry_timestamp,
            exit_price=trade.exit_price,
            exit_timestamp=trade.exit_timestamp,
            pct_gain=trade.pct_gain,
            outcome=trade.outcome,
            confluence_score=float(trade.confluence_score),
            confluence_tags=list(trade.confluence_tags),
        ))
        return
    # Update : on ne touche que les champs de cloture.
    existing.exit_price = trade.exit_price
    existing.exit_timestamp = trade.exit_timestamp
    existing.pct_gain = trade.pct_gain
    existing.outcome = trade.outcome


async def load_open_unit_trades_for_symbol_tf(
    session: AsyncSession, symbol_id: int, timeframe_id: int
) -> list[UnitTradeDTO]:
    res = await session.execute(
        select(UnitTrade).where(
            UnitTrade.symbol_id == symbol_id,
            UnitTrade.timeframe_id == timeframe_id,
            UnitTrade.exit_price.is_(None),
        )
    )
    rows = res.scalars().all()
    return [
        UnitTradeDTO(
            id=str(r.id),
            hypothesis_id=str(r.hypothesis_id),
            symbol=_symbol_str_for(r.symbol_id),  # filled by caller via JOIN if needed
            timeframe="",   # idem
            side=Side(r.side),
            pattern_kind=PatternKind(r.pattern_kind),
            entry_price=float(r.entry_price),
            entry_timestamp=r.entry_timestamp,
            exit_price=(float(r.exit_price) if r.exit_price is not None else None),
            exit_timestamp=r.exit_timestamp,
            pct_gain=(float(r.pct_gain) if r.pct_gain is not None else None),
            outcome=r.outcome,
            confluence_score=float(r.confluence_score),
            confluence_tags=tuple(r.confluence_tags or []),
        )
        for r in rows
    ]


def _symbol_str_for(symbol_id: int) -> str:
    """Placeholder — la résolution complète passe par un JOIN dans la query.

    Le scanner connaît symbol/timeframe par contexte ; ce champ est rempli après
    chargement via ``replace`` avant d'être passé à ``reconcile_with_engine_step``.
    """
    return ""


async def record_scan_run(
    session: AsyncSession,
    *,
    symbol_id: int,
    timeframe_id: int,
    ts_started: datetime,
    ts_finished: datetime,
    candles_fetched: int,
    patterns_detected: int,
    hypotheses_active: int,
    error: str | None = None,
) -> None:
    session.add(ScanRun(
        symbol_id=symbol_id,
        timeframe_id=timeframe_id,
        ts_started=ts_started,
        ts_finished=ts_finished,
        candles_fetched=candles_fetched,
        patterns_detected=patterns_detected,
        hypotheses_active=hypotheses_active,
        error=(error[:500] if error else None),
    ))


__all__ = [
    "ensure_symbol_id",
    "ensure_timeframe_id",
    "load_active_hypotheses",
    "load_open_unit_trades_for_symbol_tf",
    "upsert_hypothesis",
    "insert_unit_trade",
    "record_scan_run",
]
