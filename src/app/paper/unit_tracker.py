"""Tracker paper trading **unit-based** (1 unité par trade, gain en %).

Modèle complémentaire au moteur ``engine_replay`` (USDT + fees + funding). Ici :
- Chaque trade pèse exactement 1 unité.
- Le gain est ``(exit/entry − 1) × 100`` (signé pour SHORT : ``(entry/exit − 1) × 100``).
- Pas de fees, pas de funding, pas de slippage modélisés.
- Cumul disponible en simple (somme arithmétique) et compound (produit).

Piloté par les transitions d'état du :class:`HypothesisEngine` : on ouvre un trade
quand l'hypothèse passe TRIGGERED, on le ferme sur TARGET_HIT ou STOPPED.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime

from app.schemas.domain import Side
from app.schemas.hypothesis import HypothesisDTO, HypothesisState
from app.schemas.patterns import PatternKind


@dataclass(frozen=True, slots=True)
class UnitTradeDTO:
    id: str
    hypothesis_id: str
    symbol: str
    timeframe: str
    side: Side
    pattern_kind: PatternKind
    entry_price: float
    entry_timestamp: datetime
    exit_price: float | None = None
    exit_timestamp: datetime | None = None
    pct_gain: float | None = None
    outcome: str | None = None
    confluence_score: float = 0.0
    confluence_tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_closed(self) -> bool:
        return self.exit_price is not None


@dataclass(frozen=True, slots=True)
class CumulativeStats:
    total_trades: int
    closed_trades: int
    open_trades: int
    win_count: int
    loss_count: int
    breakeven_count: int
    win_rate: float
    avg_pct_gain: float
    cumulative_simple_pct: float    # somme arithmétique
    cumulative_compound_pct: float  # compound (1+r₁)(1+r₂)... − 1
    best_pct: float
    worst_pct: float
    expectancy_pct: float           # win_rate * avg_win + (1-win_rate) * avg_loss


def compute_pct_gain(side: Side, entry: float, exit_: float) -> float:
    if entry <= 0.0 or exit_ <= 0.0:
        return 0.0
    if side == Side.LONG:
        return (exit_ / entry - 1.0) * 100.0
    return (entry / exit_ - 1.0) * 100.0


class UnitTracker:
    """Helpers stateless pour gérer les unit trades."""

    @staticmethod
    def open_from_hypothesis(h: HypothesisDTO) -> UnitTradeDTO | None:
        if h.state != HypothesisState.TRIGGERED:
            return None
        if h.triggered_price is None or h.triggered_at is None:
            return None
        return UnitTradeDTO(
            id=str(uuid.uuid4()),
            hypothesis_id=h.id,
            symbol=h.symbol,
            timeframe=h.timeframe,
            side=h.side,
            pattern_kind=h.pattern.kind,
            entry_price=float(h.triggered_price),
            entry_timestamp=h.triggered_at,
            confluence_score=h.confluence_score,
            confluence_tags=h.confluence_tags,
        )

    @staticmethod
    def close_from_hypothesis(
        trade: UnitTradeDTO, h: HypothesisDTO
    ) -> UnitTradeDTO | None:
        if trade.is_closed:
            return trade
        if h.state not in (HypothesisState.TARGET_HIT, HypothesisState.STOPPED):
            return None
        if h.outcome_price is None or h.closed_at is None:
            return None
        pct = compute_pct_gain(trade.side, trade.entry_price, float(h.outcome_price))
        return replace(
            trade,
            exit_price=float(h.outcome_price),
            exit_timestamp=h.closed_at,
            pct_gain=round(pct, 6),
            outcome=h.state.value,
        )

    @staticmethod
    def force_close(
        trade: UnitTradeDTO,
        *,
        exit_price: float,
        exit_timestamp: datetime,
        reason: str = "MANUAL_CLOSE",
    ) -> UnitTradeDTO:
        pct = compute_pct_gain(trade.side, trade.entry_price, exit_price)
        return replace(
            trade,
            exit_price=exit_price,
            exit_timestamp=exit_timestamp,
            pct_gain=round(pct, 6),
            outcome=reason,
        )

    @staticmethod
    def compute_cumulative(trades: list[UnitTradeDTO]) -> CumulativeStats:
        closed = [t for t in trades if t.is_closed and t.pct_gain is not None]
        open_count = sum(1 for t in trades if not t.is_closed)
        if not closed:
            return CumulativeStats(
                total_trades=len(trades),
                closed_trades=0,
                open_trades=open_count,
                win_count=0,
                loss_count=0,
                breakeven_count=0,
                win_rate=0.0,
                avg_pct_gain=0.0,
                cumulative_simple_pct=0.0,
                cumulative_compound_pct=0.0,
                best_pct=0.0,
                worst_pct=0.0,
                expectancy_pct=0.0,
            )

        gains: list[float] = [float(t.pct_gain) for t in closed]   # type: ignore[arg-type]
        wins = [g for g in gains if g > 0]
        losses = [g for g in gains if g < 0]
        breakevens = [g for g in gains if g == 0]
        win_rate = len(wins) / len(gains)
        avg_pct = sum(gains) / len(gains)

        compound_factor = 1.0
        for g in gains:
            compound_factor *= (1.0 + g / 100.0)
        compound_pct = (compound_factor - 1.0) * 100.0

        avg_win = (sum(wins) / len(wins)) if wins else 0.0
        avg_loss = (sum(losses) / len(losses)) if losses else 0.0
        expectancy = win_rate * avg_win + (1.0 - win_rate) * avg_loss

        return CumulativeStats(
            total_trades=len(trades),
            closed_trades=len(closed),
            open_trades=open_count,
            win_count=len(wins),
            loss_count=len(losses),
            breakeven_count=len(breakevens),
            win_rate=round(win_rate, 4),
            avg_pct_gain=round(avg_pct, 4),
            cumulative_simple_pct=round(sum(gains), 4),
            cumulative_compound_pct=round(compound_pct, 4),
            best_pct=round(max(gains), 4),
            worst_pct=round(min(gains), 4),
            expectancy_pct=round(expectancy, 4),
        )


def reconcile_with_engine_step(
    open_trades: list[UnitTradeDTO],
    updated_hypotheses: list[HypothesisDTO],
) -> tuple[list[UnitTradeDTO], list[UnitTradeDTO], list[UnitTradeDTO]]:
    """Aligne les trades ouverts avec les nouvelles transitions d'hypothèses.

    Retourne (still_open, newly_closed, newly_opened) :
    - newly_opened : hypothèses TRIGGERED sans trade encore ouvert
    - newly_closed : trades correspondant à des hypothèses TARGET_HIT ou STOPPED
    - still_open : trades dont l'hypothèse reste TRIGGERED
    """
    open_by_hid = {t.hypothesis_id: t for t in open_trades}
    newly_closed: list[UnitTradeDTO] = []
    still_open: list[UnitTradeDTO] = []
    newly_opened: list[UnitTradeDTO] = []
    seen_hids: set[str] = set()

    for h in updated_hypotheses:
        seen_hids.add(h.id)
        existing = open_by_hid.get(h.id)
        if h.state == HypothesisState.TRIGGERED and existing is None:
            t = UnitTracker.open_from_hypothesis(h)
            if t is not None:
                newly_opened.append(t)
        elif h.state in (HypothesisState.TARGET_HIT, HypothesisState.STOPPED) and existing is not None:
            closed = UnitTracker.close_from_hypothesis(existing, h)
            if closed is not None:
                newly_closed.append(closed)
        elif h.state == HypothesisState.TRIGGERED and existing is not None:
            still_open.append(existing)

    # Trades dont l'hypothèse n'apparaît plus (état terminal hors target/stopped, ex INVALIDATED
    # ne devrait pas arriver puisqu'il faut un TRIGGERED préalable).
    for t in open_trades:
        if t.hypothesis_id not in seen_hids:
            still_open.append(t)

    return still_open, newly_closed, newly_opened
