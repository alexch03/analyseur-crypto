"""DTOs pour le moteur d'hypothèses de trading.

Une hypothèse est un pattern détecté qui propose un trade conditionnel. Elle
vit dans le temps et passe par des états jusqu'à validation ou invalidation.

Cycle de vie :

    FORMING ─────► ARMED ────► TRIGGERED ────► TARGET_HIT
       │             │             │       └──► STOPPED
       │             │             └──────────► EXPIRED
       │             └─► INVALIDATED  (cassure dans le mauvais sens avant trigger)
       └──► INVALIDATED (pattern cassé pendant la formation)
       └──► EXPIRED     (jamais armé après N bars)

"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime

from app.schemas.domain import Side
from app.schemas.patterns import ChartPatternDTO


class HypothesisState(str, enum.Enum):
    FORMING = "FORMING"          # Pattern détecté mais cassure encore loin
    ARMED = "ARMED"              # Prix proche du niveau de cassure, ordre virtuel armé
    TRIGGERED = "TRIGGERED"      # Cassure confirmée, position virtuelle ouverte
    TARGET_HIT = "TARGET_HIT"    # Target atteinte (gain)
    STOPPED = "STOPPED"          # Invalidation atteinte après trigger (perte)
    INVALIDATED = "INVALIDATED"  # Invalidation atteinte AVANT trigger (ordre annulé)
    EXPIRED = "EXPIRED"          # Trop de temps écoulé sans trigger


_TERMINAL_STATES: frozenset[HypothesisState] = frozenset({
    HypothesisState.TARGET_HIT,
    HypothesisState.STOPPED,
    HypothesisState.INVALIDATED,
    HypothesisState.EXPIRED,
})


def is_terminal(state: HypothesisState) -> bool:
    return state in _TERMINAL_STATES


@dataclass(frozen=True, slots=True)
class StateTransition:
    from_state: HypothesisState
    to_state: HypothesisState
    timestamp: datetime
    price: float
    reason: str


@dataclass(frozen=True, slots=True)
class HypothesisDTO:
    """Hypothèse de trade dérivée d'un pattern.

    ``entry_price`` / ``target_price`` / ``invalidation_price`` sont figés à la
    détection (ou ré-armement). Le tracker met à jour ``state``, ``triggered_at``,
    ``closed_at``, ``outcome_price``, ``confluence_score`` au fil de l'eau.
    """
    id: str                          # uuid stable pour suivi cross-tick
    pattern: ChartPatternDTO
    symbol: str
    timeframe: str
    side: Side
    entry_price: float
    target_price: float
    invalidation_price: float
    state: HypothesisState
    created_at: datetime
    updated_at: datetime
    arm_proximity_pct: float = 0.005   # Distance prix→breakout sous laquelle on passe ARMED
    expiry_bars: int = 40              # Max bougies avant expiration sans trigger
    triggered_at: datetime | None = None
    triggered_price: float | None = None
    closed_at: datetime | None = None
    outcome_price: float | None = None
    confluence_score: float = 0.0
    confluence_tags: tuple[str, ...] = field(default_factory=tuple)
    transitions: tuple[StateTransition, ...] = field(default_factory=tuple)

    @property
    def is_terminal(self) -> bool:
        return is_terminal(self.state)

    @property
    def realized_pct(self) -> float | None:
        """Gain réalisé en % (exit/entry − 1) × 100, signé selon le sens."""
        if self.triggered_price is None or self.outcome_price is None:
            return None
        if self.side == Side.LONG:
            return (self.outcome_price / self.triggered_price - 1.0) * 100.0
        return (self.triggered_price / self.outcome_price - 1.0) * 100.0
