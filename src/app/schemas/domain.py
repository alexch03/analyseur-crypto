"""Shared domain DTOs used across all modules.

These are pure data containers — no DB dependency, no side effects.
Modules communicate exclusively through these types and Protocol interfaces.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SwingKind(str, enum.Enum):
    HIGH = "HIGH"
    LOW = "LOW"


class Trend(str, enum.Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    UNDEFINED = "UNDEFINED"


class StructureEventType(str, enum.Enum):
    BOS = "BOS"
    CHOCH = "CHOCH"


class SRRole(str, enum.Enum):
    SUPPORT = "SUPPORT"
    RESISTANCE = "RESISTANCE"


class Side(str, enum.Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class FVGType(str, enum.Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


class OBType(str, enum.Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SwingPoint:
    index: int
    timestamp: datetime
    price: float
    kind: SwingKind


@dataclass(frozen=True, slots=True)
class SRLevel:
    price: float
    width: float
    touches: int
    role: SRRole


@dataclass(frozen=True, slots=True)
class StructureEvent:
    index: int
    timestamp: datetime
    event_type: StructureEventType
    direction: Trend
    swing_ref: SwingPoint


@dataclass(frozen=True, slots=True)
class FairValueGap:
    index: int
    timestamp: datetime
    top: float
    bottom: float
    fvg_type: FVGType
    mitigated: bool = False
    mitigation_index: int | None = None


@dataclass(frozen=True, slots=True)
class OrderBlock:
    index: int
    timestamp: datetime
    top: float
    bottom: float
    ob_type: OBType
    mitigated: bool = False
    mitigation_index: int | None = None


@dataclass(frozen=True, slots=True)
class MarketContextDTO:
    symbol: str
    timeframe: str
    ohlcv: pd.DataFrame
    swings: list[SwingPoint]
    sr_levels: list[SRLevel]
    structure_events: list[StructureEvent]
    trend: Trend
    fvgs: list[FairValueGap] = field(default_factory=list)
    order_blocks: list[OrderBlock] = field(default_factory=list)
    ifvgs: list[FairValueGap] = field(default_factory=list)
    indicator_tags: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TradeSetupDTO:
    symbol: str
    timeframe: str
    side: Side
    entry: float
    stop_loss: float
    take_profits: list[float]
    risk_reward: float
    confidence: float
    setup_type: str
    timestamp: datetime
    rationale: str = ""
    payload: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RankedSetupDTO:
    setup: TradeSetupDTO
    ml_score: float | None = None
