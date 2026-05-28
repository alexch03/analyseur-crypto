"""DTOs pour les patterns chartistes géométriques.

Indépendant du moteur SMC/ICT existant (schemas/domain.py). Les détecteurs
de patterns (triangles, rectangles, channels, wedges, flags, double tops, H&S)
produisent des ``ChartPatternDTO``. Le moteur d'hypothèses (schemas/hypothesis.py)
consomme ces patterns et gère leur cycle de vie dans le temps.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime


class PatternKind(str, enum.Enum):
    TRIANGLE_ASC = "TRIANGLE_ASC"
    TRIANGLE_DESC = "TRIANGLE_DESC"
    TRIANGLE_SYM = "TRIANGLE_SYM"
    RECTANGLE = "RECTANGLE"
    CHANNEL_UP = "CHANNEL_UP"
    CHANNEL_DOWN = "CHANNEL_DOWN"
    WEDGE_RISING = "WEDGE_RISING"
    WEDGE_FALLING = "WEDGE_FALLING"
    FLAG_BULL = "FLAG_BULL"
    FLAG_BEAR = "FLAG_BEAR"
    DOUBLE_TOP = "DOUBLE_TOP"
    DOUBLE_BOTTOM = "DOUBLE_BOTTOM"
    TRIPLE_TOP = "TRIPLE_TOP"
    TRIPLE_BOTTOM = "TRIPLE_BOTTOM"
    HEAD_SHOULDERS = "HEAD_SHOULDERS"
    INVERSE_HEAD_SHOULDERS = "INVERSE_HEAD_SHOULDERS"
    EXPANDING_TRIANGLE_BEARISH = "EXPANDING_TRIANGLE_BEARISH"
    EXPANDING_TRIANGLE_BULLISH = "EXPANDING_TRIANGLE_BULLISH"
    EXPANDING_TRIANGLE_SYM = "EXPANDING_TRIANGLE_SYM"
    PENNANT_BULL = "PENNANT_BULL"
    PENNANT_BEAR = "PENNANT_BEAR"


class BreakoutDirection(str, enum.Enum):
    UP = "UP"
    DOWN = "DOWN"
    UNDETERMINED = "UNDETERMINED"


@dataclass(frozen=True, slots=True)
class TrendLine:
    """Droite y = slope * x + intercept ajustée sur des indices de bougies."""
    slope: float
    intercept: float
    indices_used: tuple[int, ...]
    r_squared: float

    def value_at(self, x: float) -> float:
        return self.slope * x + self.intercept


@dataclass(frozen=True, slots=True)
class ChartPatternDTO:
    """Pattern géométrique détecté sur OHLCV.

    Conventions :
    - ``start_index`` / ``end_index`` : bornes d'indice OHLCV couvertes
    - ``upper_line`` / ``lower_line`` : enveloppes du pattern (None si non pertinent,
      ex: double top où on stocke plutôt les pivots clés via payload)
    - ``breakout_level`` : prix de cassure attendu (ex: résistance d'un triangle ASC)
    - ``invalidation_level`` : prix au-delà duquel l'hypothèse est annulée
    - ``target`` : prix cible projeté (None tant qu'indéterminé)
    - ``height`` : amplitude du pattern (sert au calcul de target)
    - ``breakout_direction`` : sens biaisé par la géométrie (ASC → UP, DESC → DOWN, SYM → UNDETERMINED)
    - ``confidence`` : score 0-1 de la qualité géométrique (R², nombre de touches, etc.)
    - ``payload`` : extras spécifiques au type de pattern (pivots, neckline, pole height...)
    """
    kind: PatternKind
    symbol: str
    timeframe: str
    start_index: int
    end_index: int
    start_timestamp: datetime
    end_timestamp: datetime
    breakout_level: float
    invalidation_level: float
    breakout_direction: BreakoutDirection
    height: float
    target: float | None = None
    upper_line: TrendLine | None = None
    lower_line: TrendLine | None = None
    confidence: float = 0.0
    payload: dict = field(default_factory=dict)
