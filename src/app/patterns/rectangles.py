"""Détection de rectangles (range horizontal).

Géométrie :
    - Support et résistance horizontaux ; les swings highs forment une bande étroite
      autour d'un prix max, les swings lows autour d'un prix min.
    - Au moins ``min_touches_per_side`` touches sur chaque ligne.
    - La largeur de bande (max-min des highs / min-max des lows) doit rester
      sous ``band_width_pct``.
    - Pattern non encore cassé : close entre support et résistance.

Target = hauteur du rectangle projetée à partir du niveau cassé (UNDETERMINED par défaut,
on n'a pas de biais directionnel a priori — l'hypothesis_engine ne spawnera pas tant
qu'on ne sait pas dans quel sens il va casser).

Pour gérer un rectangle qui propose deux scénarios (UP et DOWN), on peut émettre
deux variantes : ici on émet **une seule** entrée avec direction UNDETERMINED, et c'est
l'engine qui watch les deux bornes au moment du breakout réel.
"""

from __future__ import annotations

import pandas as pd

from app.schemas.domain import SwingKind, SwingPoint
from app.schemas.patterns import (
    BreakoutDirection,
    ChartPatternDTO,
    PatternKind,
    TrendLine,
)

_DEFAULT_WINDOW_BARS = 100
_DEFAULT_MIN_TOUCHES_PER_SIDE = 3
_DEFAULT_BAND_WIDTH_PCT = 0.015          # 1.5% max d'écart entre touches d'une même ligne
_DEFAULT_MIN_RANGE_PCT = 0.020           # 2% min entre support et résistance
_DEFAULT_INSIDE_TOL_PCT = 0.002          # tolérance dépassement pour considérer "dans la box"


class RectangleDetector:
    def __init__(
        self,
        *,
        window_bars: int = _DEFAULT_WINDOW_BARS,
        min_touches_per_side: int = _DEFAULT_MIN_TOUCHES_PER_SIDE,
        band_width_pct: float = _DEFAULT_BAND_WIDTH_PCT,
        min_range_pct: float = _DEFAULT_MIN_RANGE_PCT,
        inside_tol_pct: float = _DEFAULT_INSIDE_TOL_PCT,
    ) -> None:
        self._window = window_bars
        self._min_touches = max(2, min_touches_per_side)
        self._band_pct = band_width_pct
        self._min_range_pct = min_range_pct
        self._inside_tol = inside_tol_pct

    def detect(
        self,
        ohlcv: pd.DataFrame,
        swings: list[SwingPoint],
        *,
        symbol: str,
        timeframe: str,
    ) -> list[ChartPatternDTO]:
        n = len(ohlcv)
        if n < self._min_touches * 2:
            return []

        last_idx = n - 1
        start_window = max(0, last_idx - self._window)

        recent = [s for s in swings if start_window <= s.index <= last_idx]
        highs = sorted(
            [s for s in recent if s.kind == SwingKind.HIGH],
            key=lambda s: s.index,
        )
        lows = sorted(
            [s for s in recent if s.kind == SwingKind.LOW],
            key=lambda s: s.index,
        )
        if len(highs) < self._min_touches or len(lows) < self._min_touches:
            return []

        last_close = float(ohlcv["close"].iloc[-1])
        if last_close <= 0.0:
            return []

        resistance = _consolidate_band(
            [s.price for s in highs], ref=last_close, band_pct=self._band_pct
        )
        support = _consolidate_band(
            [s.price for s in lows], ref=last_close, band_pct=self._band_pct
        )
        if resistance is None or support is None:
            return []

        r_price, r_indices = resistance
        s_price, s_indices = support
        height = r_price - s_price
        if height <= 0 or (height / last_close) < self._min_range_pct:
            return []

        if last_close > r_price * (1.0 + self._inside_tol):
            return []
        if last_close < s_price * (1.0 - self._inside_tol):
            return []

        upper_line = TrendLine(
            slope=0.0,
            intercept=r_price,
            indices_used=tuple(highs[i].index for i in r_indices),
            r_squared=1.0,
        )
        lower_line = TrendLine(
            slope=0.0,
            intercept=s_price,
            indices_used=tuple(lows[i].index for i in s_indices),
            r_squared=1.0,
        )

        start_idx = min(
            min(highs[i].index for i in r_indices),
            min(lows[i].index for i in s_indices),
        )
        confidence = _score_rectangle(
            n_touches=len(r_indices) + len(s_indices),
            height=height,
            ref_price=last_close,
        )

        timestamps = ohlcv["timestamp"]
        return [ChartPatternDTO(
            kind=PatternKind.RECTANGLE,
            symbol=symbol,
            timeframe=timeframe,
            start_index=start_idx,
            end_index=last_idx,
            start_timestamp=timestamps.iloc[start_idx],
            end_timestamp=timestamps.iloc[last_idx],
            breakout_level=r_price,
            invalidation_level=s_price,
            breakout_direction=BreakoutDirection.UNDETERMINED,
            height=height,
            target=None,
            upper_line=upper_line,
            lower_line=lower_line,
            confidence=confidence,
            payload={
                "support_price": s_price,
                "resistance_price": r_price,
                "support_touches": len(s_indices),
                "resistance_touches": len(r_indices),
            },
        )]


def _consolidate_band(
    prices: list[float], *, ref: float, band_pct: float
) -> tuple[float, list[int]] | None:
    """Cherche un sous-ensemble de prix qui tiennent dans une bande [median*(1±band_pct/2)].

    Stratégie greedy : trie par prix, fait glisser une fenêtre dont le span <= band_pct*ref.
    Retourne (median_price, indices_dans_la_liste_originale).
    """
    if not prices:
        return None
    band = band_pct * ref
    indexed = sorted(enumerate(prices), key=lambda x: x[1])
    n = len(indexed)
    best_indices: list[int] = []
    best_median = 0.0

    lo = 0
    for hi in range(n):
        while indexed[hi][1] - indexed[lo][1] > band:
            lo += 1
        window = indexed[lo:hi + 1]
        if len(window) > len(best_indices):
            best_indices = [w[0] for w in window]
            window_prices = sorted(w[1] for w in window)
            mid = len(window_prices) // 2
            best_median = (
                window_prices[mid]
                if len(window_prices) % 2 == 1
                else (window_prices[mid - 1] + window_prices[mid]) / 2
            )

    if not best_indices:
        return None
    return best_median, best_indices


def _score_rectangle(*, n_touches: int, height: float, ref_price: float) -> float:
    touches_bonus = min(1.0, n_touches / 8.0)
    height_bonus = min(1.0, (height / ref_price) / 0.06) if ref_price > 0 else 0.0
    score = 0.6 * touches_bonus + 0.4 * height_bonus
    return round(min(1.0, max(0.0, score)), 3)
