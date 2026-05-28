"""Detecteur Expanding Triangle (megaphone / broadening pattern).

3 variantes :
    EXPANDING_TRIANGLE_BEARISH : Higher Highs + Lower Lows, slope_up > 0, slope_down < 0
        -> reversal bearish (cassure du bas)
    EXPANDING_TRIANGLE_BULLISH : meme geometrie mais cassure du haut -> bullish reversal
    EXPANDING_TRIANGLE_SYM : symetrique, breakout dans 1 des 2 sens (UNDETERMINED)

Geometrie :
    - Au moins 2 swings highs avec slope_up > min_slope (lignes superieure montante)
    - Au moins 2 swings lows avec slope_down < -min_slope (ligne inferieure descendante)
    - Les 2 lignes divergent (ecart augmente avec le temps)
    - Volatilite croissante (signal d'incertitude)

NB : C'est l'inverse d'un Triangle classique (qui converge). Megaphone = volatility
expanding = forte indecision marche -> breakout fort attendu.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.patterns._geometry import fit_line
from app.schemas.domain import SwingKind, SwingPoint
from app.schemas.patterns import (
    BreakoutDirection, ChartPatternDTO, PatternKind,
)

_DEFAULT_WINDOW_BARS = 100
_DEFAULT_MIN_PIVOTS = 2
_DEFAULT_MIN_R2 = 0.5                  # tolerance ligne (megaphone = noisy)
_DEFAULT_MIN_SLOPE_PCT_PER_BAR = 0.001 # 0.1% par bar minimum
_DEFAULT_MIN_DIVERGENCE_RATIO = 1.5    # width_end / width_start >= 1.5
_DEFAULT_MIN_WIDTH_PCT = 0.015         # ecart de 1.5% min entre lignes a la fin


class ExpandingTriangleDetector:
    """Detecte les 3 variantes d'Expanding Triangle."""

    def __init__(
        self,
        *,
        window_bars: int = _DEFAULT_WINDOW_BARS,
        min_pivots_per_side: int = _DEFAULT_MIN_PIVOTS,
        min_r_squared: float = _DEFAULT_MIN_R2,
        min_slope_pct_per_bar: float = _DEFAULT_MIN_SLOPE_PCT_PER_BAR,
        min_divergence_ratio: float = _DEFAULT_MIN_DIVERGENCE_RATIO,
        min_width_pct: float = _DEFAULT_MIN_WIDTH_PCT,
    ) -> None:
        self._window = window_bars
        self._min_pivots = max(2, min_pivots_per_side)
        self._min_r2 = float(min_r_squared)
        self._min_slope = float(min_slope_pct_per_bar)
        self._min_div = float(min_divergence_ratio)
        self._min_width = float(min_width_pct)

    def detect(
        self, ohlcv: pd.DataFrame, swings: list[SwingPoint],
        *, symbol: str, timeframe: str,
    ) -> list[ChartPatternDTO]:
        n = len(ohlcv)
        if n < 30 or len(swings) < 4:
            return []
        last_idx = n - 1
        start = max(0, last_idx - self._window)
        last_close = float(ohlcv["close"].iloc[-1])

        highs = sorted(
            [s for s in swings if s.kind == SwingKind.HIGH
             and start <= s.index <= last_idx],
            key=lambda s: s.index,
        )
        lows = sorted(
            [s for s in swings if s.kind == SwingKind.LOW
             and start <= s.index <= last_idx],
            key=lambda s: s.index,
        )
        if len(highs) < self._min_pivots or len(lows) < self._min_pivots:
            return []

        # Garde les pivots les plus recents
        highs = highs[-4:]
        lows = lows[-4:]

        upper = fit_line([h.index for h in highs], [h.price for h in highs])
        lower = fit_line([l.index for l in lows], [l.price for l in lows])
        if upper is None or lower is None:
            return []
        if upper.r_squared < self._min_r2 or lower.r_squared < self._min_r2:
            return []

        # Pour expanding : upper slope > 0 (montante), lower slope < 0 (descendante)
        ref_price = last_close if last_close > 0 else 1.0
        upper_slope_pct = upper.slope / ref_price
        lower_slope_pct = lower.slope / ref_price

        # Cas symetrique : les 2 divergent
        upper_diverges_up = upper_slope_pct > self._min_slope
        lower_diverges_down = lower_slope_pct < -self._min_slope

        if not (upper_diverges_up and lower_diverges_down):
            return []

        # Verifie qu'ils divergent vraiment (width croissante)
        start_x = min(highs[0].index, lows[0].index)
        end_x = last_idx
        width_start = upper.value_at(start_x) - lower.value_at(start_x)
        width_end = upper.value_at(end_x) - lower.value_at(end_x)
        if width_start <= 0 or width_end <= 0:
            return []
        divergence_ratio = width_end / width_start
        if divergence_ratio < self._min_div:
            return []

        width_pct = width_end / ref_price
        if width_pct < self._min_width:
            return []

        # Determine direction du breakout selon la position du prix
        upper_now = upper.value_at(last_idx)
        lower_now = lower.value_at(last_idx)
        mid = (upper_now + lower_now) / 2.0

        # Classification : si prix proche du haut ou cassure haute -> BULLISH
        # Si prix proche du bas ou cassure basse -> BEARISH
        # Sinon -> SYM
        if last_close >= upper_now * 0.998:
            kind = PatternKind.EXPANDING_TRIANGLE_BULLISH
            direction = BreakoutDirection.UP
            breakout = upper_now
            invalidation = lower_now
            height = upper_now - lower_now
            target = upper_now + height * 0.7  # megaphone targets sont moderes
        elif last_close <= lower_now * 1.002:
            kind = PatternKind.EXPANDING_TRIANGLE_BEARISH
            direction = BreakoutDirection.DOWN
            breakout = lower_now
            invalidation = upper_now
            height = upper_now - lower_now
            target = lower_now - height * 0.7
        else:
            kind = PatternKind.EXPANDING_TRIANGLE_SYM
            direction = BreakoutDirection.UNDETERMINED
            breakout = mid
            invalidation = upper_now  # par defaut sera ajuste au breakout reel
            height = upper_now - lower_now
            target = mid + height * 0.7  # placeholder

        confidence = self._score(divergence_ratio, upper.r_squared, lower.r_squared)
        return [ChartPatternDTO(
            kind=kind,
            symbol=symbol, timeframe=timeframe,
            start_index=start_x, end_index=last_idx,
            start_timestamp=ohlcv["timestamp"].iloc[start_x],
            end_timestamp=ohlcv["timestamp"].iloc[last_idx],
            breakout_level=breakout,
            invalidation_level=invalidation,
            breakout_direction=direction,
            height=height,
            target=target,
            confidence=confidence,
            upper_line=upper, lower_line=lower,
            payload={
                "divergence_ratio": round(divergence_ratio, 2),
                "width_start": width_start,
                "width_end": width_end,
                "upper_slope_pct": round(upper_slope_pct * 100, 3),
                "lower_slope_pct": round(lower_slope_pct * 100, 3),
            },
        )]

    def _score(self, divergence: float, r2_up: float, r2_down: float) -> float:
        div_bonus = min(1.0, (divergence - 1.0) / 2.0)  # 1.5 -> 0.25, 3.0 -> 1.0
        r2_bonus = (r2_up + r2_down) / 2.0
        return round(0.5 * div_bonus + 0.5 * r2_bonus, 3)
