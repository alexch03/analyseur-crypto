"""Detection of inclined parallel channels (CHANNEL_UP / CHANNEL_DOWN).

Geometry:
    - Two inclined lines with **same-signed** and **near-equal** slopes
      (slopes_diff_pct <= tolerance).
    - At least ``min_pivots_per_side`` swings on each side.
    - High R^2 on both lines.

Breakout direction is undetermined: a channel can break either way. The choice
is delegated to the hypothesis engine at the real breakout — here we emit an
UNDETERMINED entry with ``breakout_level`` = line in the trend direction (cap)
and ``invalidation_level`` = opposite line (a classical breakout targets the
horizontal break in the trend direction).

Target = projection of the channel width at the breakout point (= height).
"""

from __future__ import annotations

import pandas as pd

from app.patterns._geometry import fit_line, slope_pct_per_bar
from app.schemas.domain import SwingKind, SwingPoint
from app.schemas.patterns import (
    BreakoutDirection,
    ChartPatternDTO,
    PatternKind,
)

_DEFAULT_WINDOW_BARS = 100
_DEFAULT_MIN_PIVOTS = 3
_DEFAULT_MIN_R2 = 0.6
_DEFAULT_MIN_SLOPE_PCT_PER_BAR = 0.0015
_DEFAULT_PARALLEL_TOL_PCT = 0.30           # 30% d'écart max entre pentes normalisées
_DEFAULT_MIN_WIDTH_PCT = 0.015             # largeur min 1.5% du prix
_DEFAULT_INSIDE_TOL_PCT = 0.003


class ChannelDetector:
    def __init__(
        self,
        *,
        window_bars: int = _DEFAULT_WINDOW_BARS,
        min_pivots_per_side: int = _DEFAULT_MIN_PIVOTS,
        min_r_squared: float = _DEFAULT_MIN_R2,
        min_slope_pct_per_bar: float = _DEFAULT_MIN_SLOPE_PCT_PER_BAR,
        parallel_tol_pct: float = _DEFAULT_PARALLEL_TOL_PCT,
        min_width_pct: float = _DEFAULT_MIN_WIDTH_PCT,
        inside_tol_pct: float = _DEFAULT_INSIDE_TOL_PCT,
    ) -> None:
        self._window = window_bars
        self._min_pivots = max(2, min_pivots_per_side)
        self._min_r2 = min_r_squared
        self._min_slope = min_slope_pct_per_bar
        self._parallel_tol = parallel_tol_pct
        self._min_width_pct = min_width_pct
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
        if n < self._min_pivots * 2:
            return []

        last_idx = n - 1
        start_window = max(0, last_idx - self._window)
        recent = [s for s in swings if start_window <= s.index <= last_idx]
        highs = [s for s in recent if s.kind == SwingKind.HIGH]
        lows = [s for s in recent if s.kind == SwingKind.LOW]
        if len(highs) < self._min_pivots or len(lows) < self._min_pivots:
            return []

        upper = fit_line([s.index for s in highs], [s.price for s in highs])
        lower = fit_line([s.index for s in lows], [s.price for s in lows])
        if upper is None or lower is None:
            return []
        if upper.r_squared < self._min_r2 or lower.r_squared < self._min_r2:
            return []

        last_close = float(ohlcv["close"].iloc[-1])
        if last_close <= 0:
            return []

        upper_slope_n = slope_pct_per_bar(upper, last_close)
        lower_slope_n = slope_pct_per_bar(lower, last_close)

        if abs(upper_slope_n) < self._min_slope or abs(lower_slope_n) < self._min_slope:
            return []
        # Pentes de même signe
        if (upper_slope_n > 0) != (lower_slope_n > 0):
            return []

        avg = (abs(upper_slope_n) + abs(lower_slope_n)) / 2.0
        if avg <= 0:
            return []
        spread = abs(upper_slope_n - lower_slope_n) / avg
        if spread > self._parallel_tol:
            return []

        kind = PatternKind.CHANNEL_UP if upper_slope_n > 0 else PatternKind.CHANNEL_DOWN

        upper_now = float(upper.value_at(last_idx))
        lower_now = float(lower.value_at(last_idx))
        width = upper_now - lower_now
        if width <= 0 or (width / last_close) < self._min_width_pct:
            return []

        if last_close > upper_now * (1.0 + self._inside_tol):
            return []
        if last_close < lower_now * (1.0 - self._inside_tol):
            return []

        # Convention : breakout = cassure de la ligne dans le sens du trend (sortie
        # par le cap dans la même direction). Invalidation = ligne opposée.
        if kind == PatternKind.CHANNEL_UP:
            breakout_level, invalidation_level = upper_now, lower_now
            direction = BreakoutDirection.UP
        else:
            breakout_level, invalidation_level = lower_now, upper_now
            direction = BreakoutDirection.DOWN

        start_idx = min(
            min(s.index for s in highs),
            min(s.index for s in lows),
        )
        height = width
        target = (
            breakout_level + height
            if direction == BreakoutDirection.UP
            else breakout_level - height
        )

        confidence = _score_channel(upper, lower, width, last_close)

        timestamps = ohlcv["timestamp"]
        return [ChartPatternDTO(
            kind=kind,
            symbol=symbol,
            timeframe=timeframe,
            start_index=start_idx,
            end_index=last_idx,
            start_timestamp=timestamps.iloc[start_idx],
            end_timestamp=timestamps.iloc[last_idx],
            breakout_level=breakout_level,
            invalidation_level=invalidation_level,
            breakout_direction=direction,
            height=height,
            target=target,
            upper_line=upper,
            lower_line=lower,
            confidence=confidence,
            payload={
                "channel_slope_pct_per_bar": (upper_slope_n + lower_slope_n) / 2.0,
                "width_pct": width / last_close,
            },
        )]


def _score_channel(upper, lower, width: float, ref_price: float) -> float:
    r2_bonus = (upper.r_squared + lower.r_squared) / 2.0
    touches_bonus = min(1.0, (len(upper.indices_used) + len(lower.indices_used)) / 8.0)
    width_bonus = min(1.0, (width / ref_price) / 0.05) if ref_price > 0 else 0.0
    score = 0.5 * r2_bonus + 0.3 * touches_bonus + 0.2 * width_bonus
    return round(min(1.0, max(0.0, score)), 3)
