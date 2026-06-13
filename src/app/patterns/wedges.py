"""Detection of wedges (WEDGE_RISING / WEDGE_FALLING).

Geometry:
    Rising wedge — bearish bias:
        - Highs AND lows are rising, BUT lows rise faster than highs.
        - Lines converge upward, apex ahead.
        - Expected breakout through the **base** -> BreakoutDirection.DOWN.
    Falling wedge — bullish bias:
        - Highs AND lows are falling, BUT highs fall faster than lows.
        - Lines converge downward, apex ahead.
        - Expected breakout through the **roof** -> BreakoutDirection.UP.

Classical target = return to the starting level of the wedge (price action retraces
the whole formation).
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
_DEFAULT_MIN_R2 = 0.55
_DEFAULT_MIN_SLOPE_PCT_PER_BAR = 0.0005    # 0.05%/bar — wedge a souvent une ligne "lente"
_DEFAULT_MIN_CONVERGENCE_RATIO = 1.3       # un côté doit avoir |slope| >= 1.3× l'autre
_DEFAULT_MIN_HEIGHT_PCT = 0.012
_DEFAULT_MAX_APEX_DIST_BARS = 80
_DEFAULT_INSIDE_TOL_PCT = 0.003


class WedgeDetector:
    def __init__(
        self,
        *,
        window_bars: int = _DEFAULT_WINDOW_BARS,
        min_pivots_per_side: int = _DEFAULT_MIN_PIVOTS,
        min_r_squared: float = _DEFAULT_MIN_R2,
        min_slope_pct_per_bar: float = _DEFAULT_MIN_SLOPE_PCT_PER_BAR,
        min_convergence_ratio: float = _DEFAULT_MIN_CONVERGENCE_RATIO,
        min_height_pct: float = _DEFAULT_MIN_HEIGHT_PCT,
        max_apex_dist_bars: int = _DEFAULT_MAX_APEX_DIST_BARS,
        inside_tol_pct: float = _DEFAULT_INSIDE_TOL_PCT,
    ) -> None:
        self._window = window_bars
        self._min_pivots = max(2, min_pivots_per_side)
        self._min_r2 = min_r_squared
        self._min_slope = min_slope_pct_per_bar
        self._min_conv = min_convergence_ratio
        self._min_height_pct = min_height_pct
        self._max_apex = max_apex_dist_bars
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

        u_slope = slope_pct_per_bar(upper, last_close)
        l_slope = slope_pct_per_bar(lower, last_close)
        if abs(u_slope) < self._min_slope or abs(l_slope) < self._min_slope:
            return []
        if (u_slope > 0) != (l_slope > 0):
            return []

        kind = None
        if u_slope > 0 and l_slope > 0:
            # Rising wedge : lows montent plus vite que highs → l_slope > u_slope
            if l_slope / u_slope >= self._min_conv:
                kind = PatternKind.WEDGE_RISING
        else:
            # Falling wedge : highs descendent plus vite que lows → |u| > |l|
            if (abs(u_slope) / abs(l_slope)) >= self._min_conv:
                kind = PatternKind.WEDGE_FALLING
        if kind is None:
            return []

        if not _converges_ahead(upper, lower, last_idx, max_apex=self._max_apex):
            return []

        upper_now = float(upper.value_at(last_idx))
        lower_now = float(lower.value_at(last_idx))
        height_now = upper_now - lower_now
        if height_now <= 0 or (height_now / last_close) < self._min_height_pct:
            return []

        if last_close > upper_now * (1.0 + self._inside_tol):
            return []
        if last_close < lower_now * (1.0 - self._inside_tol):
            return []

        start_idx = min(
            min(s.index for s in highs),
            min(s.index for s in lows),
        )
        start_height = float(upper.value_at(start_idx) - lower.value_at(start_idx))

        if kind == PatternKind.WEDGE_RISING:
            direction = BreakoutDirection.DOWN
            breakout_level = lower_now
            invalidation_level = upper_now
            target = float(lower.value_at(start_idx))
        else:
            direction = BreakoutDirection.UP
            breakout_level = upper_now
            invalidation_level = lower_now
            target = float(upper.value_at(start_idx))

        confidence = _score_wedge(upper, lower, start_height, last_close)
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
            height=start_height,
            target=target,
            upper_line=upper,
            lower_line=lower,
            confidence=confidence,
            payload={
                "slope_upper_pct": u_slope,
                "slope_lower_pct": l_slope,
                "convergence_ratio": l_slope / u_slope if u_slope != 0 else 0.0,
            },
        )]


def _converges_ahead(upper, lower, last_idx: int, *, max_apex: int) -> bool:
    diff = upper.slope - lower.slope
    if abs(diff) < 1e-12:
        return False
    apex_x = (lower.intercept - upper.intercept) / diff
    return last_idx - 5 <= apex_x <= last_idx + max_apex


def _score_wedge(upper, lower, height: float, ref: float) -> float:
    r2 = (upper.r_squared + lower.r_squared) / 2.0
    touches = min(1.0, (len(upper.indices_used) + len(lower.indices_used)) / 8.0)
    height_bonus = min(1.0, (height / ref) / 0.05) if ref > 0 else 0.0
    return round(min(1.0, max(0.0, 0.55 * r2 + 0.25 * touches + 0.20 * height_bonus)), 3)
