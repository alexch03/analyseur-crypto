"""Detection of triangles: ascending, descending, symmetrical.

Geometry:
    ASC   — horizontal resistance (flattened highs) + ascending support (rising lows)
    DESC  — horizontal support (flattened lows) + descending resistance (falling highs)
    SYM   — descending highs + ascending lows (convergence with no directional bias)

Validity conditions:
    - At least ``min_pivots_per_side`` swings on each side (default 2)
    - R^2 of both lines >= ``min_r_squared``
    - Convergence: the apex (intersection) is in the near future (1 -> ~max_apex_dist bars)
      or has slightly passed (late SYM case)
    - Pattern not yet broken: last close between the two lines (with tolerance)

Target = initial triangle height projected from the breakout level.
Invalidation = line opposite to the expected breakout direction, at the current level.
"""

from __future__ import annotations

import pandas as pd

from app.patterns._geometry import fit_line, is_flat, slope_pct_per_bar
from app.schemas.domain import SwingKind, SwingPoint
from app.schemas.patterns import (
    BreakoutDirection,
    ChartPatternDTO,
    PatternKind,
    TrendLine,
)

_DEFAULT_WINDOW_BARS = 80
_DEFAULT_MIN_PIVOTS = 2
_DEFAULT_MIN_R2 = 0.55
_DEFAULT_FLAT_TOL_PCT_PER_BAR = 0.0006   # 0.06% par bougie ≈ "horizontal"
_DEFAULT_MIN_SLOPE_PCT_PER_BAR = 0.0008  # au-delà : pente non triviale
_DEFAULT_MIN_HEIGHT_PCT = 0.012          # >=1.2% pour éviter le bruit
_DEFAULT_MAX_APEX_DIST_BARS = 60         # apex pas trop loin dans le futur
_DEFAULT_INSIDE_TOL_PCT = 0.002          # close peut dépasser de 0.2% sans casser


class TriangleDetector:
    def __init__(
        self,
        *,
        window_bars: int = _DEFAULT_WINDOW_BARS,
        min_pivots_per_side: int = _DEFAULT_MIN_PIVOTS,
        min_r_squared: float = _DEFAULT_MIN_R2,
        flat_tol_pct_per_bar: float = _DEFAULT_FLAT_TOL_PCT_PER_BAR,
        min_slope_pct_per_bar: float = _DEFAULT_MIN_SLOPE_PCT_PER_BAR,
        min_height_pct: float = _DEFAULT_MIN_HEIGHT_PCT,
        max_apex_dist_bars: int = _DEFAULT_MAX_APEX_DIST_BARS,
        inside_tol_pct: float = _DEFAULT_INSIDE_TOL_PCT,
    ) -> None:
        self._window = window_bars
        self._min_pivots = max(2, min_pivots_per_side)
        self._min_r2 = min_r_squared
        self._flat_tol = flat_tol_pct_per_bar
        self._min_slope = min_slope_pct_per_bar
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

        high_line = fit_line([s.index for s in highs], [s.price for s in highs])
        low_line = fit_line([s.index for s in lows], [s.price for s in lows])
        if high_line is None or low_line is None:
            return []

        last_close = float(ohlcv["close"].iloc[-1])
        if last_close <= 0.0:
            return []

        high_slope = slope_pct_per_bar(high_line, last_close)
        low_slope = slope_pct_per_bar(low_line, last_close)
        high_flat = is_flat(high_line, last_close, tol_pct_per_bar=self._flat_tol)
        low_flat = is_flat(low_line, last_close, tol_pct_per_bar=self._flat_tol)

        kind = _classify_triangle(
            high_slope=high_slope,
            low_slope=low_slope,
            high_flat=high_flat,
            low_flat=low_flat,
            min_slope=self._min_slope,
        )
        if kind is None:
            return []

        # R² ne pénalise que les lignes inclinées : une horizontale fitte mal en R²
        # mais reste valide si la pente normalisée est sous la tolérance.
        if kind == PatternKind.TRIANGLE_ASC and low_line.r_squared < self._min_r2:
            return []
        if kind == PatternKind.TRIANGLE_DESC and high_line.r_squared < self._min_r2:
            return []
        if kind == PatternKind.TRIANGLE_SYM and (
            high_line.r_squared < self._min_r2 or low_line.r_squared < self._min_r2
        ):
            return []

        if kind == PatternKind.TRIANGLE_ASC and not _flat_band_ok(
            [s.price for s in highs], last_close, tol_pct=self._inside_tol * 3
        ):
            return []
        if kind == PatternKind.TRIANGLE_DESC and not _flat_band_ok(
            [s.price for s in lows], last_close, tol_pct=self._inside_tol * 3
        ):
            return []

        if not _converges(high_line, low_line, last_idx, max_apex=self._max_apex):
            return []

        start_idx = min(
            min(s.index for s in highs),
            min(s.index for s in lows),
        )
        height = _initial_height(high_line, low_line, start_idx)
        if height <= 0.0 or height / last_close < self._min_height_pct:
            return []

        upper_at_now = float(high_line.value_at(last_idx))
        lower_at_now = float(low_line.value_at(last_idx))
        if last_close > upper_at_now * (1.0 + self._inside_tol):
            return []
        if last_close < lower_at_now * (1.0 - self._inside_tol):
            return []

        breakout_level, invalidation_level, direction, target = _resolve_targets(
            kind=kind,
            upper_at_now=upper_at_now,
            lower_at_now=lower_at_now,
            height=height,
        )

        confidence = _score(high_line, low_line, height, last_close, kind)
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
            upper_line=high_line,
            lower_line=low_line,
            confidence=confidence,
            payload={
                "high_pivots": [(s.index, s.price) for s in highs],
                "low_pivots": [(s.index, s.price) for s in lows],
                "apex_index": _apex_index(high_line, low_line),
            },
        )]


def _flat_band_ok(prices: list[float], ref_price: float, *, tol_pct: float) -> bool:
    if not prices or ref_price <= 0.0:
        return False
    span = max(prices) - min(prices)
    return (span / ref_price) <= tol_pct


def _classify_triangle(
    *,
    high_slope: float,
    low_slope: float,
    high_flat: bool,
    low_flat: bool,
    min_slope: float,
) -> PatternKind | None:
    if high_flat and low_slope > min_slope:
        return PatternKind.TRIANGLE_ASC
    if low_flat and high_slope < -min_slope:
        return PatternKind.TRIANGLE_DESC
    if high_slope < -min_slope and low_slope > min_slope:
        return PatternKind.TRIANGLE_SYM
    return None


def _converges(upper: TrendLine, lower: TrendLine, last_idx: int, *, max_apex: int) -> bool:
    diff_slope = upper.slope - lower.slope
    if abs(diff_slope) < 1e-12:
        return False
    apex_x = (lower.intercept - upper.intercept) / diff_slope
    if apex_x < last_idx - 5:
        return False
    if apex_x > last_idx + max_apex:
        return False
    return True


def _apex_index(upper: TrendLine, lower: TrendLine) -> int | None:
    diff_slope = upper.slope - lower.slope
    if abs(diff_slope) < 1e-12:
        return None
    return int((lower.intercept - upper.intercept) / diff_slope)


def _initial_height(upper: TrendLine, lower: TrendLine, start_idx: int) -> float:
    return float(upper.value_at(start_idx) - lower.value_at(start_idx))


def _resolve_targets(
    *,
    kind: PatternKind,
    upper_at_now: float,
    lower_at_now: float,
    height: float,
) -> tuple[float, float, BreakoutDirection, float | None]:
    if kind == PatternKind.TRIANGLE_ASC:
        return upper_at_now, lower_at_now, BreakoutDirection.UP, upper_at_now + height
    if kind == PatternKind.TRIANGLE_DESC:
        return lower_at_now, upper_at_now, BreakoutDirection.DOWN, lower_at_now - height
    return upper_at_now, lower_at_now, BreakoutDirection.UNDETERMINED, None


def _score(
    upper: TrendLine,
    lower: TrendLine,
    height: float,
    ref_price: float,
    kind: PatternKind,
) -> float:
    r2_bonus = (upper.r_squared + lower.r_squared) / 2.0
    touches_bonus = min(1.0, (len(upper.indices_used) + len(lower.indices_used)) / 8.0)
    height_bonus = min(1.0, (height / ref_price) / 0.05) if ref_price > 0 else 0.0
    directional_bonus = 0.05 if kind != PatternKind.TRIANGLE_SYM else 0.0
    score = 0.55 * r2_bonus + 0.25 * touches_bonus + 0.15 * height_bonus + directional_bonus
    return round(min(1.0, max(0.0, score)), 3)
