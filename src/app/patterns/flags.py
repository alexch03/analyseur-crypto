"""Detection of Flags (and Pennants) after an impulse.

Geometry:
    Bull flag:
        1. **Pole**: upside directional move >= ``pole_min_pct`` in <= ``pole_max_bars``.
        2. **Flag**: sideways or slightly bearish consolidation over ``flag_min_bars`` to
           ``flag_max_bars`` candles, staying within ``flag_max_retrace`` of the pole.
        3. Expected breakout upward at the resistance level of the consolidation.
        4. Target = pole height projected from the breakout.

    Bear flag: mirror image.

We accept both a rectangle and a slight triangle (pennant) as consolidation,
without distinguishing them here — the simple "highs and lows within a band"
check is enough.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.schemas.domain import SwingKind, SwingPoint
from app.schemas.patterns import (
    BreakoutDirection,
    ChartPatternDTO,
    PatternKind,
)

_DEFAULT_POLE_MIN_PCT = 0.06       # >=6% de mouvement
_DEFAULT_POLE_MAX_BARS = 15
_DEFAULT_FLAG_MIN_BARS = 4
_DEFAULT_FLAG_MAX_BARS = 25
_DEFAULT_FLAG_MAX_RETRACE = 0.55   # retracement max 55% du pole
_DEFAULT_INSIDE_TOL_PCT = 0.003


@dataclass(frozen=True, slots=True)
class _Pole:
    start_idx: int
    end_idx: int
    start_price: float
    end_price: float
    direction: BreakoutDirection
    height: float


class FlagDetector:
    def __init__(
        self,
        *,
        pole_min_pct: float = _DEFAULT_POLE_MIN_PCT,
        pole_max_bars: int = _DEFAULT_POLE_MAX_BARS,
        flag_min_bars: int = _DEFAULT_FLAG_MIN_BARS,
        flag_max_bars: int = _DEFAULT_FLAG_MAX_BARS,
        flag_max_retrace: float = _DEFAULT_FLAG_MAX_RETRACE,
        inside_tol_pct: float = _DEFAULT_INSIDE_TOL_PCT,
    ) -> None:
        self._pole_min_pct = pole_min_pct
        self._pole_max_bars = pole_max_bars
        self._flag_min_bars = flag_min_bars
        self._flag_max_bars = flag_max_bars
        self._flag_max_retrace = flag_max_retrace
        self._inside_tol = inside_tol_pct

    def detect(
        self,
        ohlcv: pd.DataFrame,
        swings: list[SwingPoint],   # non utilisé ici mais conforme à l'interface
        *,
        symbol: str,
        timeframe: str,
    ) -> list[ChartPatternDTO]:
        n = len(ohlcv)
        if n < self._pole_max_bars + self._flag_min_bars + 1:
            return []

        last_idx = n - 1
        closes = ohlcv["close"].to_numpy(dtype=float)

        for flag_len in range(self._flag_min_bars, self._flag_max_bars + 1):
            flag_start = last_idx - flag_len + 1
            pole_end_idx = flag_start - 1
            if pole_end_idx < 2:
                continue

            pole = self._find_pole_ending_at(closes, pole_end_idx=pole_end_idx)
            if pole is None:
                continue

            consolidation = self._validate_flag(ohlcv, pole, flag_start, last_idx)
            if consolidation is None:
                continue

            cons_high, cons_low = consolidation
            return [self._build_pattern(
                ohlcv,
                pole=pole,
                flag_start=flag_start,
                last_idx=last_idx,
                cons_high=cons_high,
                cons_low=cons_low,
                symbol=symbol,
                timeframe=timeframe,
            )]
        return []

    def _find_pole_ending_at(
        self, closes: np.ndarray, *, pole_end_idx: int
    ) -> _Pole | None:
        """Pole ``pole_end_idx`` = high/low. Look for the opposite extremum in the window."""
        if pole_end_idx < 1:
            return None

        search_start = max(0, pole_end_idx - self._pole_max_bars + 1)
        window = closes[search_start: pole_end_idx + 1]
        end_price = float(closes[pole_end_idx])
        if end_price <= 0:
            return None

        # Bull candidate : start = lowest close in window (must be before pole_end)
        min_offset = int(np.argmin(window))
        bull_start_idx = search_start + min_offset
        bull_start_price = float(closes[bull_start_idx])
        bull_move = (end_price - bull_start_price) / bull_start_price if bull_start_price > 0 else 0.0

        # Bear candidate : start = highest close
        max_offset = int(np.argmax(window))
        bear_start_idx = search_start + max_offset
        bear_start_price = float(closes[bear_start_idx])
        bear_move = (bear_start_price - end_price) / bear_start_price if bear_start_price > 0 else 0.0

        best: _Pole | None = None
        if bull_move >= self._pole_min_pct and bull_start_idx < pole_end_idx:
            best = _Pole(
                start_idx=bull_start_idx,
                end_idx=pole_end_idx,
                start_price=bull_start_price,
                end_price=end_price,
                direction=BreakoutDirection.UP,
                height=end_price - bull_start_price,
            )
        if bear_move >= self._pole_min_pct and bear_start_idx < pole_end_idx:
            bear_pole = _Pole(
                start_idx=bear_start_idx,
                end_idx=pole_end_idx,
                start_price=bear_start_price,
                end_price=end_price,
                direction=BreakoutDirection.DOWN,
                height=bear_start_price - end_price,
            )
            if best is None or bear_pole.height > best.height:
                best = bear_pole
        return best

    def _validate_flag(
        self,
        ohlcv: pd.DataFrame,
        pole: _Pole,
        flag_start: int,
        last_idx: int,
    ) -> tuple[float, float] | None:
        highs = ohlcv["high"].to_numpy(dtype=float)
        lows = ohlcv["low"].to_numpy(dtype=float)
        sl_high = float(np.max(highs[flag_start: last_idx + 1]))
        sl_low = float(np.min(lows[flag_start: last_idx + 1]))

        # Retracement max
        if pole.direction == BreakoutDirection.UP:
            retrace = (pole.end_price - sl_low) / pole.height if pole.height > 0 else 1.0
            if retrace > self._flag_max_retrace:
                return None
            # Le sommet de la consolidation ne doit pas dépasser franchement le pole_end
            if sl_high > pole.end_price * (1.0 + self._inside_tol):
                return None
        else:
            retrace = (sl_high - pole.end_price) / pole.height if pole.height > 0 else 1.0
            if retrace > self._flag_max_retrace:
                return None
            if sl_low < pole.end_price * (1.0 - self._inside_tol):
                return None

        return sl_high, sl_low

    def _build_pattern(
        self,
        ohlcv: pd.DataFrame,
        *,
        pole: _Pole,
        flag_start: int,
        last_idx: int,
        cons_high: float,
        cons_low: float,
        symbol: str,
        timeframe: str,
    ) -> ChartPatternDTO:
        last_close = float(ohlcv["close"].iloc[-1])
        if pole.direction == BreakoutDirection.UP:
            kind = PatternKind.FLAG_BULL
            breakout_level = cons_high
            invalidation_level = cons_low
            target = breakout_level + pole.height
        else:
            kind = PatternKind.FLAG_BEAR
            breakout_level = cons_low
            invalidation_level = cons_high
            target = breakout_level - pole.height

        timestamps = ohlcv["timestamp"]
        confidence = _score_flag(
            pole_height_pct=pole.height / pole.start_price if pole.start_price > 0 else 0.0,
            flag_bars=last_idx - flag_start + 1,
            last_close=last_close,
            breakout_level=breakout_level,
        )
        return ChartPatternDTO(
            kind=kind,
            symbol=symbol,
            timeframe=timeframe,
            start_index=pole.start_idx,
            end_index=last_idx,
            start_timestamp=timestamps.iloc[pole.start_idx],
            end_timestamp=timestamps.iloc[last_idx],
            breakout_level=breakout_level,
            invalidation_level=invalidation_level,
            breakout_direction=pole.direction,
            height=pole.height,
            target=target,
            confidence=confidence,
            payload={
                "pole_start_idx": pole.start_idx,
                "pole_end_idx": pole.end_idx,
                "pole_height": pole.height,
                "pole_height_pct": pole.height / pole.start_price
                if pole.start_price > 0
                else 0.0,
                "flag_start_idx": flag_start,
                "flag_bars": last_idx - flag_start + 1,
                "consolidation_high": cons_high,
                "consolidation_low": cons_low,
            },
        )


def _score_flag(
    *, pole_height_pct: float, flag_bars: int, last_close: float, breakout_level: float
) -> float:
    pole_bonus = min(1.0, pole_height_pct / 0.15)        # 15% de pole = max
    flag_bonus = max(0.0, 1.0 - abs(flag_bars - 10) / 15.0)  # idéal autour de 10 bars
    proximity_bonus = 0.0
    if breakout_level > 0:
        prox = abs(last_close - breakout_level) / breakout_level
        proximity_bonus = max(0.0, 1.0 - prox / 0.02)    # max si à moins de 2%
    score = 0.45 * pole_bonus + 0.30 * flag_bonus + 0.25 * proximity_bonus
    return round(min(1.0, max(0.0, score)), 3)
