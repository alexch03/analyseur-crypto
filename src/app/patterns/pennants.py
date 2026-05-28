"""Detecteur Pennant (drapeau triangulaire).

PENNANT_BULL : impulsion haussière (pole) + consolidation triangulaire convergente
  -> breakout attendu UP
  -> Target = pole_length depuis le breakout

PENNANT_BEAR : impulsion baissière (pole) + consolidation triangulaire
  -> breakout attendu DOWN

Difference avec FLAG (drapeau classique) :
  FLAG : consolidation dans un canal PARALLELE (2 lignes paralleles)
  PENNANT : consolidation triangulaire (2 lignes qui CONVERGENT)

Le Pennant est generalement plus court que le Flag (~10-25 bars).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.patterns._geometry import fit_line
from app.schemas.domain import SwingKind, SwingPoint
from app.schemas.patterns import (
    BreakoutDirection, ChartPatternDTO, PatternKind,
)

_DEFAULT_POLE_MIN_PCT = 0.04            # pole >= 4% move
_DEFAULT_POLE_MAX_BARS = 15
_DEFAULT_PENNANT_MIN_BARS = 5
_DEFAULT_PENNANT_MAX_BARS = 25
_DEFAULT_PENNANT_MAX_RETRACE = 0.4      # <= 40% du pole retrace
_DEFAULT_MIN_CONVERGENCE_RATIO = 1.5    # width_start / width_end >= 1.5


class PennantDetector:
    """Detecte Pennant Bull et Pennant Bear."""

    def __init__(
        self,
        *,
        pole_min_pct: float = _DEFAULT_POLE_MIN_PCT,
        pole_max_bars: int = _DEFAULT_POLE_MAX_BARS,
        pennant_min_bars: int = _DEFAULT_PENNANT_MIN_BARS,
        pennant_max_bars: int = _DEFAULT_PENNANT_MAX_BARS,
        pennant_max_retrace: float = _DEFAULT_PENNANT_MAX_RETRACE,
        min_convergence_ratio: float = _DEFAULT_MIN_CONVERGENCE_RATIO,
    ) -> None:
        self._pole_min = pole_min_pct
        self._pole_max = pole_max_bars
        self._penn_min = pennant_min_bars
        self._penn_max = pennant_max_bars
        self._max_retrace = pennant_max_retrace
        self._min_conv = min_convergence_ratio

    def detect(
        self, ohlcv: pd.DataFrame, swings: list[SwingPoint],
        *, symbol: str, timeframe: str,
    ) -> list[ChartPatternDTO]:
        n = len(ohlcv)
        if n < 30:
            return []
        out: list[ChartPatternDTO] = []
        out.extend(self._detect_bull(ohlcv, swings, symbol, timeframe))
        out.extend(self._detect_bear(ohlcv, swings, symbol, timeframe))
        return out

    def _detect_bull(self, ohlcv, swings, symbol, timeframe):
        n = len(ohlcv)
        last_idx = n - 1
        closes = ohlcv["close"].to_numpy(dtype=float)
        highs = ohlcv["high"].to_numpy(dtype=float)
        lows = ohlcv["low"].to_numpy(dtype=float)
        last_close = float(closes[-1])
        if last_close <= 0:
            return []

        # 1. Recherche du pole : forte impulsion sur les derniers pole_max+pennant_max bars
        search_start = max(0, n - self._pole_max - self._penn_max - 5)
        # Trouve le low du debut du pole et le high de fin du pole
        for penn_end in [last_idx]:  # toujours sur la barre courante
            for penn_len in range(self._penn_min, self._penn_max + 1):
                penn_start = penn_end - penn_len
                if penn_start < search_start:
                    break
                pole_end = penn_start
                # Pole = move impulsif AVANT le pennant
                for pole_len in range(3, self._pole_max + 1):
                    pole_start = pole_end - pole_len
                    if pole_start < 0:
                        break
                    pole_low = float(lows[pole_start: pole_end + 1].min())
                    pole_high = float(highs[pole_start: pole_end + 1].max())
                    if pole_low <= 0:
                        continue
                    pole_move = (pole_high - pole_low) / pole_low
                    if pole_move < self._pole_min:
                        continue
                    # Doit etre une impulsion haussiere : close fin >= 90% du high
                    if closes[pole_end] < pole_high * 0.9:
                        continue

                    # 2. Le pennant : consolidation triangulaire convergente
                    penn_highs_idx = list(range(penn_start, penn_end + 1))
                    penn_high_y = highs[penn_start: penn_end + 1].tolist()
                    penn_low_y = lows[penn_start: penn_end + 1].tolist()

                    upper = fit_line(penn_highs_idx, penn_high_y)
                    lower = fit_line(penn_highs_idx, penn_low_y)
                    if upper is None or lower is None:
                        continue

                    # Convergence : upper descendant, lower montant
                    if upper.slope >= 0 or lower.slope <= 0:
                        continue

                    width_start = upper.value_at(penn_start) - lower.value_at(penn_start)
                    width_end = upper.value_at(penn_end) - lower.value_at(penn_end)
                    if width_start <= 0 or width_end <= 0:
                        continue
                    conv_ratio = width_start / width_end
                    if conv_ratio < self._min_conv:
                        continue

                    # 3. Retracement modere
                    penn_low_val = float(lows[penn_start: penn_end + 1].min())
                    retrace = (pole_high - penn_low_val) / (pole_high - pole_low)
                    if retrace > self._max_retrace:
                        continue

                    # OK, trouve. Genere le pattern.
                    breakout_level = float(upper.value_at(penn_end))
                    invalidation = float(lower.value_at(penn_end))
                    pole_length = pole_high - pole_low
                    target = breakout_level + pole_length

                    return [ChartPatternDTO(
                        kind=PatternKind.PENNANT_BULL,
                        symbol=symbol, timeframe=timeframe,
                        start_index=pole_start, end_index=last_idx,
                        start_timestamp=ohlcv["timestamp"].iloc[pole_start],
                        end_timestamp=ohlcv["timestamp"].iloc[last_idx],
                        breakout_level=breakout_level,
                        invalidation_level=invalidation,
                        breakout_direction=BreakoutDirection.UP,
                        height=pole_length,
                        target=target,
                        confidence=min(1.0, conv_ratio / 4.0 + 0.3),
                        upper_line=upper, lower_line=lower,
                        payload={
                            "pole_start": pole_start,
                            "pole_end": pole_end,
                            "pole_pct_move": round(pole_move * 100, 2),
                            "convergence_ratio": round(conv_ratio, 2),
                            "retrace_pct": round(retrace * 100, 1),
                        },
                    )]
        return []

    def _detect_bear(self, ohlcv, swings, symbol, timeframe):
        n = len(ohlcv)
        last_idx = n - 1
        closes = ohlcv["close"].to_numpy(dtype=float)
        highs = ohlcv["high"].to_numpy(dtype=float)
        lows = ohlcv["low"].to_numpy(dtype=float)
        last_close = float(closes[-1])
        if last_close <= 0:
            return []
        search_start = max(0, n - self._pole_max - self._penn_max - 5)

        for penn_end in [last_idx]:
            for penn_len in range(self._penn_min, self._penn_max + 1):
                penn_start = penn_end - penn_len
                if penn_start < search_start:
                    break
                pole_end = penn_start
                for pole_len in range(3, self._pole_max + 1):
                    pole_start = pole_end - pole_len
                    if pole_start < 0:
                        break
                    pole_high = float(highs[pole_start: pole_end + 1].max())
                    pole_low = float(lows[pole_start: pole_end + 1].min())
                    if pole_high <= 0:
                        continue
                    pole_move = (pole_high - pole_low) / pole_high
                    if pole_move < self._pole_min:
                        continue
                    # Impulsion baissiere : close fin <= 110% du low
                    if closes[pole_end] > pole_low * 1.1:
                        continue

                    penn_highs_idx = list(range(penn_start, penn_end + 1))
                    upper = fit_line(penn_highs_idx, highs[penn_start: penn_end + 1].tolist())
                    lower = fit_line(penn_highs_idx, lows[penn_start: penn_end + 1].tolist())
                    if upper is None or lower is None:
                        continue
                    if upper.slope >= 0 or lower.slope <= 0:
                        continue

                    width_start = upper.value_at(penn_start) - lower.value_at(penn_start)
                    width_end = upper.value_at(penn_end) - lower.value_at(penn_end)
                    if width_start <= 0 or width_end <= 0:
                        continue
                    conv_ratio = width_start / width_end
                    if conv_ratio < self._min_conv:
                        continue

                    penn_high_val = float(highs[penn_start: penn_end + 1].max())
                    retrace = (penn_high_val - pole_low) / (pole_high - pole_low)
                    if retrace > self._max_retrace:
                        continue

                    breakout_level = float(lower.value_at(penn_end))
                    invalidation = float(upper.value_at(penn_end))
                    pole_length = pole_high - pole_low
                    target = breakout_level - pole_length

                    return [ChartPatternDTO(
                        kind=PatternKind.PENNANT_BEAR,
                        symbol=symbol, timeframe=timeframe,
                        start_index=pole_start, end_index=last_idx,
                        start_timestamp=ohlcv["timestamp"].iloc[pole_start],
                        end_timestamp=ohlcv["timestamp"].iloc[last_idx],
                        breakout_level=breakout_level,
                        invalidation_level=invalidation,
                        breakout_direction=BreakoutDirection.DOWN,
                        height=pole_length,
                        target=target,
                        confidence=min(1.0, conv_ratio / 4.0 + 0.3),
                        upper_line=upper, lower_line=lower,
                        payload={
                            "pole_start": pole_start,
                            "pole_end": pole_end,
                            "pole_pct_move": round(pole_move * 100, 2),
                            "convergence_ratio": round(conv_ratio, 2),
                            "retrace_pct": round(retrace * 100, 1),
                        },
                    )]
        return []
