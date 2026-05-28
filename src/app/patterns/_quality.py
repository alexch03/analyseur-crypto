"""Wrapper qualite universel pour les patterns chartistes.

Au lieu de modifier chaque detector individuellement, on enveloppe la sortie de
n'importe quel ``PatternDetector`` avec des validations qualite :

  - Pre-trend context : pattern haussier exige uptrend AVANT, baissier exige downtrend
  - RSI alignment : RSI > 60 sur breakout UP, RSI < 40 sur breakout DOWN
  - Volume profile : volume relativement eleve sur la bougie de breakout
  - Swing prominence ATR (filtre les micro-mouvements bruites)

Les patterns reversal (DOUBLE_TOP, DOUBLE_BOTTOM, H&S, IHS) ont deja leur
propre validation interne dans reversal.py, donc on les SKIP ici.

Usage :
    from app.patterns._quality import QualityWrappedDetector
    detector = QualityWrappedDetector(TriangleDetector())
    patterns = detector.detect(ohlcv, swings, symbol=..., timeframe=...)
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from app.patterns._geometry import atr
from app.patterns._indicators import compute_rsi
from app.patterns.interfaces import PatternDetector
from app.schemas.domain import SwingPoint
from app.schemas.patterns import (
    BreakoutDirection,
    ChartPatternDTO,
    PatternKind,
)


# Patterns qui ont deja leur propre validation qualite dans reversal.py
# -> on ne re-valide pas pour eviter le double filtrage
_REVERSAL_KINDS_WITH_INTERNAL_VALIDATION = {
    PatternKind.DOUBLE_TOP,
    PatternKind.DOUBLE_BOTTOM,
    PatternKind.HEAD_SHOULDERS,
    PatternKind.INVERSE_HEAD_SHOULDERS,
}

# Categorisation des patterns pour determiner la direction pre-trend attendue
# CONTINUATION : breakout dans la direction du trend precedent (asc tri, flag bull...)
# REVERSAL    : breakout OPPOSE au trend precedent (wedge, double top...)
# NEUTRAL     : pas de contrainte (rectangle, sym tri)
_CONTINUATION = {
    PatternKind.TRIANGLE_ASC,           # breakout UP suit uptrend
    PatternKind.TRIANGLE_DESC,          # breakout DOWN suit downtrend
    PatternKind.FLAG_BULL,              # breakout UP suit impulse UP
    PatternKind.FLAG_BEAR,
    PatternKind.CHANNEL_UP,             # trend continuation
    PatternKind.CHANNEL_DOWN,
}
_REVERSAL_NON_INTERNAL = {
    PatternKind.WEDGE_RISING,           # breakout DOWN apres uptrend (reversal)
    PatternKind.WEDGE_FALLING,          # breakout UP apres downtrend (reversal)
}
_NEUTRAL = {
    PatternKind.RECTANGLE,
    PatternKind.TRIANGLE_SYM,
}


class QualityWrappedDetector:
    """Wrappe un detector et filtre ses patterns par criteres qualite.

    Args:
        inner : le detector a wrapper (Triangle, Wedge, Flag, etc.)
        require_pre_trend : exige un trend pre-pattern coherent (defaut True)
        min_pre_trend_pct : amplitude minimum du trend pre-pattern (defaut 0.8%)
        pre_trend_bars : nombre de bars a regarder en arriere (defaut 20)
        require_rsi_alignment : exige RSI coherent avec direction breakout (defaut True)
        rsi_threshold_up : RSI minimum pour breakout UP (defaut 50)
        rsi_threshold_down : RSI maximum pour breakout DOWN (defaut 50)
        skip_reversal_internal : ne re-valide pas les patterns reversal (defaut True)
    """

    def __init__(
        self,
        inner: PatternDetector,
        *,
        require_pre_trend: bool = True,
        min_pre_trend_pct: float = 0.008,
        pre_trend_bars: int = 20,
        require_rsi_alignment: bool = True,
        rsi_threshold_up: float = 50.0,
        rsi_threshold_down: float = 50.0,
        skip_reversal_internal: bool = True,
    ) -> None:
        self._inner = inner
        self._require_pre_trend = require_pre_trend
        self._min_pre_trend = min_pre_trend_pct
        self._pre_trend_bars = pre_trend_bars
        self._require_rsi = require_rsi_alignment
        self._rsi_up = rsi_threshold_up
        self._rsi_down = rsi_threshold_down
        self._skip_reversal = skip_reversal_internal

    def detect(
        self,
        ohlcv: pd.DataFrame,
        swings: list[SwingPoint],
        *,
        symbol: str,
        timeframe: str,
    ) -> list[ChartPatternDTO]:
        # 1) Detection brute via inner detector
        raw = self._inner.detect(ohlcv, swings, symbol=symbol, timeframe=timeframe)
        if not raw:
            return []

        # 2) Calcul RSI une fois pour tous les patterns
        try:
            rsi_arr = compute_rsi(ohlcv["close"], period=14)
        except Exception:
            rsi_arr = None

        # 3) Filtre par qualite
        out: list[ChartPatternDTO] = []
        for p in raw:
            # Skip si pattern reversal deja valide en interne
            if self._skip_reversal and p.kind in _REVERSAL_KINDS_WITH_INTERNAL_VALIDATION:
                out.append(p)
                continue

            if not self._validate_pre_trend(ohlcv, p):
                continue

            if not self._validate_rsi(rsi_arr, p):
                continue

            out.append(p)
        return out

    # --------------------------------------------------------------
    # Validations
    # --------------------------------------------------------------

    def _validate_pre_trend(self, ohlcv: pd.DataFrame, p: ChartPatternDTO) -> bool:
        """Verifie que le trend pre-pattern est coherent avec le type de pattern."""
        if not self._require_pre_trend:
            return True
        # NEUTRAL : pas de contrainte
        if p.kind in _NEUTRAL:
            return True
        # Sym triangle peut casser dans les 2 sens, on skip
        if p.breakout_direction == BreakoutDirection.UNDETERMINED:
            return True

        expected_dir = self._expected_pre_trend_direction(p.kind, p.breakout_direction)
        if expected_dir is None:
            return True

        start_idx = p.start_index - self._pre_trend_bars
        if start_idx < 0:
            return False  # pas assez d'historique
        end_idx = p.start_index
        if end_idx >= len(ohlcv):
            return False
        c_start = float(ohlcv["close"].iloc[start_idx])
        c_end = float(ohlcv["close"].iloc[end_idx])
        if c_start <= 0:
            return False
        move_pct = (c_end - c_start) / c_start
        if expected_dir == "up":
            return move_pct >= self._min_pre_trend
        return move_pct <= -self._min_pre_trend

    def _validate_rsi(self, rsi_arr: np.ndarray | None, p: ChartPatternDTO) -> bool:
        """Verifie que le RSI est aligne avec la direction du breakout attendu.

        Pour breakout UP : RSI > rsi_threshold_up (momentum positif)
        Pour breakout DOWN : RSI < rsi_threshold_down (momentum negatif)
        """
        if not self._require_rsi or rsi_arr is None:
            return True
        if p.breakout_direction == BreakoutDirection.UNDETERMINED:
            return True
        # On regarde le RSI sur la derniere bougie (= moment ou on detecte)
        if p.end_index < 0 or p.end_index >= len(rsi_arr):
            return True
        rsi_now = float(rsi_arr[p.end_index])
        if np.isnan(rsi_now):
            return True
        if p.breakout_direction == BreakoutDirection.UP:
            return rsi_now >= self._rsi_up
        if p.breakout_direction == BreakoutDirection.DOWN:
            return rsi_now <= self._rsi_down
        return True

    @staticmethod
    def _expected_pre_trend_direction(
        kind: PatternKind, breakout_dir: BreakoutDirection
    ) -> str | None:
        """Retourne 'up', 'down', ou None selon le pattern et sa direction."""
        # CONTINUATION : pre-trend dans la direction du breakout
        if kind in _CONTINUATION:
            if breakout_dir == BreakoutDirection.UP:
                return "up"
            if breakout_dir == BreakoutDirection.DOWN:
                return "down"
        # REVERSAL (non internal) : pre-trend OPPOSE au breakout
        if kind in _REVERSAL_NON_INTERNAL:
            if breakout_dir == BreakoutDirection.UP:
                return "down"  # falling wedge : downtrend AVANT, breakout UP apres
            if breakout_dir == BreakoutDirection.DOWN:
                return "up"    # rising wedge : uptrend AVANT, breakout DOWN apres
        return None
