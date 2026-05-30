"""Detecteur Cup & Handle (William O'Neil) + version inversee.

CUP_AND_HANDLE (bullish continuation/reversal) :
    Geometrie en "U" suivi d'un mini-drawback (handle).
    - Left rim et right rim au meme niveau (resistance)
    - Cup : declin progressif puis remontee en forme de U (pas V)
    - Handle : petite consolidation/pullback de 10-50% de la profondeur cup
    - Breakout = close au-dessus du right rim
    - Target = depth du cup ajoute au breakout

INVERSE_CUP_AND_HANDLE (bearish, miroir) :
    Geometrie en "n" inverse, handle qui monte legerement.
    - Breakout = close en-dessous du right rim
    - Target = depth ajoute en bas

DETECTION :
- Recherche d'une "valley" entre 2 sommets quasi-egaux (left/right rim)
- Verification que la valley est U-shape (smooth, pas dechiree)
- Puis recherche du handle apres le right rim

Cup minimum 20 bars, max 100. Handle 5-25 bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.patterns._indicators import atr_at, compute_atr
from app.schemas.domain import SwingKind, SwingPoint
from app.schemas.patterns import (
    BreakoutDirection, ChartPatternDTO, PatternKind,
)

# Durcissement post-mesure : 35% de faux positifs sur random walk (20 seeds)
# avec les anciens defauts. Recalibre depuis les donnees live 24h :
#  - depth 8-10% = 20.9% WR (sweet spot), 4-8% = 15% WR, >10% = degrade (BEAR)
#  - smoothness winners median 0.99% vs losers 1.33% : seuil 1.8%
#  - cup_bars winners median 52.5 : maintient max 60
#  - handle_retrace : winners ont en fait des handles legerement plus profonds (32% vs 24%) -> seuil 35% maintenu
#  - cup plus courte max (60 vs 100) : reduit la combinatoire de 80x22=1760 a 40x16=640
_DEFAULT_CUP_MIN_BARS = 25
_DEFAULT_CUP_MAX_BARS = 60
_DEFAULT_CUP_DEPTH_MIN_PCT = 0.08      # 8% min (etait 5%) - elimine 80% des faux signaux, garde le bucket gagnant
_DEFAULT_CUP_DEPTH_MAX_PCT = 0.40      # cup <= 40% (etait 50%)
_DEFAULT_RIM_TOL_PCT = 0.015           # 1.5% tolerance entre left et right rim (etait 3%)
_DEFAULT_HANDLE_MIN_BARS = 4
_DEFAULT_HANDLE_MAX_BARS = 20
_DEFAULT_HANDLE_MAX_RETRACE = 0.35     # handle <= 35% (etait 50%)
_DEFAULT_U_SHAPE_TOLERANCE = 0.18      # smoothness plus strict (etait 0.35)
# Buffer ATR pour le SL : un SL purement structurel (handle_low/high) se fait wick
# par le bruit. Donnees live : Cup&Handle a SL moyen 2.3% et 14% WR ; on elargit
# de 1.0 ATR par defaut pour absorber le bruit.
_DEFAULT_SL_ATR_MULT = 1.0
_DEFAULT_ATR_PERIOD = 14


class CupHandleDetector:
    """Detecte CUP_AND_HANDLE et INVERSE_CUP_AND_HANDLE."""

    def __init__(
        self,
        *,
        cup_min_bars: int = _DEFAULT_CUP_MIN_BARS,
        cup_max_bars: int = _DEFAULT_CUP_MAX_BARS,
        cup_depth_min_pct: float = _DEFAULT_CUP_DEPTH_MIN_PCT,
        cup_depth_max_pct: float = _DEFAULT_CUP_DEPTH_MAX_PCT,
        rim_tol_pct: float = _DEFAULT_RIM_TOL_PCT,
        handle_min_bars: int = _DEFAULT_HANDLE_MIN_BARS,
        handle_max_bars: int = _DEFAULT_HANDLE_MAX_BARS,
        handle_max_retrace: float = _DEFAULT_HANDLE_MAX_RETRACE,
        u_shape_tolerance: float = _DEFAULT_U_SHAPE_TOLERANCE,
        sl_atr_mult: float = _DEFAULT_SL_ATR_MULT,
        atr_period: int = _DEFAULT_ATR_PERIOD,
    ) -> None:
        self._cup_min = cup_min_bars
        self._cup_max = cup_max_bars
        self._depth_min = cup_depth_min_pct
        self._depth_max = cup_depth_max_pct
        self._rim_tol = rim_tol_pct
        self._h_min = handle_min_bars
        self._h_max = handle_max_bars
        self._h_retrace = handle_max_retrace
        self._u_tol = u_shape_tolerance
        self._sl_atr_mult = float(sl_atr_mult)
        self._atr_period = int(atr_period)

    def _atr_buffered_invalidation(
        self, ohlcv: pd.DataFrame, raw_invalidation: float, side_long: bool
    ) -> float:
        """Elargit le SL d'un multiple d'ATR pour absorber le bruit.

        Si side_long=True : invalidation finale = raw - k*ATR.
        Sinon : raw + k*ATR.
        """
        if self._sl_atr_mult <= 0:
            return raw_invalidation
        atr_series = compute_atr(ohlcv, period=self._atr_period)
        last_close = float(ohlcv["close"].iloc[-1])
        atr_val = atr_at(atr_series, len(ohlcv) - 1, fallback_pct_of=last_close)
        if atr_val <= 0:
            return raw_invalidation
        buf = atr_val * self._sl_atr_mult
        return raw_invalidation - buf if side_long else raw_invalidation + buf

    def detect(
        self, ohlcv: pd.DataFrame, swings: list[SwingPoint],
        *, symbol: str, timeframe: str,
    ) -> list[ChartPatternDTO]:
        n = len(ohlcv)
        if n < self._cup_min + self._h_min + 5:
            return []
        out: list[ChartPatternDTO] = []
        out.extend(self._detect_bullish(ohlcv, symbol, timeframe))
        out.extend(self._detect_bearish(ohlcv, symbol, timeframe))
        return out

    def _detect_bullish(self, ohlcv, symbol, timeframe):
        n = len(ohlcv)
        last_idx = n - 1
        highs = ohlcv["high"].to_numpy(dtype=float)
        lows = ohlcv["low"].to_numpy(dtype=float)
        closes = ohlcv["close"].to_numpy(dtype=float)
        last_close = float(closes[-1])
        if last_close <= 0:
            return []

        # 1. Cherche un handle qui se termine maintenant (handle_end = last_idx)
        # On essaie differentes longueurs de handle
        for handle_len in range(self._h_min, self._h_max + 1):
            handle_start = last_idx - handle_len
            if handle_start < self._cup_min:
                break
            # Right rim = juste avant le handle
            right_rim_idx = handle_start
            right_rim = float(highs[right_rim_idx])
            if right_rim <= 0:
                continue

            # Handle : doit etre un pullback modere (descendant ou plat)
            handle_low = float(lows[handle_start: last_idx + 1].min())
            handle_high = float(highs[handle_start: last_idx + 1].max())
            if handle_high > right_rim * 1.005:
                # Handle ne doit pas casser le right rim avant le breakout
                continue

            # 2. Cherche le left rim (sommet au meme niveau, plus en arriere)
            for cup_len in range(self._cup_min, self._cup_max + 1):
                left_rim_idx = right_rim_idx - cup_len
                if left_rim_idx < 0:
                    break
                left_rim = float(highs[left_rim_idx])
                if left_rim <= 0:
                    continue
                # Les 2 rims doivent etre proches
                if abs(left_rim - right_rim) / max(1e-9, max(left_rim, right_rim)) > self._rim_tol:
                    continue
                avg_rim = (left_rim + right_rim) / 2.0

                # 3. Verifie la profondeur du cup
                cup_low = float(lows[left_rim_idx: right_rim_idx + 1].min())
                depth = avg_rim - cup_low
                depth_pct = depth / avg_rim
                if depth_pct < self._depth_min or depth_pct > self._depth_max:
                    continue

                # 4. Verifie le retracement du handle
                handle_retrace = (right_rim - handle_low) / depth
                if handle_retrace > self._h_retrace:
                    continue

                # 5. Verifie la "U-shape" du cup
                # Un cup propre = le low est dans le tiers MILIEU temporel,
                # et la descente/remontee est progressive (pas un V abrupte)
                cup_low_idx = int(left_rim_idx + np.argmin(lows[left_rim_idx: right_rim_idx + 1]))
                middle_start = left_rim_idx + cup_len // 3
                middle_end = right_rim_idx - cup_len // 3
                if not (middle_start <= cup_low_idx <= middle_end):
                    continue

                # Smoothness : standard deviation des lows dans le tiers du fond
                bottom_slice = lows[middle_start: middle_end + 1]
                if len(bottom_slice) < 3:
                    continue
                bottom_std = float(np.std(bottom_slice))
                bottom_mean = float(bottom_slice.mean())
                if bottom_mean <= 0:
                    continue
                smoothness = bottom_std / bottom_mean
                if smoothness > self._u_tol * 0.1:  # tolerance smoothness
                    continue

                # 6. Pattern non encore casse
                if last_close > right_rim * 1.01:
                    continue  # deja casse au-dessus, trop tard pour entrer
                if last_close < cup_low * 0.97:
                    continue  # passe sous le low du cup = invalide

                # OK, pattern detecte
                breakout_level = right_rim
                # SL : handle_low elargi d'un buffer ATR (sinon wick systematique).
                invalidation = self._atr_buffered_invalidation(
                    ohlcv, handle_low, side_long=True
                )
                target = right_rim + depth  # target classique = depth ajoute au breakout

                confidence = self._score_cup(depth_pct, handle_retrace, smoothness)
                return [ChartPatternDTO(
                    kind=PatternKind.CUP_AND_HANDLE,
                    symbol=symbol, timeframe=timeframe,
                    start_index=left_rim_idx, end_index=last_idx,
                    start_timestamp=ohlcv["timestamp"].iloc[left_rim_idx],
                    end_timestamp=ohlcv["timestamp"].iloc[last_idx],
                    breakout_level=breakout_level,
                    invalidation_level=invalidation,
                    breakout_direction=BreakoutDirection.UP,
                    height=depth,
                    target=target,
                    confidence=confidence,
                    payload={
                        "left_rim": (left_rim_idx, left_rim),
                        "right_rim": (right_rim_idx, right_rim),
                        "cup_low": (cup_low_idx, cup_low),
                        "cup_depth_pct": round(depth_pct * 100, 2),
                        "cup_bars": cup_len,
                        "handle_bars": handle_len,
                        "handle_retrace_pct": round(handle_retrace * 100, 1),
                        "smoothness": round(smoothness * 100, 3),
                    },
                )]
        return []

    def _detect_bearish(self, ohlcv, symbol, timeframe):
        """Inverse cup & handle : 'n-shape' avec handle qui monte legerement."""
        n = len(ohlcv)
        last_idx = n - 1
        highs = ohlcv["high"].to_numpy(dtype=float)
        lows = ohlcv["low"].to_numpy(dtype=float)
        closes = ohlcv["close"].to_numpy(dtype=float)
        last_close = float(closes[-1])
        if last_close <= 0:
            return []

        for handle_len in range(self._h_min, self._h_max + 1):
            handle_start = last_idx - handle_len
            if handle_start < self._cup_min:
                break
            right_rim_idx = handle_start
            right_rim = float(lows[right_rim_idx])  # bas du n inverse
            if right_rim <= 0:
                continue

            handle_high = float(highs[handle_start: last_idx + 1].max())
            handle_low = float(lows[handle_start: last_idx + 1].min())
            if handle_low < right_rim * 0.995:
                continue  # handle casse le rim avant breakout

            for cup_len in range(self._cup_min, self._cup_max + 1):
                left_rim_idx = right_rim_idx - cup_len
                if left_rim_idx < 0:
                    break
                left_rim = float(lows[left_rim_idx])
                if left_rim <= 0:
                    continue
                if abs(left_rim - right_rim) / max(1e-9, max(left_rim, right_rim)) > self._rim_tol:
                    continue
                avg_rim = (left_rim + right_rim) / 2.0

                cup_high = float(highs[left_rim_idx: right_rim_idx + 1].max())
                depth = cup_high - avg_rim
                depth_pct = depth / avg_rim
                if depth_pct < self._depth_min or depth_pct > self._depth_max:
                    continue

                handle_retrace = (handle_high - right_rim) / depth
                if handle_retrace > self._h_retrace:
                    continue

                cup_high_idx = int(left_rim_idx + np.argmax(highs[left_rim_idx: right_rim_idx + 1]))
                middle_start = left_rim_idx + cup_len // 3
                middle_end = right_rim_idx - cup_len // 3
                if not (middle_start <= cup_high_idx <= middle_end):
                    continue

                top_slice = highs[middle_start: middle_end + 1]
                if len(top_slice) < 3:
                    continue
                top_std = float(np.std(top_slice))
                top_mean = float(top_slice.mean())
                if top_mean <= 0:
                    continue
                smoothness = top_std / top_mean
                if smoothness > self._u_tol * 0.1:
                    continue

                if last_close < right_rim * 0.99:
                    continue
                if last_close > cup_high * 1.03:
                    continue

                breakout_level = right_rim
                # SL : handle_high elargi d'un buffer ATR (anti-wick).
                invalidation = self._atr_buffered_invalidation(
                    ohlcv, handle_high, side_long=False
                )
                target = right_rim - depth

                confidence = self._score_cup(depth_pct, handle_retrace, smoothness)
                return [ChartPatternDTO(
                    kind=PatternKind.INVERSE_CUP_AND_HANDLE,
                    symbol=symbol, timeframe=timeframe,
                    start_index=left_rim_idx, end_index=last_idx,
                    start_timestamp=ohlcv["timestamp"].iloc[left_rim_idx],
                    end_timestamp=ohlcv["timestamp"].iloc[last_idx],
                    breakout_level=breakout_level,
                    invalidation_level=invalidation,
                    breakout_direction=BreakoutDirection.DOWN,
                    height=depth,
                    target=target,
                    confidence=confidence,
                    payload={
                        "left_rim": (left_rim_idx, left_rim),
                        "right_rim": (right_rim_idx, right_rim),
                        "cup_high": (cup_high_idx, cup_high),
                        "cup_depth_pct": round(depth_pct * 100, 2),
                        "cup_bars": cup_len,
                        "handle_bars": handle_len,
                        "handle_retrace_pct": round(handle_retrace * 100, 1),
                        "smoothness": round(smoothness * 100, 3),
                    },
                )]
        return []

    def _score_cup(self, depth_pct: float, handle_retrace: float,
                    smoothness: float) -> float:
        # Bon cup : depth 15-25%, handle <30% retrace, smoothness < 0.02
        depth_score = 1.0 - min(1.0, abs(depth_pct - 0.20) / 0.20)
        handle_score = 1.0 - min(1.0, handle_retrace / 0.5)
        smooth_score = 1.0 - min(1.0, smoothness / 0.035)
        score = 0.4 * depth_score + 0.3 * handle_score + 0.3 * smooth_score
        return round(min(1.0, max(0.0, score)), 3)
