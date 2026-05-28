"""Détecteurs de patterns de retournement : Double Top / Bottom + H&S / iH&S.

Géométrie :
    Double top (bearish reversal) :
        - 2 swing highs proches en prix (|Δ| <= ``twin_tol_pct``)
        - 1 swing low entre les deux = neckline
        - Cassure attendue sous la neckline → DOWN
        - Target = (avg_highs − neckline) projetée sous la neckline
        - Invalidation = nouveau plus haut au-dessus des sommets

    Double bottom : miroir.

    Head & Shoulders (bearish) :
        - 3 swing highs avec head > épaules ; épaules à peu près au même prix
        - 2 swing lows entre eux = neckline (régression linéaire entre les 2 lows)
        - Cassure attendue sous la neckline
        - Target = (head − neckline_at_break) projeté sous la neckline

    Inverse H&S : miroir.
"""

from __future__ import annotations

import pandas as pd

from app.patterns._geometry import fit_line
from app.schemas.domain import SwingKind, SwingPoint
from app.schemas.patterns import (
    BreakoutDirection,
    ChartPatternDTO,
    PatternKind,
    TrendLine,
)

_DEFAULT_WINDOW_BARS = 120
_DEFAULT_TWIN_TOL_PCT = 0.02            # garde permissif pour avoir un volume
_DEFAULT_SHOULDER_TOL_PCT = 0.04
_DEFAULT_MIN_HEAD_PROMINENCE_PCT = 0.015
_DEFAULT_MIN_NECK_DISTANCE_PCT = 0.015  # 1.5% (original)
_DEFAULT_NECK_BUFFER_PCT = 0.002
# Optimisations issues de l'analyse MFE/MAE (run_loop, mai 2026) :
# - Winners atteignent 121-138% du target initial -> target_multiplier=1.2 pour ameliorer RR
# - SL legerement resserre pour ameliorer R:R sans tuer trigger rate
_DEFAULT_TARGET_MULTIPLIER = 1.3        # winners font 121-138% du target naturel
_DEFAULT_SL_TIGHTEN_PCT = 0.0           # SL naturel = invalidation pattern (trade actif plus longtemps)


class ReversalDetector:
    def __init__(
        self,
        *,
        window_bars: int = _DEFAULT_WINDOW_BARS,
        twin_tol_pct: float = _DEFAULT_TWIN_TOL_PCT,
        shoulder_tol_pct: float = _DEFAULT_SHOULDER_TOL_PCT,
        min_head_prominence_pct: float = _DEFAULT_MIN_HEAD_PROMINENCE_PCT,
        min_neck_distance_pct: float = _DEFAULT_MIN_NECK_DISTANCE_PCT,
        neck_buffer_pct: float = _DEFAULT_NECK_BUFFER_PCT,
        target_multiplier: float = _DEFAULT_TARGET_MULTIPLIER,
        sl_tighten_pct: float = _DEFAULT_SL_TIGHTEN_PCT,
    ) -> None:
        self._window = window_bars
        self._twin_tol = twin_tol_pct
        self._shoulder_tol = shoulder_tol_pct
        self._head_prom = min_head_prominence_pct
        self._min_neck = min_neck_distance_pct
        self._neck_buf = neck_buffer_pct
        self._target_mult = float(target_multiplier)
        self._sl_tighten = float(sl_tighten_pct)

    def _tighten_sl(self, entry: float, raw_invalidation: float) -> float:
        """Resserre le SL vers entry de sl_tighten_pct (0.30 = 30% plus proche).
        Garde toujours le SL au-dessus de entry pour SHORT / au-dessous pour LONG."""
        if self._sl_tighten <= 0:
            return raw_invalidation
        sl_distance = raw_invalidation - entry
        return entry + sl_distance * (1.0 - self._sl_tighten)

    def detect(
        self,
        ohlcv: pd.DataFrame,
        swings: list[SwingPoint],
        *,
        symbol: str,
        timeframe: str,
    ) -> list[ChartPatternDTO]:
        n = len(ohlcv)
        if n < 20 or len(swings) < 3:
            return []
        last_idx = n - 1
        start_window = max(0, last_idx - self._window)
        recent = sorted([s for s in swings if start_window <= s.index <= last_idx], key=lambda s: s.index)

        out: list[ChartPatternDTO] = []
        out.extend(self._detect_double_top(ohlcv, recent, symbol, timeframe))
        out.extend(self._detect_double_bottom(ohlcv, recent, symbol, timeframe))
        out.extend(self._detect_hs(ohlcv, recent, symbol, timeframe))
        out.extend(self._detect_ihs(ohlcv, recent, symbol, timeframe))
        return out

    # ------------------------------------------------------------------
    # Double top
    # ------------------------------------------------------------------
    def _detect_double_top(self, ohlcv, swings, symbol, timeframe) -> list[ChartPatternDTO]:
        last_close = float(ohlcv["close"].iloc[-1])
        if last_close <= 0:
            return []
        last_idx = len(ohlcv) - 1
        highs = [s for s in swings if s.kind == SwingKind.HIGH]
        if len(highs) < 2:
            return []
        h2 = highs[-1]
        h1 = highs[-2]
        avg = (h1.price + h2.price) / 2.0
        if abs(h1.price - h2.price) / avg > self._twin_tol:
            return []
        between_lows = [s for s in swings if s.kind == SwingKind.LOW and h1.index < s.index < h2.index]
        if not between_lows:
            return []
        neckline = min(between_lows, key=lambda s: s.price)
        if (avg - neckline.price) / avg < self._min_neck:
            return []
        # Pattern non encore cassé : close > neckline (avec un peu de tolérance)
        if last_close < neckline.price * (1.0 - self._neck_buf):
            return []
        # Et close pas au-dessus du double top
        if last_close > avg * (1.0 + self._twin_tol):
            return []

        height = avg - neckline.price
        # SL = max(h1,h2) resserre vers entree pour ameliorer RR
        raw_inv = max(h1.price, h2.price)
        tightened_inv = self._tighten_sl(neckline.price, raw_inv)
        # Target etendu pour capturer les winners qui font 138% du target naturel
        target_extended = neckline.price - height * self._target_mult
        confidence = _score_twin(h1.price, h2.price, neckline.price, last_close, avg)
        return [ChartPatternDTO(
            kind=PatternKind.DOUBLE_TOP,
            symbol=symbol,
            timeframe=timeframe,
            start_index=h1.index,
            end_index=last_idx,
            start_timestamp=ohlcv["timestamp"].iloc[h1.index],
            end_timestamp=ohlcv["timestamp"].iloc[last_idx],
            breakout_level=neckline.price,
            invalidation_level=tightened_inv,
            breakout_direction=BreakoutDirection.DOWN,
            height=height,
            target=target_extended,
            confidence=confidence,
            payload={
                "high1": (h1.index, h1.price),
                "high2": (h2.index, h2.price),
                "neckline_price": neckline.price,
                "neckline_index": neckline.index,
            },
        )]

    # ------------------------------------------------------------------
    # Double bottom (miroir)
    # ------------------------------------------------------------------
    # NOTE optimisation : conditions identiques a DOUBLE_TOP (les ecarts
    # rentabilite viennent de la confluence trend HTF, pas de la geometrie).
    def _detect_double_bottom(self, ohlcv, swings, symbol, timeframe) -> list[ChartPatternDTO]:
        last_close = float(ohlcv["close"].iloc[-1])
        if last_close <= 0:
            return []
        last_idx = len(ohlcv) - 1
        lows = [s for s in swings if s.kind == SwingKind.LOW]
        if len(lows) < 2:
            return []
        l2 = lows[-1]
        l1 = lows[-2]
        avg = (l1.price + l2.price) / 2.0
        if abs(l1.price - l2.price) / avg > self._twin_tol:
            return []
        between_highs = [s for s in swings if s.kind == SwingKind.HIGH and l1.index < s.index < l2.index]
        if not between_highs:
            return []
        neckline = max(between_highs, key=lambda s: s.price)
        if (neckline.price - avg) / avg < self._min_neck:
            return []
        if last_close > neckline.price * (1.0 + self._neck_buf):
            return []
        if last_close < avg * (1.0 - self._twin_tol):
            return []
        height = neckline.price - avg
        raw_inv = min(l1.price, l2.price)
        tightened_inv = self._tighten_sl(neckline.price, raw_inv)
        target_extended = neckline.price + height * self._target_mult
        confidence = _score_twin(l1.price, l2.price, neckline.price, last_close, avg)
        return [ChartPatternDTO(
            kind=PatternKind.DOUBLE_BOTTOM,
            symbol=symbol,
            timeframe=timeframe,
            start_index=l1.index,
            end_index=last_idx,
            start_timestamp=ohlcv["timestamp"].iloc[l1.index],
            end_timestamp=ohlcv["timestamp"].iloc[last_idx],
            breakout_level=neckline.price,
            invalidation_level=tightened_inv,
            breakout_direction=BreakoutDirection.UP,
            height=height,
            target=target_extended,
            confidence=confidence,
            payload={
                "low1": (l1.index, l1.price),
                "low2": (l2.index, l2.price),
                "neckline_price": neckline.price,
                "neckline_index": neckline.index,
            },
        )]

    # ------------------------------------------------------------------
    # Head & Shoulders (bearish)
    # ------------------------------------------------------------------
    def _detect_hs(self, ohlcv, swings, symbol, timeframe) -> list[ChartPatternDTO]:
        last_close = float(ohlcv["close"].iloc[-1])
        if last_close <= 0:
            return []
        last_idx = len(ohlcv) - 1
        highs = [s for s in swings if s.kind == SwingKind.HIGH]
        if len(highs) < 3:
            return []
        ls, head, rs = highs[-3], highs[-2], highs[-1]
        # Head doit être plus haut que les épaules d'au moins X%
        if head.price <= ls.price * (1.0 + self._head_prom):
            return []
        if head.price <= rs.price * (1.0 + self._head_prom):
            return []
        # Épaules proches
        avg_sh = (ls.price + rs.price) / 2.0
        if abs(ls.price - rs.price) / avg_sh > self._shoulder_tol:
            return []
        # Neckline = régression sur les 2 lows entre les sommets
        neck_lows = [
            s for s in swings
            if s.kind == SwingKind.LOW and ls.index < s.index < rs.index
        ]
        if len(neck_lows) < 2:
            return []
        nl1 = min((s for s in neck_lows if s.index < head.index), default=None, key=lambda s: s.price)
        nl2 = min((s for s in neck_lows if s.index > head.index), default=None, key=lambda s: s.price)
        if nl1 is None or nl2 is None:
            return []
        line = fit_line([nl1.index, nl2.index], [nl1.price, nl2.price])
        if line is None:
            return []
        neck_now = float(line.value_at(last_idx))
        if (head.price - neck_now) / head.price < self._min_neck:
            return []
        if last_close < neck_now * (1.0 - self._neck_buf):
            return []
        if last_close > head.price:
            return []
        height = head.price - neck_now
        tightened_inv = self._tighten_sl(neck_now, head.price)
        target_extended = neck_now - height * self._target_mult
        confidence = _score_hs(ls.price, head.price, rs.price, neck_now, head.price)
        return [ChartPatternDTO(
            kind=PatternKind.HEAD_SHOULDERS,
            symbol=symbol,
            timeframe=timeframe,
            start_index=ls.index,
            end_index=last_idx,
            start_timestamp=ohlcv["timestamp"].iloc[ls.index],
            end_timestamp=ohlcv["timestamp"].iloc[last_idx],
            breakout_level=neck_now,
            invalidation_level=tightened_inv,
            breakout_direction=BreakoutDirection.DOWN,
            height=height,
            target=target_extended,
            lower_line=line,
            confidence=confidence,
            payload={
                "left_shoulder": (ls.index, ls.price),
                "head": (head.index, head.price),
                "right_shoulder": (rs.index, rs.price),
                "neckline_left": (nl1.index, nl1.price),
                "neckline_right": (nl2.index, nl2.price),
            },
        )]

    # ------------------------------------------------------------------
    # Inverse H&S (bullish)
    # ------------------------------------------------------------------
    def _detect_ihs(self, ohlcv, swings, symbol, timeframe) -> list[ChartPatternDTO]:
        last_close = float(ohlcv["close"].iloc[-1])
        if last_close <= 0:
            return []
        last_idx = len(ohlcv) - 1
        lows = [s for s in swings if s.kind == SwingKind.LOW]
        if len(lows) < 3:
            return []
        ls, head, rs = lows[-3], lows[-2], lows[-1]
        if head.price >= ls.price * (1.0 - self._head_prom):
            return []
        if head.price >= rs.price * (1.0 - self._head_prom):
            return []
        avg_sh = (ls.price + rs.price) / 2.0
        if abs(ls.price - rs.price) / avg_sh > self._shoulder_tol:
            return []
        neck_highs = [
            s for s in swings
            if s.kind == SwingKind.HIGH and ls.index < s.index < rs.index
        ]
        if len(neck_highs) < 2:
            return []
        nh1 = max((s for s in neck_highs if s.index < head.index), default=None, key=lambda s: s.price)
        nh2 = max((s for s in neck_highs if s.index > head.index), default=None, key=lambda s: s.price)
        if nh1 is None or nh2 is None:
            return []
        line = fit_line([nh1.index, nh2.index], [nh1.price, nh2.price])
        if line is None:
            return []
        neck_now = float(line.value_at(last_idx))
        if (neck_now - head.price) / neck_now < self._min_neck:
            return []
        if last_close > neck_now * (1.0 + self._neck_buf):
            return []
        if last_close < head.price:
            return []
        height = neck_now - head.price
        tightened_inv = self._tighten_sl(neck_now, head.price)
        target_extended = neck_now + height * self._target_mult
        confidence = _score_hs(ls.price, head.price, rs.price, neck_now, head.price)
        return [ChartPatternDTO(
            kind=PatternKind.INVERSE_HEAD_SHOULDERS,
            symbol=symbol,
            timeframe=timeframe,
            start_index=ls.index,
            end_index=last_idx,
            start_timestamp=ohlcv["timestamp"].iloc[ls.index],
            end_timestamp=ohlcv["timestamp"].iloc[last_idx],
            breakout_level=neck_now,
            invalidation_level=tightened_inv,
            breakout_direction=BreakoutDirection.UP,
            height=height,
            target=target_extended,
            upper_line=line,
            confidence=confidence,
            payload={
                "left_shoulder": (ls.index, ls.price),
                "head": (head.index, head.price),
                "right_shoulder": (rs.index, rs.price),
                "neckline_left": (nh1.index, nh1.price),
                "neckline_right": (nh2.index, nh2.price),
            },
        )]


def _score_twin(p1: float, p2: float, neck: float, last: float, avg: float) -> float:
    sym = 1.0 - min(1.0, abs(p1 - p2) / max(1e-9, avg) / 0.05)
    height_pct = abs(avg - neck) / avg if avg > 0 else 0.0
    height_bonus = min(1.0, height_pct / 0.10)
    score = 0.6 * sym + 0.4 * height_bonus
    return round(min(1.0, max(0.0, score)), 3)


def _score_hs(ls: float, head: float, rs: float, neck: float, ref: float) -> float:
    avg_sh = (ls + rs) / 2.0
    sym = 1.0 - min(1.0, abs(ls - rs) / max(1e-9, avg_sh) / 0.05)
    prom = min(1.0, (head - avg_sh) / max(1e-9, avg_sh) / 0.10)
    score = 0.5 * sym + 0.5 * prom
    return round(min(1.0, max(0.0, score)), 3)
