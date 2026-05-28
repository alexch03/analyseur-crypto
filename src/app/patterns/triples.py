"""Detecteurs Triple Top / Triple Bottom (reversal patterns).

TRIPLE TOP (bearish reversal) :
    - 3 swing highs proches en prix (|max-min| / avg < twin_tol)
    - 2 swing lows entre eux = neckline (approximation par le min)
    - Cassure attendue sous la neckline -> DOWN
    - Target = (avg_highs - neckline) projetee sous la neckline
    - Invalidation = nouveau plus haut au-dessus des 3 sommets

TRIPLE BOTTOM (bullish) : miroir.

NB : Si on a 3 highs/lows mais qu'ils sont "tres alignes" (head distinct),
on bascule plutot sur HEAD_SHOULDERS. Les triples sont rigoureusement
3 sommets alignes au meme niveau.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.patterns._indicators import (
    bearish_rsi_div_on_tops, bullish_rsi_div_on_bottoms,
    compute_rsi, volume_accumulation_on_bottoms, volume_exhaustion_on_tops,
)
from app.schemas.domain import SwingKind, SwingPoint
from app.schemas.patterns import (
    BreakoutDirection, ChartPatternDTO, PatternKind,
)

_DEFAULT_WINDOW_BARS = 150
_DEFAULT_TWIN_TOL_PCT = 0.025          # 2.5% écart max entre les 3 sommets/creux
_DEFAULT_MIN_NECK_DISTANCE_PCT = 0.02  # neck-to-top >= 2%
_DEFAULT_NECK_BUFFER_PCT = 0.002
_DEFAULT_MIN_BARS_BETWEEN = 5          # min entre chaque sommet/creux
_DEFAULT_REQUIRE_RSI_DIVERGENCE = True  # exige div RSI entre top1 et top3
_DEFAULT_MIN_RSI_DIV_POINTS = 2.0


class TripleDetector:
    """Detecte Triple Top et Triple Bottom."""

    def __init__(
        self,
        *,
        window_bars: int = _DEFAULT_WINDOW_BARS,
        twin_tol_pct: float = _DEFAULT_TWIN_TOL_PCT,
        min_neck_distance_pct: float = _DEFAULT_MIN_NECK_DISTANCE_PCT,
        neck_buffer_pct: float = _DEFAULT_NECK_BUFFER_PCT,
        min_bars_between: int = _DEFAULT_MIN_BARS_BETWEEN,
        require_rsi_divergence: bool = _DEFAULT_REQUIRE_RSI_DIVERGENCE,
        min_rsi_div_points: float = _DEFAULT_MIN_RSI_DIV_POINTS,
    ) -> None:
        self._window = window_bars
        self._twin_tol = twin_tol_pct
        self._min_neck = min_neck_distance_pct
        self._neck_buf = neck_buffer_pct
        self._min_bars = min_bars_between
        self._require_rsi_div = require_rsi_divergence
        self._min_rsi_div = min_rsi_div_points

    def detect(
        self, ohlcv: pd.DataFrame, swings: list[SwingPoint],
        *, symbol: str, timeframe: str,
    ) -> list[ChartPatternDTO]:
        n = len(ohlcv)
        if n < 30 or len(swings) < 4:
            return []
        last_idx = n - 1
        start = max(0, last_idx - self._window)
        recent = sorted(
            [s for s in swings if start <= s.index <= last_idx],
            key=lambda s: s.index,
        )
        # Precalcule RSI pour divergence
        rsi_arr = None
        if self._require_rsi_div:
            try:
                rsi_arr = compute_rsi(ohlcv["close"], period=14)
            except Exception:
                pass

        out: list[ChartPatternDTO] = []
        out.extend(self._detect_top(ohlcv, recent, symbol, timeframe, rsi_arr))
        out.extend(self._detect_bottom(ohlcv, recent, symbol, timeframe, rsi_arr))
        return out

    def _swings_spaced(self, *swings: SwingPoint) -> bool:
        idx = sorted(s.index for s in swings)
        for i in range(len(idx) - 1):
            if idx[i + 1] - idx[i] < self._min_bars:
                return False
        return True

    def _detect_top(self, ohlcv, swings, symbol, timeframe, rsi):
        last_close = float(ohlcv["close"].iloc[-1])
        last_idx = len(ohlcv) - 1
        highs = [s for s in swings if s.kind == SwingKind.HIGH]
        if len(highs) < 3:
            return []
        h1, h2, h3 = highs[-3], highs[-2], highs[-1]
        if not self._swings_spaced(h1, h2, h3):
            return []
        # Les 3 tops doivent etre proches
        prices = [h1.price, h2.price, h3.price]
        avg = sum(prices) / 3.0
        spread = max(prices) - min(prices)
        if spread / max(1e-9, avg) > self._twin_tol:
            return []
        # Pas un HEAD_SHOULDERS deguise : h2 (middle) ne doit pas etre dominant
        if h2.price > max(h1.price, h3.price) + spread * 0.5:
            return []
        # Neckline = min des lows entre h1 et h3
        between_lows = [
            s for s in swings
            if s.kind == SwingKind.LOW and h1.index < s.index < h3.index
        ]
        if len(between_lows) < 2:
            return []
        neckline = min(between_lows, key=lambda s: s.price)
        if (avg - neckline.price) / avg < self._min_neck:
            return []
        # Pattern non encore casse
        if last_close < neckline.price * (1.0 - self._neck_buf):
            return []
        if last_close > avg * (1.0 + self._twin_tol):
            return []
        # Divergence RSI optionnelle : RSI[h3] < RSI[h1]
        rsi_div = False
        if self._require_rsi_div and rsi is not None:
            if not bearish_rsi_div_on_tops(rsi, h1.index, h3.index,
                                            min_rsi_drop=self._min_rsi_div):
                return []
            rsi_div = True

        height = avg - neckline.price
        confidence = self._score_triple(prices, neckline.price, last_close)
        return [ChartPatternDTO(
            kind=PatternKind.TRIPLE_TOP,
            symbol=symbol, timeframe=timeframe,
            start_index=h1.index, end_index=last_idx,
            start_timestamp=ohlcv["timestamp"].iloc[h1.index],
            end_timestamp=ohlcv["timestamp"].iloc[last_idx],
            breakout_level=neckline.price,
            invalidation_level=max(prices),
            breakout_direction=BreakoutDirection.DOWN,
            height=height,
            target=neckline.price - height,
            confidence=confidence,
            payload={
                "top1": (h1.index, h1.price),
                "top2": (h2.index, h2.price),
                "top3": (h3.index, h3.price),
                "neckline_price": neckline.price,
                "rsi_divergence": rsi_div,
            },
        )]

    def _detect_bottom(self, ohlcv, swings, symbol, timeframe, rsi):
        last_close = float(ohlcv["close"].iloc[-1])
        last_idx = len(ohlcv) - 1
        lows = [s for s in swings if s.kind == SwingKind.LOW]
        if len(lows) < 3:
            return []
        l1, l2, l3 = lows[-3], lows[-2], lows[-1]
        if not self._swings_spaced(l1, l2, l3):
            return []
        prices = [l1.price, l2.price, l3.price]
        avg = sum(prices) / 3.0
        spread = max(prices) - min(prices)
        if spread / max(1e-9, avg) > self._twin_tol:
            return []
        if l2.price < min(l1.price, l3.price) - spread * 0.5:
            return []
        between_highs = [
            s for s in swings
            if s.kind == SwingKind.HIGH and l1.index < s.index < l3.index
        ]
        if len(between_highs) < 2:
            return []
        neckline = max(between_highs, key=lambda s: s.price)
        if (neckline.price - avg) / avg < self._min_neck:
            return []
        if last_close > neckline.price * (1.0 + self._neck_buf):
            return []
        if last_close < avg * (1.0 - self._twin_tol):
            return []
        rsi_div = False
        if self._require_rsi_div and rsi is not None:
            if not bullish_rsi_div_on_bottoms(rsi, l1.index, l3.index,
                                                min_rsi_rise=self._min_rsi_div):
                return []
            rsi_div = True

        height = neckline.price - avg
        confidence = self._score_triple(prices, neckline.price, last_close)
        return [ChartPatternDTO(
            kind=PatternKind.TRIPLE_BOTTOM,
            symbol=symbol, timeframe=timeframe,
            start_index=l1.index, end_index=last_idx,
            start_timestamp=ohlcv["timestamp"].iloc[l1.index],
            end_timestamp=ohlcv["timestamp"].iloc[last_idx],
            breakout_level=neckline.price,
            invalidation_level=min(prices),
            breakout_direction=BreakoutDirection.UP,
            height=height,
            target=neckline.price + height,
            confidence=confidence,
            payload={
                "bot1": (l1.index, l1.price),
                "bot2": (l2.index, l2.price),
                "bot3": (l3.index, l3.price),
                "neckline_price": neckline.price,
                "rsi_divergence": rsi_div,
            },
        )]

    def _score_triple(self, prices: list[float], neck: float, last: float) -> float:
        avg = sum(prices) / 3.0
        spread = max(prices) - min(prices)
        sym = 1.0 - min(1.0, spread / max(1e-9, avg) / 0.05)
        height_pct = abs(avg - neck) / max(1e-9, avg)
        height_bonus = min(1.0, height_pct / 0.10)
        score = 0.6 * sym + 0.4 * height_bonus
        return round(min(1.0, max(0.0, score)), 3)
