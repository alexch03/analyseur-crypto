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

Qualité v2 (issue de l'analyse trades) :
    - Trend context PRE-pattern : pour qu'un retournement soit valide, il faut
      qu'il y ait quelque chose à retourner. DOUBLE_TOP = exige uptrend AVANT,
      DOUBLE_BOTTOM = exige downtrend AVANT.
    - Validité des sommets : le swing high doit avoir un close dans le tiers
      supérieur de la bougie (pas juste un wick).
    - Espacement minimum entre swings : evite les patterns trop serrés (noise).
    - Swing prominence : filtre les micro-swings sous ATR * threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.patterns._geometry import atr, fit_line
from app.patterns._indicators import (
    bearish_rsi_div_on_tops,
    bullish_rsi_div_on_bottoms,
    compute_rsi,
    compute_volume_sma,
    compute_vwap_rolling,
    price_above_vwap,
    price_below_vwap,
    volume_accumulation_on_bottoms,
    volume_exhaustion_on_tops,
)
from app.schemas.domain import SwingKind, SwingPoint
from app.schemas.patterns import (
    BreakoutDirection,
    ChartPatternDTO,
    PatternKind,
    TrendLine,
)

# ────────────────────────────────────────────────────────────────────
# Defaults v2 (apres analyse MFE/MAE + amelioration patterns)
# ────────────────────────────────────────────────────────────────────
_DEFAULT_WINDOW_BARS = 120
_DEFAULT_TWIN_TOL_PCT = 0.02
_DEFAULT_SHOULDER_TOL_PCT = 0.04
_DEFAULT_MIN_HEAD_PROMINENCE_PCT = 0.015
_DEFAULT_MIN_NECK_DISTANCE_PCT = 0.015
_DEFAULT_NECK_BUFFER_PCT = 0.002
_DEFAULT_TARGET_MULTIPLIER = 1.3        # winners atteignent 121-138% du target naturel
_DEFAULT_SL_TIGHTEN_PCT = 0.0           # SL naturel = invalidation pattern

# Nouveau : qualite des patterns
_DEFAULT_MIN_BARS_BETWEEN_SWINGS = 5    # espacement min entre les 2 tops/lows (anti-noise)
_DEFAULT_MIN_SWING_PROMINENCE_ATR = 0.5 # swing doit etre >= 0.5x ATR (filtre micro-swings)
_DEFAULT_REQUIRE_PRE_TREND = True       # exige trend context AVANT le pattern
_DEFAULT_PRE_TREND_BARS = 20            # lookback pour mesurer le trend pre-pattern
_DEFAULT_MIN_PRE_TREND_PCT = 0.008      # 0.8% min de move dans le bon sens
_DEFAULT_REQUIRE_SWING_BODY = True      # swing high/low doit avoir un body solide
_DEFAULT_MIN_BODY_RATIO = 0.4           # close du swing dans le tiers superieur/inferieur

# Confluence RSI / VWAP / Volume (v3)
# RSI : divergence requise = baisse de momentum confirmee (excellent signal)
# Volume : OFF par defaut car divise trop le volume sans gain net (88.9% win mais
#          -10% cumul vs RSI seul)
# VWAP : OFF par defaut (peu d'impact additionnel sur 15m crypto majors)
_DEFAULT_REQUIRE_RSI_DIVERGENCE = True  # exige divergence RSI sur sommets/creux
_DEFAULT_MIN_RSI_DIV_POINTS = 2.0       # 2 pts RSI (compromise qualite/volume)
_DEFAULT_REQUIRE_VOLUME_CONFIRMATION = False  # OFF par defaut (trop strict combine a RSI)
_DEFAULT_REQUIRE_VWAP_ALIGNMENT = False # OFF par defaut
_DEFAULT_VWAP_PERIOD = 96               # VWAP rolling sur 1 jour (15m x 96 = 1d)


@dataclass
class _IndicatorContext:
    """Indicateurs precalcules pour eviter le recalcul a chaque detection."""
    rsi: np.ndarray | None = None
    vwap: np.ndarray | None = None
    volumes: np.ndarray | None = None
    last_vwap: float = float("nan")


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
        # Nouveaux params qualite
        min_bars_between_swings: int = _DEFAULT_MIN_BARS_BETWEEN_SWINGS,
        min_swing_prominence_atr: float = _DEFAULT_MIN_SWING_PROMINENCE_ATR,
        require_pre_trend: bool = _DEFAULT_REQUIRE_PRE_TREND,
        pre_trend_bars: int = _DEFAULT_PRE_TREND_BARS,
        min_pre_trend_pct: float = _DEFAULT_MIN_PRE_TREND_PCT,
        require_swing_body: bool = _DEFAULT_REQUIRE_SWING_BODY,
        min_body_ratio: float = _DEFAULT_MIN_BODY_RATIO,
        require_rsi_divergence: bool = _DEFAULT_REQUIRE_RSI_DIVERGENCE,
        min_rsi_divergence_points: float = _DEFAULT_MIN_RSI_DIV_POINTS,
        require_volume_confirmation: bool = _DEFAULT_REQUIRE_VOLUME_CONFIRMATION,
        require_vwap_alignment: bool = _DEFAULT_REQUIRE_VWAP_ALIGNMENT,
        vwap_period: int = _DEFAULT_VWAP_PERIOD,
    ) -> None:
        self._window = window_bars
        self._twin_tol = twin_tol_pct
        self._shoulder_tol = shoulder_tol_pct
        self._head_prom = min_head_prominence_pct
        self._min_neck = min_neck_distance_pct
        self._neck_buf = neck_buffer_pct
        self._target_mult = float(target_multiplier)
        self._sl_tighten = float(sl_tighten_pct)
        self._min_bars_between = int(min_bars_between_swings)
        self._min_swing_prom_atr = float(min_swing_prominence_atr)
        self._require_pre_trend = bool(require_pre_trend)
        self._pre_trend_bars = int(pre_trend_bars)
        self._min_pre_trend = float(min_pre_trend_pct)
        self._require_body = bool(require_swing_body)
        self._min_body_ratio = float(min_body_ratio)
        self._require_rsi_div = bool(require_rsi_divergence)
        self._min_rsi_div_pts = float(min_rsi_divergence_points)
        self._require_vol_conf = bool(require_volume_confirmation)
        self._require_vwap = bool(require_vwap_alignment)
        self._vwap_period = int(vwap_period)

    # ------------------------------------------------------------------
    # Helpers qualite
    # ------------------------------------------------------------------

    def _tighten_sl(self, entry: float, raw_invalidation: float) -> float:
        if self._sl_tighten <= 0:
            return raw_invalidation
        sl_distance = raw_invalidation - entry
        return entry + sl_distance * (1.0 - self._sl_tighten)

    def _get_atr(self, ohlcv: pd.DataFrame) -> float:
        if len(ohlcv) < 15:
            return 0.0
        h = ohlcv["high"].to_numpy(dtype=float)
        l = ohlcv["low"].to_numpy(dtype=float)
        c = ohlcv["close"].to_numpy(dtype=float)
        return atr(h, l, c, period=14)

    def _swing_is_prominent(self, ohlcv: pd.DataFrame, swing: SwingPoint,
                            atr_val: float) -> bool:
        """Le swing doit dépasser ses voisins d'au moins min_prom_atr × ATR."""
        if atr_val <= 0 or self._min_swing_prom_atr <= 0:
            return True
        i = swing.index
        if i < 5 or i >= len(ohlcv) - 5:
            return True  # bord
        local_window = 10
        lo_idx = max(0, i - local_window)
        hi_idx = min(len(ohlcv), i + local_window + 1)
        if swing.kind == SwingKind.HIGH:
            neighbors_max = float(ohlcv["high"].iloc[lo_idx:i].max()) if i > lo_idx else 0
            prominence = swing.price - neighbors_max
        else:
            neighbors_min = float(ohlcv["low"].iloc[lo_idx:i].min()) if i > lo_idx else float("inf")
            prominence = neighbors_min - swing.price
        return prominence >= self._min_swing_prom_atr * atr_val

    def _swing_has_solid_body(self, ohlcv: pd.DataFrame, swing: SwingPoint) -> bool:
        """Le swing high doit avoir close dans le tiers superieur de la bougie
        (pas un long wick). Idem swing low dans le tiers inferieur.

        body_ratio = (close - low) / (high - low) pour LOW
                   = (high - close) / (high - low) pour HIGH (proximite au high)
        On accepte si <= 1 - min_body_ratio (close proche du high pour swing HIGH).
        """
        if not self._require_body or self._min_body_ratio <= 0:
            return True
        i = swing.index
        if i < 0 or i >= len(ohlcv):
            return True
        bar = ohlcv.iloc[i]
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        rng = high - low
        if rng <= 0:
            return True
        if swing.kind == SwingKind.HIGH:
            # Pour HIGH : close doit etre dans le tiers superieur
            # close/high ratio >= 1 - min_body_ratio
            pos_in_bar = (close - low) / rng  # 0=low, 1=high
            return pos_in_bar >= self._min_body_ratio
        else:
            # Pour LOW : close doit etre dans le tiers inferieur
            pos_in_bar = (close - low) / rng
            return pos_in_bar <= 1.0 - self._min_body_ratio

    def _pre_trend_ok(self, ohlcv: pd.DataFrame, swing_index: int,
                      expected: str) -> bool:
        """Verifie qu'il y a un trend AVANT le pattern pour valider le retournement.

        Pour DOUBLE_TOP / HEAD_SHOULDERS (bearish reversal) : exige uptrend avant.
        Pour DOUBLE_BOTTOM / IHS (bullish reversal) : exige downtrend avant.

        expected = "up" ou "down".
        Mesure : (close[swing] - close[swing - pre_trend_bars]) / close[start].
        """
        if not self._require_pre_trend:
            return True
        start_idx = swing_index - self._pre_trend_bars
        if start_idx < 0:
            return False  # pas assez d'historique = on rejette
        close_start = float(ohlcv["close"].iloc[start_idx])
        close_end = float(ohlcv["close"].iloc[swing_index])
        if close_start <= 0:
            return False
        move_pct = (close_end - close_start) / close_start
        if expected == "up":
            return move_pct >= self._min_pre_trend
        else:
            return move_pct <= -self._min_pre_trend

    def _swings_well_spaced(self, sw1: SwingPoint, sw2: SwingPoint) -> bool:
        return abs(sw2.index - sw1.index) >= self._min_bars_between

    # ------------------------------------------------------------------
    # detect (entry point)
    # ------------------------------------------------------------------

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

        # Filtre prominence ATR sur les swings
        atr_val = self._get_atr(ohlcv)
        all_recent = sorted(
            [s for s in swings if start_window <= s.index <= last_idx],
            key=lambda s: s.index,
        )
        recent = [s for s in all_recent if self._swing_is_prominent(ohlcv, s, atr_val)]

        # Pre-calcule RSI/VWAP/Volume_SMA pour toutes les detections (cout amorti)
        rsi_arr = None
        vwap_arr = None
        vol_arr = None
        last_vwap = float("nan")
        if self._require_rsi_div or self._require_vol_conf or self._require_vwap:
            try:
                rsi_arr = compute_rsi(ohlcv["close"], period=14)
                if self._require_vwap:
                    vwap_arr = compute_vwap_rolling(ohlcv, period=self._vwap_period)
                    last_vwap = float(vwap_arr[-1]) if len(vwap_arr) else float("nan")
                if self._require_vol_conf:
                    vol_arr = ohlcv["volume"].to_numpy(dtype=float)
            except Exception:
                pass

        ctx = _IndicatorContext(
            rsi=rsi_arr, vwap=vwap_arr, volumes=vol_arr,
            last_vwap=last_vwap,
        )

        out: list[ChartPatternDTO] = []
        out.extend(self._detect_double_top(ohlcv, recent, symbol, timeframe, ctx))
        out.extend(self._detect_double_bottom(ohlcv, recent, symbol, timeframe, ctx))
        out.extend(self._detect_hs(ohlcv, recent, symbol, timeframe, ctx))
        out.extend(self._detect_ihs(ohlcv, recent, symbol, timeframe, ctx))
        return out

    # ------------------------------------------------------------------
    # Double top (bearish reversal)
    # ------------------------------------------------------------------
    def _detect_double_top(self, ohlcv, swings, symbol, timeframe,
                            ctx: _IndicatorContext) -> list[ChartPatternDTO]:
        last_close = float(ohlcv["close"].iloc[-1])
        if last_close <= 0:
            return []
        last_idx = len(ohlcv) - 1
        highs = [s for s in swings if s.kind == SwingKind.HIGH]
        if len(highs) < 2:
            return []
        h2 = highs[-1]
        h1 = highs[-2]
        # 1. Espacement temporel
        if not self._swings_well_spaced(h1, h2):
            return []
        # 2. Twin tolerance
        avg = (h1.price + h2.price) / 2.0
        if abs(h1.price - h2.price) / avg > self._twin_tol:
            return []
        # 3. Pre-trend uptrend
        if not self._pre_trend_ok(ohlcv, h1.index, "up"):
            return []
        # 4. Validite des sommets
        if not self._swing_has_solid_body(ohlcv, h1):
            return []
        if not self._swing_has_solid_body(ohlcv, h2):
            return []
        # 5. RSI divergence bearish : RSI sur top2 < RSI sur top1
        rsi_div = False
        if self._require_rsi_div and ctx.rsi is not None:
            if not bearish_rsi_div_on_tops(ctx.rsi, h1.index, h2.index,
                                            min_rsi_drop=self._min_rsi_div_pts):
                return []
            rsi_div = True
        # 6. Volume exhaustion : volume sur top2 < volume sur top1
        vol_conf = False
        if self._require_vol_conf and ctx.volumes is not None:
            if not volume_exhaustion_on_tops(ctx.volumes, h1.index, h2.index):
                return []
            vol_conf = True
        # 7. VWAP : prix sous VWAP confirme la pression vendeuse (optionnel)
        vwap_ok = True
        if self._require_vwap and not np.isnan(ctx.last_vwap):
            if not price_below_vwap(last_close, ctx.last_vwap, threshold_pct=0.0):
                return []
        # 8. Neckline geometry
        between_lows = [s for s in swings if s.kind == SwingKind.LOW and h1.index < s.index < h2.index]
        if not between_lows:
            return []
        neckline = min(between_lows, key=lambda s: s.price)
        if (avg - neckline.price) / avg < self._min_neck:
            return []
        if last_close < neckline.price * (1.0 - self._neck_buf):
            return []
        if last_close > avg * (1.0 + self._twin_tol):
            return []

        height = avg - neckline.price
        raw_inv = max(h1.price, h2.price)
        tightened_inv = self._tighten_sl(neckline.price, raw_inv)
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
                "pre_trend_validated": True,
                "rsi_divergence": rsi_div,
                "volume_exhaustion": vol_conf,
            },
        )]

    # ------------------------------------------------------------------
    # Double bottom (bullish reversal)
    # ------------------------------------------------------------------
    def _detect_double_bottom(self, ohlcv, swings, symbol, timeframe,
                               ctx: _IndicatorContext) -> list[ChartPatternDTO]:
        last_close = float(ohlcv["close"].iloc[-1])
        if last_close <= 0:
            return []
        last_idx = len(ohlcv) - 1
        lows = [s for s in swings if s.kind == SwingKind.LOW]
        if len(lows) < 2:
            return []
        l2 = lows[-1]
        l1 = lows[-2]
        if not self._swings_well_spaced(l1, l2):
            return []
        avg = (l1.price + l2.price) / 2.0
        if abs(l1.price - l2.price) / avg > self._twin_tol:
            return []
        if not self._pre_trend_ok(ohlcv, l1.index, "down"):
            return []
        if not self._swing_has_solid_body(ohlcv, l1):
            return []
        if not self._swing_has_solid_body(ohlcv, l2):
            return []
        # 5. RSI divergence bullish : RSI sur bot2 > RSI sur bot1
        rsi_div = False
        if self._require_rsi_div and ctx.rsi is not None:
            if not bullish_rsi_div_on_bottoms(ctx.rsi, l1.index, l2.index,
                                                min_rsi_rise=self._min_rsi_div_pts):
                return []
            rsi_div = True
        # 6. Volume accumulation : volume sur bot2 > volume sur bot1
        vol_conf = False
        if self._require_vol_conf and ctx.volumes is not None:
            if not volume_accumulation_on_bottoms(ctx.volumes, l1.index, l2.index):
                return []
            vol_conf = True
        # 7. VWAP : prix au-dessus VWAP confirme pression acheteuse
        if self._require_vwap and not np.isnan(ctx.last_vwap):
            if not price_above_vwap(last_close, ctx.last_vwap):
                return []
        # 8. Neckline
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
                "pre_trend_validated": True,
                "rsi_divergence": rsi_div,
                "volume_accumulation": vol_conf,
            },
        )]

    # ------------------------------------------------------------------
    # Head & Shoulders (bearish)
    # ------------------------------------------------------------------
    def _detect_hs(self, ohlcv, swings, symbol, timeframe,
                    ctx: _IndicatorContext) -> list[ChartPatternDTO]:
        last_close = float(ohlcv["close"].iloc[-1])
        if last_close <= 0:
            return []
        last_idx = len(ohlcv) - 1
        highs = [s for s in swings if s.kind == SwingKind.HIGH]
        if len(highs) < 3:
            return []
        ls, head, rs = highs[-3], highs[-2], highs[-1]
        # Espacement entre LS, head et RS
        if not self._swings_well_spaced(ls, head) or not self._swings_well_spaced(head, rs):
            return []
        # Head doit être plus haut que les épaules d'au moins X%
        if head.price <= ls.price * (1.0 + self._head_prom):
            return []
        if head.price <= rs.price * (1.0 + self._head_prom):
            return []
        # Épaules proches
        avg_sh = (ls.price + rs.price) / 2.0
        if abs(ls.price - rs.price) / avg_sh > self._shoulder_tol:
            return []
        # Pre-trend uptrend avant pattern
        if not self._pre_trend_ok(ohlcv, ls.index, "up"):
            return []
        # Validite des sommets
        if not self._swing_has_solid_body(ohlcv, ls):
            return []
        if not self._swing_has_solid_body(ohlcv, head):
            return []
        if not self._swing_has_solid_body(ohlcv, rs):
            return []
        # RSI divergence : RSI sur head doit etre < RSI sur left shoulder (force qui s'epuise)
        rsi_div = False
        if self._require_rsi_div and ctx.rsi is not None:
            if not bearish_rsi_div_on_tops(ctx.rsi, ls.index, head.index,
                                            min_rsi_drop=self._min_rsi_div_pts):
                return []
            rsi_div = True
        # Volume exhaustion : vol sur right shoulder < vol sur left shoulder
        vol_conf = False
        if self._require_vol_conf and ctx.volumes is not None:
            if not volume_exhaustion_on_tops(ctx.volumes, ls.index, rs.index):
                return []
            vol_conf = True
        # VWAP optionnel
        if self._require_vwap and not np.isnan(ctx.last_vwap):
            if not price_below_vwap(float(ohlcv["close"].iloc[-1]), ctx.last_vwap):
                return []
        # Neckline
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
                "rsi_divergence": rsi_div,
                "volume_exhaustion": vol_conf,
            },
        )]

    # ------------------------------------------------------------------
    # Inverse H&S (bullish)
    # ------------------------------------------------------------------
    def _detect_ihs(self, ohlcv, swings, symbol, timeframe,
                     ctx: _IndicatorContext) -> list[ChartPatternDTO]:
        last_close = float(ohlcv["close"].iloc[-1])
        if last_close <= 0:
            return []
        last_idx = len(ohlcv) - 1
        lows = [s for s in swings if s.kind == SwingKind.LOW]
        if len(lows) < 3:
            return []
        ls, head, rs = lows[-3], lows[-2], lows[-1]
        if not self._swings_well_spaced(ls, head) or not self._swings_well_spaced(head, rs):
            return []
        if head.price >= ls.price * (1.0 - self._head_prom):
            return []
        if head.price >= rs.price * (1.0 - self._head_prom):
            return []
        avg_sh = (ls.price + rs.price) / 2.0
        if abs(ls.price - rs.price) / avg_sh > self._shoulder_tol:
            return []
        # Pre-trend downtrend avant pattern
        if not self._pre_trend_ok(ohlcv, ls.index, "down"):
            return []
        if not self._swing_has_solid_body(ohlcv, ls):
            return []
        if not self._swing_has_solid_body(ohlcv, head):
            return []
        if not self._swing_has_solid_body(ohlcv, rs):
            return []
        # RSI divergence bullish : RSI[head] > RSI[left_shoulder] (faiblesse qui se renverse)
        rsi_div = False
        if self._require_rsi_div and ctx.rsi is not None:
            if not bullish_rsi_div_on_bottoms(ctx.rsi, ls.index, head.index,
                                                min_rsi_rise=self._min_rsi_div_pts):
                return []
            rsi_div = True
        # Volume accumulation
        vol_conf = False
        if self._require_vol_conf and ctx.volumes is not None:
            if not volume_accumulation_on_bottoms(ctx.volumes, ls.index, rs.index):
                return []
            vol_conf = True
        # VWAP
        if self._require_vwap and not np.isnan(ctx.last_vwap):
            if not price_above_vwap(float(ohlcv["close"].iloc[-1]), ctx.last_vwap):
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
                "rsi_divergence": rsi_div,
                "volume_accumulation": vol_conf,
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
