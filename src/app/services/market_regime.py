"""Detection du regime de marche : BULL / BEAR / RANGE + volatilite.

Utilise BTC comme proxy principal + breadth (% de symbols above SMA20).

Le regime est :
  - calcule en continu (a chaque scan cycle)
  - historise en DB (table market_regime_history)
  - lu par HypothesisEngine pour adapter le filtrage des patterns

Regimes :
  TREND  : BULL | BEAR | RANGE
  VOL    : LOW | NORMAL | HIGH
  STRENGTH : 0.0 - 1.0 (confidence du regime trend)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

Trend = Literal["BULL", "BEAR", "RANGE"]
Vol = Literal["LOW", "NORMAL", "HIGH"]


@dataclass
class MarketRegime:
    trend: Trend
    volatility: Vol
    strength: float           # 0..1, confiance dans le regime trend
    btc_change_24h_pct: float
    btc_above_sma50: bool
    btc_above_sma200: bool
    breadth_pct: float        # % de symbols above their SMA20 (univers)
    atr_pct: float            # ATR / Price (volatilite relative)
    detected_at: datetime

    def is_bull(self) -> bool: return self.trend == "BULL"
    def is_bear(self) -> bool: return self.trend == "BEAR"
    def is_range(self) -> bool: return self.trend == "RANGE"
    def is_high_vol(self) -> bool: return self.volatility == "HIGH"

    def as_dict(self) -> dict:
        return {
            "trend": self.trend,
            "volatility": self.volatility,
            "strength": round(self.strength, 3),
            "btc_change_24h_pct": round(self.btc_change_24h_pct, 2),
            "btc_above_sma50": self.btc_above_sma50,
            "btc_above_sma200": self.btc_above_sma200,
            "breadth_pct": round(self.breadth_pct, 1),
            "atr_pct": round(self.atr_pct, 3),
            "detected_at": self.detected_at.isoformat(),
        }


def _sma(series: pd.Series, period: int) -> float:
    if len(series) < period:
        return float(series.mean())
    return float(series.tail(period).mean())


def _atr_pct(ohlcv: pd.DataFrame, period: int = 14) -> float:
    """ATR (en %) sur les `period` dernieres bougies."""
    if len(ohlcv) < period + 1:
        return 0.0
    h = ohlcv["high"].to_numpy(dtype=float)[-(period + 1):]
    l = ohlcv["low"].to_numpy(dtype=float)[-(period + 1):]
    c = ohlcv["close"].to_numpy(dtype=float)[-(period + 1):]
    prev_c = c[:-1]
    h_l = h[1:] - l[1:]
    h_pc = np.abs(h[1:] - prev_c)
    l_pc = np.abs(l[1:] - prev_c)
    tr = np.maximum.reduce([h_l, h_pc, l_pc])
    atr = float(tr.mean())
    last = float(c[-1])
    return (atr / last) * 100.0 if last > 0 else 0.0


def detect_regime(
    btc_ohlcv: pd.DataFrame,
    *,
    breadth_pct: float | None = None,
) -> MarketRegime:
    """Detecte le regime depuis OHLCV BTC + (optionnel) breadth global.

    Args:
        btc_ohlcv : DataFrame avec colonnes timestamp, open, high, low, close, volume.
                    Idealement >= 200 bougies pour la SMA200.
        breadth_pct : si fourni, % de symbols au-dessus de leur SMA20. Si None,
                      le regime est detecte sur BTC seul.

    Returns:
        MarketRegime
    """
    closes = btc_ohlcv["close"].astype(float)
    last = float(closes.iloc[-1])
    sma50 = _sma(closes, 50)
    sma200 = _sma(closes, 200) if len(closes) >= 200 else _sma(closes, len(closes))
    btc_above_50 = last > sma50
    btc_above_200 = last > sma200

    # Change 24h (suppose 15m TF : 24h = 96 bars)
    bars_24h = min(96, len(closes) - 1)
    if bars_24h > 0:
        prev_close = float(closes.iloc[-bars_24h - 1])
        change_24h = (last / prev_close - 1.0) * 100.0 if prev_close > 0 else 0.0
    else:
        change_24h = 0.0

    # SMA50 slope (10 dernieres bougies)
    if len(closes) >= 60:
        sma50_now = _sma(closes, 50)
        sma50_prev = float(closes.iloc[-60:-10].mean())  # SMA50 d'il y a 10 bougies
        sma50_slope = (sma50_now / sma50_prev - 1.0) * 100.0 if sma50_prev > 0 else 0.0
    else:
        sma50_slope = 0.0

    # Breadth fallback : si non fourni, derive de BTC alone
    if breadth_pct is None:
        # Approximation : si BTC > SMA50 = 60%, si BTC > SMA200 = +20%
        breadth_pct = 50.0
        if btc_above_50: breadth_pct += 10
        if btc_above_200: breadth_pct += 10
        breadth_pct += np.sign(sma50_slope) * min(10, abs(sma50_slope) * 2)

    # Volatilite
    atr_pct = _atr_pct(btc_ohlcv, period=14)
    if atr_pct < 0.5:
        vol: Vol = "LOW"
    elif atr_pct > 1.5:
        vol = "HIGH"
    else:
        vol = "NORMAL"

    # Classification trend
    # Score bullish : btc_above_50 + btc_above_200 + breadth + sma50_slope + change_24h
    score_bull = 0.0
    if btc_above_50: score_bull += 0.25
    if btc_above_200: score_bull += 0.25
    if breadth_pct >= 55: score_bull += 0.2
    elif breadth_pct >= 50: score_bull += 0.1
    if sma50_slope > 0.5: score_bull += 0.15
    elif sma50_slope > 0: score_bull += 0.05
    if change_24h > 1: score_bull += 0.15
    elif change_24h > 0: score_bull += 0.05

    score_bear = 0.0
    if not btc_above_50: score_bear += 0.25
    if not btc_above_200: score_bear += 0.25
    if breadth_pct <= 45: score_bear += 0.2
    elif breadth_pct < 50: score_bear += 0.1
    if sma50_slope < -0.5: score_bear += 0.15
    elif sma50_slope < 0: score_bear += 0.05
    if change_24h < -1: score_bear += 0.15
    elif change_24h < 0: score_bear += 0.05

    if score_bull >= 0.65 and score_bull > score_bear + 0.15:
        trend: Trend = "BULL"
        strength = min(1.0, score_bull)
    elif score_bear >= 0.65 and score_bear > score_bull + 0.15:
        trend = "BEAR"
        strength = min(1.0, score_bear)
    else:
        trend = "RANGE"
        strength = 1.0 - abs(score_bull - score_bear)  # plus c'est proche, plus c'est range

    return MarketRegime(
        trend=trend, volatility=vol, strength=strength,
        btc_change_24h_pct=change_24h,
        btc_above_sma50=btc_above_50, btc_above_sma200=btc_above_200,
        breadth_pct=float(breadth_pct), atr_pct=atr_pct,
        detected_at=datetime.now(timezone.utc),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Adaptation des patterns selon le regime
# ─────────────────────────────────────────────────────────────────────────────


# Pour chaque pattern, le "regime ideal" (boost) ou "regime contre" (penalize)
PATTERN_REGIME_AFFINITY: dict[str, dict[str, float]] = {
    # bull_boost, bear_boost, range_boost (multiplier sur quality 0..1)
    "DOUBLE_TOP":             {"BULL": 0.5, "BEAR": 1.2, "RANGE": 1.0},  # bearish reversal
    "DOUBLE_BOTTOM":          {"BULL": 1.2, "BEAR": 0.5, "RANGE": 1.0},  # bullish reversal
    "HEAD_SHOULDERS":         {"BULL": 0.4, "BEAR": 1.3, "RANGE": 0.9},
    "INVERSE_HEAD_SHOULDERS": {"BULL": 1.3, "BEAR": 0.4, "RANGE": 0.9},
    "CHANNEL_UP":             {"BULL": 1.2, "BEAR": 0.6, "RANGE": 1.0},
    "CHANNEL_DOWN":           {"BULL": 0.6, "BEAR": 1.2, "RANGE": 1.0},
    "FLAG_BULL":              {"BULL": 1.3, "BEAR": 0.5, "RANGE": 0.8},
    "FLAG_BEAR":              {"BULL": 0.5, "BEAR": 1.3, "RANGE": 0.8},
    "WEDGE_RISING":           {"BULL": 0.8, "BEAR": 1.1, "RANGE": 1.0},  # bearish reversal
    "WEDGE_FALLING":          {"BULL": 1.1, "BEAR": 0.8, "RANGE": 1.0},  # bullish reversal
    "TRIANGLE_ASC":           {"BULL": 1.2, "BEAR": 0.7, "RANGE": 1.0},
    "TRIANGLE_DESC":          {"BULL": 0.7, "BEAR": 1.2, "RANGE": 1.0},
    "TRIANGLE_SYM":           {"BULL": 1.0, "BEAR": 1.0, "RANGE": 1.1},
    "RECTANGLE":              {"BULL": 0.9, "BEAR": 0.9, "RANGE": 1.3},  # consolidation
    # Nouveaux patterns (mai 2026)
    "TRIPLE_TOP":             {"BULL": 0.4, "BEAR": 1.3, "RANGE": 0.9},  # 3 sommets = bearish reversal
    "TRIPLE_BOTTOM":          {"BULL": 1.3, "BEAR": 0.4, "RANGE": 0.9},
    "EXPANDING_TRIANGLE_BEARISH": {"BULL": 0.5, "BEAR": 1.2, "RANGE": 1.0},
    "EXPANDING_TRIANGLE_BULLISH": {"BULL": 1.2, "BEAR": 0.5, "RANGE": 1.0},
    "EXPANDING_TRIANGLE_SYM": {"BULL": 1.0, "BEAR": 1.0, "RANGE": 1.1},   # megaphone = forte vol incertaine
    "PENNANT_BULL":           {"BULL": 1.3, "BEAR": 0.5, "RANGE": 0.8},   # continuation bullish
    "PENNANT_BEAR":           {"BULL": 0.5, "BEAR": 1.3, "RANGE": 0.8},
}


def pattern_regime_score(pattern_kind: str, regime: MarketRegime) -> float:
    """Retourne un multiplicateur (0..1.5) selon affinite pattern x regime.

    1.0 = neutre, > 1.0 = boost (regime favorable), < 1.0 = penalize (contre-trend).
    """
    affin = PATTERN_REGIME_AFFINITY.get(pattern_kind)
    if not affin:
        return 1.0
    base = affin.get(regime.trend, 1.0)
    # Module par la strength du regime : si strength=0.5 (incertain), moins d'impact
    delta = base - 1.0
    return 1.0 + delta * regime.strength


def should_reject_pattern(pattern_kind: str, regime: MarketRegime,
                            min_score: float = 0.6) -> tuple[bool, str]:
    """Retourne (reject, reason). Rejette si le pattern est trop contre-regime."""
    score = pattern_regime_score(pattern_kind, regime)
    if score < min_score:
        return True, f"pattern {pattern_kind} score={score:.2f} < min={min_score} in regime {regime.trend}"
    return False, ""
