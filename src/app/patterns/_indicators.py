"""Indicateurs techniques pour confluence pattern-specific.

Fonctions :
    compute_rsi(closes, period=14) -> np.ndarray
    compute_vwap_rolling(ohlcv, period=96) -> np.ndarray   # VWAP rolling (intraday-like)
    compute_volume_sma(volumes, period=20) -> np.ndarray

Helpers pattern :
    rsi_at_index(rsi, idx) -> float
    bearish_rsi_div_on_tops(rsi, idx_top1, idx_top2) -> bool
    bullish_rsi_div_on_bottoms(rsi, idx_bot1, idx_bot2) -> bool
    volume_exhaustion_on_tops(volumes, idx_top1, idx_top2) -> bool   # vol[top2] < vol[top1]
    volume_accumulation_on_bottoms(volumes, idx_bot1, idx_bot2) -> bool  # vol[bot2] > vol[bot1]
    price_above_vwap(close, vwap, threshold_pct=0.0) -> bool
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────────────
# Indicateurs
# ────────────────────────────────────────────────────────────────────


def compute_rsi(closes: np.ndarray | pd.Series, period: int = 14) -> np.ndarray:
    """RSI Wilder. Renvoie un np.ndarray de meme longueur que closes (NaN au debut)."""
    if isinstance(closes, np.ndarray):
        s = pd.Series(closes)
    else:
        s = closes.astype(float)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).to_numpy(dtype=float)


def compute_vwap_rolling(
    ohlcv: pd.DataFrame, period: int = 96
) -> np.ndarray:
    """VWAP rolling sur ``period`` bougies (par defaut 1 jour en 15m).

    VWAP = sum(typical_price * volume) / sum(volume) sur la fenetre.
    typical_price = (high + low + close) / 3.
    Retourne np.ndarray de meme longueur (NaN au debut < period).
    """
    if len(ohlcv) == 0:
        return np.array([])
    tp = (ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3.0
    vol = ohlcv["volume"].astype(float)
    pv = tp * vol
    # Rolling sums
    pv_sum = pv.rolling(window=period, min_periods=1).sum()
    v_sum = vol.rolling(window=period, min_periods=1).sum().replace(0.0, np.nan)
    vwap = (pv_sum / v_sum).fillna(method="bfill").fillna(method="ffill")
    return vwap.to_numpy(dtype=float)


def compute_volume_sma(volumes: np.ndarray | pd.Series, period: int = 20) -> np.ndarray:
    if isinstance(volumes, np.ndarray):
        s = pd.Series(volumes)
    else:
        s = volumes.astype(float)
    return s.rolling(window=period, min_periods=1).mean().to_numpy(dtype=float)


def compute_atr(
    ohlcv: pd.DataFrame, period: int = 14
) -> np.ndarray:
    """Average True Range Wilder.

    Retourne un np.ndarray de meme longueur (NaN au debut). Utile pour
    placer un SL avec un buffer adapte a la volatilite (vs SL purement
    structurel qui se fait wick).
    """
    if len(ohlcv) == 0:
        return np.array([])
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    close = ohlcv["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr.fillna(0.0).to_numpy(dtype=float)


def atr_at(atr: np.ndarray, idx: int, fallback_pct_of: float | None = None) -> float:
    """Acces atr[idx] avec garde. Si NaN/0 et fallback fourni, retourne fallback_pct_of*0.005."""
    if idx < 0 or idx >= len(atr):
        return (fallback_pct_of * 0.005) if fallback_pct_of else 0.0
    v = float(atr[idx])
    if not np.isfinite(v) or v <= 0:
        return (fallback_pct_of * 0.005) if fallback_pct_of else 0.0
    return v


# ────────────────────────────────────────────────────────────────────
# Helpers pattern-specific
# ────────────────────────────────────────────────────────────────────


def rsi_at(rsi: np.ndarray, idx: int) -> float:
    """Acces sur rsi avec bornes. NaN -> 50."""
    if idx < 0 or idx >= len(rsi):
        return 50.0
    v = float(rsi[idx])
    return 50.0 if np.isnan(v) else v


def bearish_rsi_div_on_tops(
    rsi: np.ndarray, idx_top1: int, idx_top2: int,
    min_rsi_drop: float = 2.0,
) -> bool:
    """Bearish divergence : prix fait HH (top2 >= top1) mais RSI fait LH (rsi2 < rsi1).

    Pour DOUBLE_TOP les tops sont a peu pres egaux, on accepte si RSI baisse.
    Pour HEAD_SHOULDERS (head > shoulders), idem : RSI au head doit etre < RSI a left_shoulder.

    min_rsi_drop : baisse minimum (en points RSI) pour qualifier une divergence.
    """
    r1 = rsi_at(rsi, idx_top1)
    r2 = rsi_at(rsi, idx_top2)
    return (r1 - r2) >= min_rsi_drop


def bullish_rsi_div_on_bottoms(
    rsi: np.ndarray, idx_bot1: int, idx_bot2: int,
    min_rsi_rise: float = 2.0,
) -> bool:
    """Bullish divergence : prix fait LL mais RSI fait HL (rsi2 > rsi1)."""
    r1 = rsi_at(rsi, idx_bot1)
    r2 = rsi_at(rsi, idx_bot2)
    return (r2 - r1) >= min_rsi_rise


def volume_exhaustion_on_tops(
    volumes: np.ndarray, idx_top1: int, idx_top2: int,
    avg_window: int = 3,
) -> bool:
    """Bearish : volume sur top2 doit etre INFERIEUR au volume sur top1.

    Indique un essoufflement des acheteurs sur le 2e top.
    On moyenne sur avg_window bougies autour de chaque top pour plus de robustesse.
    """
    if idx_top1 < 0 or idx_top2 < 0 or idx_top2 >= len(volumes):
        return False
    half = avg_window // 2
    v1_slice = volumes[max(0, idx_top1 - half): idx_top1 + half + 1]
    v2_slice = volumes[max(0, idx_top2 - half): idx_top2 + half + 1]
    if len(v1_slice) == 0 or len(v2_slice) == 0:
        return False
    return float(v2_slice.mean()) < float(v1_slice.mean())


def volume_accumulation_on_bottoms(
    volumes: np.ndarray, idx_bot1: int, idx_bot2: int,
    avg_window: int = 3,
) -> bool:
    """Bullish : volume sur bot2 doit etre SUPERIEUR au volume sur bot1.

    Indique de l'accumulation / interet des acheteurs sur le 2e creux.
    """
    if idx_bot1 < 0 or idx_bot2 < 0 or idx_bot2 >= len(volumes):
        return False
    half = avg_window // 2
    v1_slice = volumes[max(0, idx_bot1 - half): idx_bot1 + half + 1]
    v2_slice = volumes[max(0, idx_bot2 - half): idx_bot2 + half + 1]
    if len(v1_slice) == 0 or len(v2_slice) == 0:
        return False
    return float(v2_slice.mean()) > float(v1_slice.mean())


def price_above_vwap(close: float, vwap_value: float, threshold_pct: float = 0.0) -> bool:
    """Verifie si le prix est au-dessus du VWAP par au moins threshold_pct."""
    if vwap_value <= 0 or np.isnan(vwap_value):
        return False
    return close >= vwap_value * (1.0 + threshold_pct)


def price_below_vwap(close: float, vwap_value: float, threshold_pct: float = 0.0) -> bool:
    """Verifie si le prix est au-dessous du VWAP."""
    if vwap_value <= 0 or np.isnan(vwap_value):
        return False
    return close <= vwap_value * (1.0 - threshold_pct)
