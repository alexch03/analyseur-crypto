"""RSI and simple divergences (recent candles) for filter / score."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-style RSI (EMA on gains / losses)."""
    c = close.astype(float)
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _swing_low_indices(lows: np.ndarray, order: int = 2) -> list[int]:
    n = len(lows)
    out: list[int] = []
    for i in range(order, n - order):
        w = lows[i - order : i + order + 1]
        if lows[i] <= w.min() + 1e-12 and lows[i] == w.min():
            out.append(i)
    return out


def _swing_high_indices(highs: np.ndarray, order: int = 2) -> list[int]:
    n = len(highs)
    out: list[int] = []
    for i in range(order, n - order):
        w = highs[i - order : i + order + 1]
        if highs[i] >= w.max() - 1e-12 and highs[i] == w.max():
            out.append(i)
    return out


def rsi_bullish_divergence_recent(
    ohlcv: pd.DataFrame,
    *,
    lookback: int = 50,
    swing_order: int = 2,
    rsi_period: int = 14,
) -> bool:
    """Most recent window: two price lows with LL on price and HL on RSI."""
    if len(ohlcv) < lookback + rsi_period + swing_order * 2 + 2:
        return False
    lows = ohlcv["low"].to_numpy(dtype=float)
    rsi = compute_rsi(ohlcv["close"], rsi_period).to_numpy(dtype=float)
    start = len(ohlcv) - lookback
    idxs = [i for i in _swing_low_indices(lows, swing_order) if i >= start and not np.isnan(rsi[i])]
    if len(idxs) < 2:
        return False
    i, j = idxs[-2], idxs[-1]
    if j <= i:
        return False
    if lows[j] >= lows[i] - 1e-9:
        return False
    if rsi[j] <= rsi[i] + 1e-9:
        return False
    return True


def rsi_bearish_divergence_recent(
    ohlcv: pd.DataFrame,
    *,
    lookback: int = 50,
    swing_order: int = 2,
    rsi_period: int = 14,
) -> bool:
    """Most recent window: two price highs with HH on price and LH on RSI."""
    if len(ohlcv) < lookback + rsi_period + swing_order * 2 + 2:
        return False
    highs = ohlcv["high"].to_numpy(dtype=float)
    rsi = compute_rsi(ohlcv["close"], rsi_period).to_numpy(dtype=float)
    start = len(ohlcv) - lookback
    idxs = [i for i in _swing_high_indices(highs, swing_order) if i >= start and not np.isnan(rsi[i])]
    if len(idxs) < 2:
        return False
    i, j = idxs[-2], idxs[-1]
    if j <= i:
        return False
    if highs[j] <= highs[i] + 1e-9:
        return False
    if rsi[j] >= rsi[i] - 1e-9:
        return False
    return True
