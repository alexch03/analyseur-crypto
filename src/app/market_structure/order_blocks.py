"""Order Block (OB) detection.

Definitions (v1 — deterministic):

Bullish Order Block:
    The last bearish (down-close) candle before a strong bullish impulse move
    that creates a BOS or displaces price significantly.
    - Identify the last down candle (close < open) before an up-move
      where the move's range exceeds `impulse_atr_mult * ATR`.
    - The OB zone is [candle.low, candle.high] of that last bearish candle.
    - When price later returns to this zone, it's expected to act as demand.

Bearish Order Block:
    The last bullish (up-close) candle before a strong bearish impulse move.
    - Identify the last up candle (close > open) before a down-move
      where the move's range exceeds `impulse_atr_mult * ATR`.
    - The OB zone is [candle.low, candle.high] of that last bullish candle.
    - When price later returns to this zone, it's expected to act as supply.

Mitigation:
    An OB is mitigated when price later closes through the OB zone
    (beyond the far edge):
    - Bullish OB: mitigated when close < OB.low
    - Bearish OB: mitigated when close > OB.high
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.schemas.domain import OBType, OrderBlock


def _compute_atr_array(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(highs)
    if n < 2:
        return np.full(n, np.mean(highs - lows) if n > 0 else 1.0)

    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))

    atr = np.full(n, np.mean(tr[:period]))
    for i in range(period, n):
        atr[i] = np.mean(tr[max(0, i - period + 1) : i + 1])
    return atr


def detect_order_blocks(
    ohlcv: pd.DataFrame,
    *,
    impulse_atr_mult: float = 1.5,
    atr_period: int = 14,
    lookback: int = 3,
) -> list[OrderBlock]:
    """Detect order blocks in OHLCV data.

    Parameters
    ----------
    ohlcv:
        DataFrame with open, high, low, close columns.
    impulse_atr_mult:
        The impulse move following the OB candle must exceed this multiple of ATR.
    lookback:
        Number of candles to look back from the impulse start to find the OB candle.
    """
    opens = ohlcv["open"].to_numpy(dtype=float)
    highs = ohlcv["high"].to_numpy(dtype=float)
    lows = ohlcv["low"].to_numpy(dtype=float)
    closes = ohlcv["close"].to_numpy(dtype=float)
    timestamps = ohlcv["timestamp"] if "timestamp" in ohlcv.columns else ohlcv.index

    n = len(opens)
    if n < 4:
        return []

    atr = _compute_atr_array(highs, lows, closes, atr_period)

    obs: list[OrderBlock] = []

    for i in range(2, n):
        threshold = impulse_atr_mult * atr[i]

        # Bullish impulse: current candle is strong up
        if closes[i] - opens[i] > threshold:
            for j in range(i - 1, max(i - lookback - 1, -1), -1):
                if closes[j] < opens[j]:  # bearish candle = the OB
                    obs.append(
                        OrderBlock(
                            index=j,
                            timestamp=timestamps.iloc[j] if hasattr(timestamps, "iloc") else timestamps[j],
                            top=float(highs[j]),
                            bottom=float(lows[j]),
                            ob_type=OBType.BULLISH,
                        )
                    )
                    break

        # Bearish impulse: current candle is strong down
        if opens[i] - closes[i] > threshold:
            for j in range(i - 1, max(i - lookback - 1, -1), -1):
                if closes[j] > opens[j]:  # bullish candle = the OB
                    obs.append(
                        OrderBlock(
                            index=j,
                            timestamp=timestamps.iloc[j] if hasattr(timestamps, "iloc") else timestamps[j],
                            top=float(highs[j]),
                            bottom=float(lows[j]),
                            ob_type=OBType.BEARISH,
                        )
                    )
                    break

    obs = _deduplicate(obs)
    obs = _check_mitigation(obs, closes)
    return obs


def _deduplicate(obs: list[OrderBlock]) -> list[OrderBlock]:
    """Remove duplicate OBs at the same index and type."""
    seen: set[tuple[int, str]] = set()
    unique: list[OrderBlock] = []
    for ob in obs:
        key = (ob.index, ob.ob_type.value)
        if key not in seen:
            seen.add(key)
            unique.append(ob)
    return unique


def _check_mitigation(obs: list[OrderBlock], closes: np.ndarray) -> list[OrderBlock]:
    """Mark OBs as mitigated when price closes through the zone."""
    n = len(closes)
    result: list[OrderBlock] = []

    for ob in obs:
        mitigated = False
        mit_idx = None

        for j in range(ob.index + 2, n):
            if ob.ob_type == OBType.BULLISH and closes[j] < ob.bottom:
                mitigated = True
                mit_idx = j
                break
            if ob.ob_type == OBType.BEARISH and closes[j] > ob.top:
                mitigated = True
                mit_idx = j
                break

        if mitigated:
            result.append(
                OrderBlock(
                    index=ob.index,
                    timestamp=ob.timestamp,
                    top=ob.top,
                    bottom=ob.bottom,
                    ob_type=ob.ob_type,
                    mitigated=True,
                    mitigation_index=mit_idx,
                )
            )
        else:
            result.append(ob)

    return result
