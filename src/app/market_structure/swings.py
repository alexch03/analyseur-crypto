"""Swing point (fractal pivot) detection.

Algorithm (v1 — fractal):
    A pivot HIGH at bar index i is confirmed when:
        high[i] > high[j]  for all j in [i-left, i-1]      (strict)
        high[i] >= high[j] for all j in [i+1,   i+right]   (non-strict right — avoids
                                                              duplicate tops at equal highs)

    A pivot LOW at bar index i is confirmed when:
        low[i] < low[j]    for all j in [i-left, i-1]
        low[i] <= low[j]   for all j in [i+1,   i+right]

Parameters:
    left  — number of bars to the left that must be lower/higher.
    right — number of bars to the right (confirmation window).

Both default to 2, giving a classic 5-bar fractal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.schemas.domain import SwingKind, SwingPoint


def detect_swings(
    ohlcv: pd.DataFrame,
    *,
    left: int = 2,
    right: int = 2,
) -> list[SwingPoint]:
    """Return an ordered list of swing points detected on the given OHLCV DataFrame.

    The DataFrame must contain at least columns ``high``, ``low``, and either
    ``timestamp`` or a DatetimeIndex.
    """
    highs: np.ndarray = ohlcv["high"].to_numpy(dtype=float)
    lows: np.ndarray = ohlcv["low"].to_numpy(dtype=float)

    timestamps = (
        ohlcv["timestamp"] if "timestamp" in ohlcv.columns else ohlcv.index
    )

    n = len(highs)
    points: list[SwingPoint] = []

    for i in range(left, n - right):
        # --- Swing HIGH ---
        is_high = True
        for j in range(i - left, i):
            if highs[i] <= highs[j]:
                is_high = False
                break
        if is_high:
            for j in range(i + 1, i + right + 1):
                if highs[i] < highs[j]:
                    is_high = False
                    break

        if is_high:
            points.append(
                SwingPoint(
                    index=i,
                    timestamp=timestamps.iloc[i] if hasattr(timestamps, "iloc") else timestamps[i],
                    price=float(highs[i]),
                    kind=SwingKind.HIGH,
                )
            )

        # --- Swing LOW ---
        is_low = True
        for j in range(i - left, i):
            if lows[i] >= lows[j]:
                is_low = False
                break
        if is_low:
            for j in range(i + 1, i + right + 1):
                if lows[i] > lows[j]:
                    is_low = False
                    break

        if is_low:
            points.append(
                SwingPoint(
                    index=i,
                    timestamp=timestamps.iloc[i] if hasattr(timestamps, "iloc") else timestamps[i],
                    price=float(lows[i]),
                    kind=SwingKind.LOW,
                )
            )

    points.sort(key=lambda p: p.index)
    return points
