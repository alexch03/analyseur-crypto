"""Support / resistance level detection via swing clustering.

Algorithm (v1):
    1. Compute a price tolerance ε = atr_mult * ATR(atr_period) over the full OHLCV.
       Falls back to a percentage of the median candle range if ATR cannot be computed.
    2. Sort swing prices.
    3. Greedily merge consecutive swings whose prices lie within ε of each other into
       clusters.
    4. Each cluster becomes an SR level:
       - price  = median of the cluster prices
       - width  = max − min within the cluster
       - touches = number of swings in the cluster
       - role   = SUPPORT if majority are LOWs, RESISTANCE if majority are HIGHs
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.schemas.domain import SRLevel, SRRole, SwingKind, SwingPoint


def _compute_atr(ohlcv: pd.DataFrame, period: int = 14) -> float:
    high = ohlcv["high"].to_numpy(dtype=float)
    low = ohlcv["low"].to_numpy(dtype=float)
    close = ohlcv["close"].to_numpy(dtype=float)

    if len(high) < 2:
        return float(np.median(high - low)) if len(high) > 0 else 1.0

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))

    if len(tr) < period:
        return float(np.mean(tr))

    atr = np.mean(tr[-period:])
    return float(atr)


def detect_sr_levels(
    ohlcv: pd.DataFrame,
    swings: list[SwingPoint],
    *,
    atr_mult: float = 0.5,
    atr_period: int = 14,
    min_touches: int = 2,
) -> list[SRLevel]:
    """Cluster swing points into support/resistance levels."""
    if not swings:
        return []

    atr = _compute_atr(ohlcv, period=atr_period)
    epsilon = atr_mult * atr

    sorted_swings = sorted(swings, key=lambda s: s.price)

    clusters: list[list[SwingPoint]] = []
    current_cluster: list[SwingPoint] = [sorted_swings[0]]

    for sw in sorted_swings[1:]:
        if sw.price - current_cluster[-1].price <= epsilon:
            current_cluster.append(sw)
        else:
            clusters.append(current_cluster)
            current_cluster = [sw]
    clusters.append(current_cluster)

    levels: list[SRLevel] = []
    for cluster in clusters:
        if len(cluster) < min_touches:
            continue

        prices = [s.price for s in cluster]
        lows_count = sum(1 for s in cluster if s.kind == SwingKind.LOW)
        highs_count = len(cluster) - lows_count

        levels.append(
            SRLevel(
                price=float(np.median(prices)),
                width=max(prices) - min(prices),
                touches=len(cluster),
                role=SRRole.SUPPORT if lows_count >= highs_count else SRRole.RESISTANCE,
            )
        )

    return levels
