"""Fair Value Gap (FVG) and Inverse FVG (iFVG) detection.

Definitions (v1 — deterministic):

FVG Bullish (imbalance haussière):
    Three consecutive candles where:
        candle[i-1].high < candle[i+1].low
    The gap zone is [candle[i-1].high, candle[i+1].low].
    The middle candle (i) is the impulse candle.
    This represents an area where only buyers were present.

FVG Bearish (imbalance baissière):
    Three consecutive candles where:
        candle[i-1].low > candle[i+1].high
    The gap zone is [candle[i+1].high, candle[i-1].low].
    The middle candle (i) is the impulse candle.

Mitigation:
    An FVG is mitigated when price **enters** the gap zone (not merely
    touches the boundary).
    - Bullish FVG [bottom, top]: mitigated when a candle's low < midpoint
      of the gap (price traded through the interior).
    - Bearish FVG [bottom, top]: mitigated when a candle's high > midpoint
      of the gap.

iFVG (Inverse FVG):
    When an FVG gets fully mitigated (price passes through), the zone flips
    and becomes an iFVG of the opposite type (support becomes resistance and vice versa).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.schemas.domain import FVGType, FairValueGap


def detect_fvg(
    ohlcv: pd.DataFrame,
    *,
    min_gap_atr_ratio: float = 0.1,
) -> list[FairValueGap]:
    """Detect all Fair Value Gaps in the OHLCV data.

    Parameters
    ----------
    ohlcv:
        DataFrame with columns high, low, close, and either timestamp col or DatetimeIndex.
    min_gap_atr_ratio:
        Minimum gap size as a fraction of recent ATR to filter noise.
    """
    highs = ohlcv["high"].to_numpy(dtype=float)
    lows = ohlcv["low"].to_numpy(dtype=float)
    closes = ohlcv["close"].to_numpy(dtype=float)

    timestamps = ohlcv["timestamp"] if "timestamp" in ohlcv.columns else ohlcv.index

    n = len(highs)
    if n < 3:
        return []

    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ),
    )
    atr = float(np.mean(tr[-min(14, len(tr)):])) if len(tr) > 0 else 1.0
    min_gap = min_gap_atr_ratio * atr

    fvgs: list[FairValueGap] = []

    for i in range(1, n - 1):
        # Bullish FVG: gap between candle[i-1].high and candle[i+1].low
        gap_bull = lows[i + 1] - highs[i - 1]
        if gap_bull > min_gap:
            fvgs.append(
                FairValueGap(
                    index=i,
                    timestamp=timestamps.iloc[i] if hasattr(timestamps, "iloc") else timestamps[i],
                    top=float(lows[i + 1]),
                    bottom=float(highs[i - 1]),
                    fvg_type=FVGType.BULLISH,
                )
            )

        # Bearish FVG: gap between candle[i-1].low and candle[i+1].high
        gap_bear = lows[i - 1] - highs[i + 1]
        if gap_bear > min_gap:
            fvgs.append(
                FairValueGap(
                    index=i,
                    timestamp=timestamps.iloc[i] if hasattr(timestamps, "iloc") else timestamps[i],
                    top=float(lows[i - 1]),
                    bottom=float(highs[i + 1]),
                    fvg_type=FVGType.BEARISH,
                )
            )

    fvgs = _check_mitigation(fvgs, highs, lows)
    return fvgs


def _check_mitigation(
    fvgs: list[FairValueGap],
    highs: np.ndarray,
    lows: np.ndarray,
) -> list[FairValueGap]:
    """Mark FVGs as mitigated when price penetrates into the gap interior.

    Criterion: price must reach at least the midpoint of the gap zone, not
    merely touch the boundary.  This avoids premature invalidation by wicks.
    """
    result: list[FairValueGap] = []
    n = len(highs)

    for fvg in fvgs:
        mitigated = False
        mit_idx = None
        mid = (fvg.top + fvg.bottom) * 0.5

        start = fvg.index + 2
        for j in range(start, n):
            if fvg.fvg_type == FVGType.BULLISH:
                if lows[j] <= mid:
                    mitigated = True
                    mit_idx = j
                    break
            else:
                if highs[j] >= mid:
                    mitigated = True
                    mit_idx = j
                    break

        if mitigated:
            result.append(
                FairValueGap(
                    index=fvg.index,
                    timestamp=fvg.timestamp,
                    top=fvg.top,
                    bottom=fvg.bottom,
                    fvg_type=fvg.fvg_type,
                    mitigated=True,
                    mitigation_index=mit_idx,
                )
            )
        else:
            result.append(fvg)

    return result


def detect_ifvg(fvgs: list[FairValueGap]) -> list[FairValueGap]:
    """Return inverse FVGs: fully mitigated FVGs that flip polarity."""
    ifvgs: list[FairValueGap] = []
    for fvg in fvgs:
        if fvg.mitigated and fvg.mitigation_index is not None:
            flipped_type = FVGType.BEARISH if fvg.fvg_type == FVGType.BULLISH else FVGType.BULLISH
            ifvgs.append(
                FairValueGap(
                    index=fvg.mitigation_index,
                    timestamp=fvg.timestamp,
                    top=fvg.top,
                    bottom=fvg.bottom,
                    fvg_type=flipped_type,
                    mitigated=False,
                )
            )
    return ifvgs
