"""Automatic parameter policy for SMC analysis per timeframe.

Goal:
- Avoid manual tuning for non-technical users.
- Use robust defaults per timeframe.
- Adapt proximity/targets dynamically to volatility regime.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


_BASE_PRESETS: dict[str, dict[str, Any]] = {
    "5m": {
        "rr_min": 1.8,
        "fvg_proximity_pct": 0.0025,
        "ob_proximity_pct": 0.0025,
        "max_setups": 5,
        "swing_left": 2,
        "swing_right": 2,
    },
    "15m": {
        "rr_min": 1.9,
        "fvg_proximity_pct": 0.0030,
        "ob_proximity_pct": 0.0030,
        "max_setups": 4,
        "swing_left": 2,
        "swing_right": 2,
    },
    "1h": {
        "rr_min": 2.0,
        "fvg_proximity_pct": 0.0040,
        "ob_proximity_pct": 0.0040,
        "max_setups": 4,
        "swing_left": 3,
        "swing_right": 3,
    },
    "4h": {
        "rr_min": 2.2,
        "fvg_proximity_pct": 0.0050,
        "ob_proximity_pct": 0.0050,
        "max_setups": 4,
        "swing_left": 4,
        "swing_right": 4,
    },
    "1d": {
        "rr_min": 2.5,
        "fvg_proximity_pct": 0.0065,
        "ob_proximity_pct": 0.0065,
        "max_setups": 3,
        "swing_left": 5,
        "swing_right": 5,
    },
}


def _volatility_ratio(ohlcv_df: pd.DataFrame) -> float:
    if len(ohlcv_df) < 20:
        return 0.0
    high = ohlcv_df["high"].astype(float)
    low = ohlcv_df["low"].astype(float)
    close = ohlcv_df["close"].astype(float)
    tr = (high - low).rolling(14).mean().iloc[-1]
    c = close.iloc[-1]
    if c <= 0:
        return 0.0
    return float(tr / c)


def resolve_smc_parameters(
    *,
    timeframe: str,
    ohlcv_df: pd.DataFrame | None,
    auto_enabled: bool,
    manual_params: dict[str, Any],
) -> dict[str, Any]:
    """Return engine + structure params used by scan/backtest/optimize.

    Returned keys:
      rr_min, fvg_proximity_pct, ob_proximity_pct, max_setups, swing_left, swing_right
    """
    if not auto_enabled:
        return {
            "rr_min": float(manual_params.get("rr_min", 2.0)),
            "fvg_proximity_pct": float(manual_params.get("fvg_proximity_pct", 0.004)),
            "ob_proximity_pct": float(manual_params.get("ob_proximity_pct", 0.004)),
            "max_setups": int(manual_params.get("max_setups", 4)),
            "swing_left": int(manual_params.get("swing_left", 3)),
            "swing_right": int(manual_params.get("swing_right", 3)),
        }

    p = dict(_BASE_PRESETS.get(timeframe, _BASE_PRESETS["1h"]))
    vol = _volatility_ratio(ohlcv_df) if ohlcv_df is not None else 0.0

    # Dynamic volatility adaptation.
    if vol > 0.02:
        p["fvg_proximity_pct"] *= 1.25
        p["ob_proximity_pct"] *= 1.25
        p["rr_min"] += 0.2
    elif 0 < vol < 0.008:
        p["fvg_proximity_pct"] *= 0.9
        p["ob_proximity_pct"] *= 0.9

    return p
