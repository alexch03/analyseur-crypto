"""Outils géométriques partagés entre détecteurs de patterns.

Tous internes au package patterns/ (préfixe `_`).
"""

from __future__ import annotations

import numpy as np

from app.schemas.patterns import TrendLine


def fit_line(indices: list[int], prices: list[float]) -> TrendLine | None:
    """Régression linéaire ordinaire ; retourne None si <2 points."""
    if len(indices) < 2 or len(indices) != len(prices):
        return None
    xs = np.asarray(indices, dtype=float)
    ys = np.asarray(prices, dtype=float)
    n = len(xs)
    x_mean = xs.mean()
    y_mean = ys.mean()
    dx = xs - x_mean
    dy = ys - y_mean
    denom = float((dx * dx).sum())
    if denom <= 0.0:
        return None
    slope = float((dx * dy).sum() / denom)
    intercept = float(y_mean - slope * x_mean)
    y_pred = slope * xs + intercept
    ss_res = float(((ys - y_pred) ** 2).sum())
    ss_tot = float(((ys - y_mean) ** 2).sum())
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else (1.0 if ss_res == 0.0 else 0.0)
    return TrendLine(
        slope=slope,
        intercept=intercept,
        indices_used=tuple(int(i) for i in indices),
        r_squared=max(0.0, min(1.0, r2)),
    )


def slope_pct_per_bar(line: TrendLine, ref_price: float) -> float:
    """Pente exprimée en % du prix de référence par bougie."""
    if ref_price <= 0.0:
        return 0.0
    return line.slope / ref_price


def is_flat(line: TrendLine, ref_price: float, *, tol_pct_per_bar: float) -> bool:
    """Vrai si la pente est inférieure (en valeur absolue) à ``tol_pct_per_bar``."""
    return abs(slope_pct_per_bar(line, ref_price)) <= tol_pct_per_bar


def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, *, period: int = 14) -> float:
    """ATR simple sur les ``period`` dernières barres ; 0 si pas assez de données."""
    n = len(highs)
    if n < 2:
        return 0.0
    period = min(period, n - 1)
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ),
    )
    return float(tr[-period:].mean())
