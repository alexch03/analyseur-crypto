"""Indicateurs techniques partagés — calculés au moment de la détection.

Brique commune aux 3 chantiers :
  - features du modèle (#1),
  - dimensionnement du stop en ATR (#2),
  - génération de dataset par replay (#3).

Tout est calculé en numpy/pandas (pas de dépendance TA-Lib) et UNIQUEMENT sur les
barres <= idx (aucune fuite look-ahead). ``compute_features`` borne la fenêtre à
``LOOKBACK`` barres pour rester rapide en replay (appelé des milliers de fois).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACK = 300  # suffisant pour EMA200 ; borne le coût en replay

# Features dérivées d'OHLCV (sans les niveaux du trade). stop_dist_atr est dérivée
# ailleurs (dataset.engineer_features) car elle dépend du stop.
INDICATOR_OHLCV: list[str] = [
    "atr_pct",          # ATR(14) / prix — régime de volatilité
    "adx",              # force de tendance (chop si < ~20) — validé walk-forward (≈trend_strength)
    "rsi",              # RSI(14) — surachat/survente
    "ema50_dist_pct",   # (close - EMA50) / close — position vs tendance moyenne
    "ema200_dist_pct",  # (close - EMA200) / close — position vs tendance longue
    "bb_width_pct",     # largeur Bollinger / prix — squeeze vs expansion
    "bb_pos",           # (close - SMA20) / std20 — POSITION dans les bandes — validé walk-forward
    "vol_ratio",        # volume / moyenne(20) — confirmation de cassure — validé walk-forward (≈volume_spike)
    "entry_body_ratio", # |close-open| / (high-low) bougie courante — validé walk-forward
    "htf_trend_score",  # tendance du timeframe supérieur (NaN si non fourni)
]


def _wilder_rma(x: np.ndarray, n: int) -> np.ndarray:
    """Moyenne mobile de Wilder (RMA), base de RSI/ATR/ADX."""
    out = np.full(len(x), np.nan, dtype=float)
    if len(x) < n:
        return out
    out[n - 1] = x[:n].mean()
    for i in range(n, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def rsi(close: np.ndarray, n: int = 14) -> float:
    if len(close) < n + 1:
        return float("nan")
    d = np.diff(close)
    gain = _wilder_rma(np.where(d > 0, d, 0.0), n)
    loss = _wilder_rma(np.where(d < 0, -d, 0.0), n)
    g, l = gain[-1], loss[-1]
    if np.isnan(g) or np.isnan(l):
        return float("nan")
    if l == 0:
        return 100.0
    rs = g / l
    return float(100.0 - 100.0 / (1.0 + rs))


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> float:
    if len(close) < n + 1:
        return float("nan")
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    rma = _wilder_rma(tr, n)
    return float(rma[-1])


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> float:
    if len(close) < 2 * n + 1:
        return float("nan")
    up = high[1:] - high[:-1]
    dn = low[:-1] - low[1:]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    atr_ = _wilder_rma(tr, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100.0 * _wilder_rma(plus_dm, n) / atr_
        mdi = 100.0 * _wilder_rma(minus_dm, n) / atr_
        dx = 100.0 * np.abs(pdi - mdi) / (pdi + mdi)
    dx = np.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0)
    adx_ = _wilder_rma(dx[n - 1:], n)  # DX valide à partir de n-1
    return float(adx_[-1]) if len(adx_) and not np.isnan(adx_[-1]) else float("nan")


def _ema_last(close: np.ndarray, span: int) -> float:
    if len(close) < 2:
        return float("nan")
    return float(pd.Series(close).ewm(span=span, adjust=False).mean().iloc[-1])


def _htf_trend_score(htf: pd.DataFrame | None, ts) -> float:
    """Tendance du TF supérieur à l'instant ts : (close - EMA50) / close, borné [-1,1]."""
    if htf is None or "timestamp" not in htf.columns or len(htf) < 10:
        return float("nan")
    sub = htf[pd.to_datetime(htf["timestamp"], utc=True) <= pd.to_datetime(ts, utc=True)]
    if len(sub) < 10:
        return float("nan")
    close = sub["close"].to_numpy(float)
    c = close[-1]
    if c <= 0:
        return float("nan")
    ema = _ema_last(close, 50)
    return float(np.clip((c - ema) / c, -1.0, 1.0))


def compute_features(ohlcv: pd.DataFrame, idx: int = -1,
                     htf: pd.DataFrame | None = None) -> dict[str, float]:
    """Vecteur d'indicateurs au barreau ``idx`` (par défaut le dernier).

    N'utilise que les barres <= idx (pas de look-ahead). Retourne des NaN si
    l'historique est insuffisant (le pipeline modèle imputera).
    """
    n = len(ohlcv)
    if n == 0:
        return {k: float("nan") for k in INDICATOR_OHLCV}
    if idx < 0:
        idx = n + idx
    lo = max(0, idx + 1 - LOOKBACK)
    s = ohlcv.iloc[lo:idx + 1]
    close = s["close"].to_numpy(float)
    high = s["high"].to_numpy(float)
    low = s["low"].to_numpy(float)
    vol = s["volume"].to_numpy(float) if "volume" in s.columns else np.array([])
    open_ = s["open"].to_numpy(float) if "open" in s.columns else np.array([])

    feat: dict[str, float] = {k: float("nan") for k in INDICATOR_OHLCV}
    if len(close) < 30 or close[-1] <= 0:
        return feat
    c = close[-1]
    a = atr(high, low, close, 14)
    feat["atr_pct"] = (a / c) if a == a else float("nan")
    feat["adx"] = adx(high, low, close, 14)
    feat["rsi"] = rsi(close, 14)
    feat["ema50_dist_pct"] = (c - _ema_last(close, 50)) / c
    if len(close) >= 150:
        feat["ema200_dist_pct"] = (c - _ema_last(close, 200)) / c
    m = close[-20:].mean()
    sd = close[-20:].std()
    feat["bb_width_pct"] = (4.0 * sd / m) if m > 0 else float("nan")
    feat["bb_pos"] = ((c - m) / sd) if sd > 0 else float("nan")
    if len(vol) >= 21:
        base = vol[-21:-1].mean()
        feat["vol_ratio"] = (vol[-1] / base) if base > 0 else float("nan")
    if len(open_) > 0:
        rng = high[-1] - low[-1]
        feat["entry_body_ratio"] = abs(c - open_[-1]) / rng if rng > 0 else float("nan")
    if htf is not None and "timestamp" in s.columns:
        feat["htf_trend_score"] = _htf_trend_score(htf, s["timestamp"].iloc[-1])
    return feat
