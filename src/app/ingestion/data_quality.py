"""Validation et nettoyage de la qualite des donnees OHLCV.

Trois garde-fous, partages entre le scanner live, le backfill et le backtest,
pour garantir que les detecteurs et le moteur de replay ne consomment jamais
des donnees corrompues ou non representatives du live :

1. validate_ohlcv_df  : coherence OHLC (high>=low, open/close dans [low,high], volume>=0)
2. detect_gaps        : trous temporels (bougies manquantes) au-dela d'une tolerance
3. strip_unclosed_candle : retire la derniere bougie si elle est encore en formation

Les fonctions de validation/gap NE MODIFIENT PAS le DataFrame : elles renvoient
des diagnostics (listes de problemes) que l'appelant logge en WARNING. Seul
strip_unclosed_candle renvoie un DataFrame potentiellement tronque.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

_REQUIRED_COLS = ("timestamp", "open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class OHLCVIssue:
    index: int
    kind: str  # "high_lt_low" | "open_out_of_range" | "close_out_of_range" | "neg_volume" | "nan"
    detail: str


def validate_ohlcv_df(df: pd.DataFrame, *, max_report: int = 20) -> list[OHLCVIssue]:
    """Verifie la coherence d'un DataFrame OHLCV.

    Renvoie la liste des anomalies (vide = donnees saines). Ne leve pas.
    Regles :
      - high >= low
      - low <= open <= high  et  low <= close <= high
      - volume >= 0
      - pas de NaN sur les colonnes OHLCV
    """
    issues: list[OHLCVIssue] = []
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        return [OHLCVIssue(-1, "missing_columns", f"colonnes absentes: {missing}")]

    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)

    import numpy as np

    nan_mask = (
        np.isnan(o) | np.isnan(h) | np.isnan(low) | np.isnan(c) | np.isnan(v)
    )
    hl_mask = h < low
    o_mask = (o < low) | (o > h)
    c_mask = (c < low) | (c > h)
    vneg_mask = v < 0

    for i in range(len(df)):
        if len(issues) >= max_report:
            break
        if nan_mask[i]:
            issues.append(OHLCVIssue(i, "nan", "valeur NaN dans OHLCV"))
            continue
        if hl_mask[i]:
            issues.append(OHLCVIssue(i, "high_lt_low", f"high={h[i]} < low={low[i]}"))
        if o_mask[i]:
            issues.append(OHLCVIssue(i, "open_out_of_range", f"open={o[i]} hors [{low[i]},{h[i]}]"))
        if c_mask[i]:
            issues.append(OHLCVIssue(i, "close_out_of_range", f"close={c[i]} hors [{low[i]},{h[i]}]"))
        if vneg_mask[i]:
            issues.append(OHLCVIssue(i, "neg_volume", f"volume={v[i]} < 0"))
    return issues


@dataclass(frozen=True)
class Gap:
    index: int            # index de la bougie APRES le trou
    prev_ts: pd.Timestamp
    ts: pd.Timestamp
    missing_bars: int


def detect_gaps(
    df: pd.DataFrame, bar_seconds: float, *, tolerance: float = 1.5
) -> list[Gap]:
    """Detecte les trous temporels (bougies manquantes).

    Un trou est signale si l'ecart entre deux timestamps consecutifs depasse
    ``bar_seconds * tolerance``. Renvoie la liste des trous (vide = serie continue).
    """
    if "timestamp" not in df.columns or len(df) < 2 or bar_seconds <= 0:
        return []
    ts = pd.to_datetime(df["timestamp"])
    deltas = ts.diff().dt.total_seconds().to_numpy()
    gaps: list[Gap] = []
    threshold = bar_seconds * tolerance
    for i in range(1, len(df)):
        d = deltas[i]
        if d is not None and d > threshold:
            missing = int(round(d / bar_seconds)) - 1
            gaps.append(Gap(i, ts.iloc[i - 1], ts.iloc[i], max(0, missing)))
    return gaps


def strip_unclosed_candle(
    df: pd.DataFrame, bar_seconds: float, now: datetime
) -> pd.DataFrame:
    """Retire la derniere bougie si elle est encore en formation.

    Une bougie ouverte a ``ts_open`` se cloture a ``ts_open + bar_seconds``.
    Si ``now`` est anterieur a cette cloture, la bougie est incomplete (close
    mouvant) et NE DOIT PAS etre utilisee pour une decision de trade.
    """
    if len(df) == 0 or "timestamp" not in df.columns or bar_seconds <= 0:
        return df
    last_ts = pd.Timestamp(df["timestamp"].iloc[-1])
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")
    now_ts = pd.Timestamp(now)
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    close_ts = last_ts + pd.Timedelta(seconds=bar_seconds)
    if now_ts < close_ts:
        return df.iloc[:-1].reset_index(drop=True)
    return df
