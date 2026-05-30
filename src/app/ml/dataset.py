"""Construction du dataset supervisé (features, label) pour la sélection de trades.

Label : 1 = TARGET_HIT (gagnant), 0 = STOPPED (perdant). On n'inclut que les
trades RÉELLEMENT pris (table ``unit_trades`` = hypothèses passées TRIGGERED)
au résultat connu — pas les INVALIDATED (ordre annulé avant entrée) ni les
ouverts (outcome NULL).

Les features sont *reconstruites* (la table dédiée ``feature_snapshots`` est
vide) par jointure :

    unit_trades  ⋈  hypotheses (pattern_snapshot, niveaux)  ⋈  market_regime_snapshots

Lancer :  PYTHONPATH=src python -m app.ml.dataset [db_path]
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.features import (
    ALL_FEATURES,
    CONFLUENCE_TAGS,
    LABEL,
    META_COLUMNS,
    NUMERIC_FEATURES,
    parse_snapshot,
    parse_tags,
)
from app.ml.indicators import INDICATOR_OHLCV

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB = ROOT / "analyseur.db"
DATASET_PATH = ROOT / "data" / "ml" / "dataset.csv"

# Régime au moment de l'ENTRÉE : dernier snapshot <= entry_timestamp.
_SQL = """
SELECT
  u.id, u.hypothesis_id, u.symbol_id, u.timeframe_id, u.side, u.pattern_kind,
  u.entry_price, u.entry_timestamp, u.pct_gain, u.outcome,
  u.confluence_score, u.confluence_tags,
  h.entry_price        AS h_entry,
  h.target_price       AS target_price,
  h.invalidation_price AS invalidation_price,
  h.pattern_snapshot   AS pattern_snapshot,
  (SELECT trend    FROM market_regime_snapshots s
     WHERE s.snapshot_ts <= u.entry_timestamp ORDER BY s.snapshot_ts DESC LIMIT 1) AS regime_trend,
  (SELECT strength FROM market_regime_snapshots s
     WHERE s.snapshot_ts <= u.entry_timestamp ORDER BY s.snapshot_ts DESC LIMIT 1) AS regime_strength
FROM unit_trades u
JOIN hypotheses h ON h.id = u.hypothesis_id
WHERE u.outcome IN ('TARGET_HIT', 'STOPPED')
ORDER BY u.entry_timestamp
"""


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule toutes les features (ALL_FEATURES) depuis les colonnes brutes.

    PARTAGÉ entre offline (build_dataset) et live (inférence dans le moteur) afin
    de garantir la parité train/serve : la moindre divergence de calcul fausserait
    les prédictions en production. N'ajoute PAS le label (inconnu en live).

    Colonnes brutes attendues : confluence_score, confluence_tags, pattern_snapshot,
    h_entry, target_price, invalidation_price, regime_trend, regime_strength,
    entry_timestamp, side, pattern_kind, timeframe_id.
    """
    # --- Tags de confluence -> binaires ----------------------------------
    tags = df["confluence_tags"].apply(parse_tags)
    for t in CONFLUENCE_TAGS:
        df[f"tag_{t}"] = tags.apply(lambda lst, tag=t: int(tag in lst))

    # --- Géométrie depuis pattern_snapshot -------------------------------
    snaps = df["pattern_snapshot"].apply(parse_snapshot)
    df["geom_confidence"] = _num(snaps.apply(lambda d: d.get("confidence")))
    breakout = _num(snaps.apply(lambda d: d.get("breakout_level"))).abs()
    height = _num(snaps.apply(lambda d: d.get("height"))).abs()
    df["height_pct"] = (height / breakout).where(breakout > 0)

    # --- Reward:Risk & distances (niveaux du plan d'hypothèse) -----------
    entry = _num(df["h_entry"]).abs()
    tgt = _num(df["target_price"])
    inv = _num(df["invalidation_price"])
    stop_dist = (_num(df["h_entry"]) - inv).abs()
    tgt_dist = (tgt - _num(df["h_entry"])).abs()
    df["stop_dist_pct"] = (stop_dist / entry).where(entry > 0)
    df["tgt_dist_pct"] = (tgt_dist / entry).where(entry > 0)
    df["rr"] = (tgt_dist / stop_dist).where(stop_dist > 0).clip(upper=20.0)

    # --- Régime ----------------------------------------------------------
    df["regime_trend"] = df["regime_trend"].fillna("UNKNOWN").astype(str)
    df["regime_known"] = (df["regime_trend"] != "UNKNOWN").astype(int)
    df["regime_strength"] = _num(df["regime_strength"])

    # --- Temporalité -----------------------------------------------------
    ts = pd.to_datetime(df["entry_timestamp"], errors="coerce", utc=True)
    df["entry_hour"] = ts.dt.hour.astype("float64")
    df["entry_dow"] = ts.dt.dayofweek.astype("float64")

    # --- Types catégoriels -----------------------------------------------
    for c in ("side", "pattern_kind", "timeframe_id"):
        df[c] = df[c].astype(str)

    # --- Indicateurs techniques ------------------------------------------
    # Présents si fournis (replay/live via indicators.compute_features),
    # sinon NaN (dataset reconstruit depuis la DB sans OHLCV) -> imputés.
    for c in INDICATOR_OHLCV:
        df[c] = _num(df[c]) if c in df.columns else np.nan
    df["stop_dist_atr"] = (df["stop_dist_pct"] / df["atr_pct"]).where(df["atr_pct"] > 0)
    return df


def build_dataset(db_path: str | Path = DEFAULT_DB) -> pd.DataFrame:
    """Retourne un DataFrame [META_COLUMNS + ALL_FEATURES + LABEL], une ligne/trade clos."""
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(_SQL, conn)
    finally:
        conn.close()
    if df.empty:
        raise RuntimeError(f"Aucun trade clos (TARGET_HIT/STOPPED) dans {db_path}")

    df[LABEL] = (df["outcome"] == "TARGET_HIT").astype(int)
    df = engineer_features(df)
    keep = META_COLUMNS + ALL_FEATURES + [LABEL]
    return df[keep].copy()


def load_dataset(path: str | Path = DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def live_feature_row(
    *,
    pattern_kind: str,
    side: str,
    timeframe_id,
    confluence_score: float,
    confluence_tags,
    pattern_snapshot,
    entry: float,
    target: float,
    invalidation: float,
    regime_trend: str | None = None,
    regime_strength: float | None = None,
    entry_timestamp=None,
) -> pd.DataFrame:
    """Construit UNE ligne de features pour l'inférence live, via ``engineer_features``.

    Utilise exactement le même calcul que l'entraînement (parité train/serve).
    ``confluence_tags`` accepte une liste ou un JSON ; ``pattern_snapshot`` un dict
    ou un JSON. Retourne un DataFrame 1×len(ALL_FEATURES) prêt pour predict_proba.
    """
    raw = pd.DataFrame([{
        "confluence_score": confluence_score,
        "confluence_tags": confluence_tags,
        "pattern_snapshot": pattern_snapshot,
        "h_entry": entry,
        "target_price": target,
        "invalidation_price": invalidation,
        "regime_trend": regime_trend,
        "regime_strength": regime_strength,
        "entry_timestamp": entry_timestamp,
        "side": side,
        "pattern_kind": pattern_kind,
        "timeframe_id": str(timeframe_id),
    }])
    return engineer_features(raw)[ALL_FEATURES]


def _diagnostics(df: pd.DataFrame) -> None:
    n = len(df)
    pos = int(df[LABEL].sum())
    print(f"\nDataset: {n} trades clos | gagnants={pos} ({100*pos/n:.1f}%) | perdants={n-pos}")
    print(f"Features ({len(ALL_FEATURES)}): {ALL_FEATURES}")

    print("\n--- Winrate par pattern (n>=10) ---")
    g = df.groupby("pattern_kind")[LABEL].agg(["count", "mean"]).sort_values("mean")
    for k, r in g.iterrows():
        flag = "" if r["count"] >= 10 else "  (faible n)"
        print(f"  {k:28} n={int(r['count']):4}  wr={100*r['mean']:5.1f}%{flag}")

    print("\n--- Régime au close (couverture) ---")
    for k, r in df.groupby("regime_trend")[LABEL].agg(["count", "mean"]).iterrows():
        print(f"  {k:10} n={int(r['count']):4}  wr={100*r['mean']:5.1f}%")

    print("\n--- NaN par feature numérique ---")
    for c in NUMERIC_FEATURES:
        miss = int(df[c].isna().sum())
        if miss:
            print(f"  {c:18} {miss} NaN ({100*miss/n:.1f}%)")
    print("  (les NaN seront imputés (médiane) par le pipeline)")


def main() -> None:
    db = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB
    print(f"### build_dataset depuis {db}")
    df = build_dataset(db)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)
    print(f"Sauvegardé -> {DATASET_PATH}")
    _diagnostics(df)


if __name__ == "__main__":
    main()
