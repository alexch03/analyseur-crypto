"""Spécification des variables (features) du modèle de sélection de trades.

Source unique de vérité partagée entre :
  - la construction du dataset offline (``dataset.py``),
  - l'entraînement (``model.py``),
  - l'inférence live dans le moteur (phase 5).

Toute feature déclarée ici est automatiquement prise en compte par le pipeline
sklearn (cf. ``model.build_pipeline``). On distingue trois familles, traitées
différemment par le préprocesseur :

  - NUMERIC      : imputées (médiane) puis standardisées.
  - CATEGORICAL  : one-hot encodées (chaque modalité = une variable indépendante).
  - BINARY       : 0/1 déjà prêtes (tags de confluence, drapeaux).
"""

from __future__ import annotations

import json

from app.ml.indicators import INDICATOR_OHLCV

# ---------------------------------------------------------------------------
# Tags de confluence informatifs.
# `directional_bias` est présent sur 100 % des trades -> variance nulle -> exclu.
# ---------------------------------------------------------------------------
CONFLUENCE_TAGS: list[str] = [
    "volume_weak",
    "volume_expansion",
    "trend_aligned",
    "trend_flat",
    "trend_counter",
]
TAG_FEATURES: list[str] = [f"tag_{t}" for t in CONFLUENCE_TAGS]

NUMERIC_FEATURES: list[str] = [
    "confluence_score",   # score legacy (gardé comme une variable parmi d'autres)
    "geom_confidence",    # qualité géométrique du fit (pattern_snapshot.confidence)
    "height_pct",         # amplitude du pattern / prix
    "rr",                 # reward:risk = dist(target) / dist(stop)
    "stop_dist_pct",      # distance entry->invalidation, en %
    "tgt_dist_pct",       # distance entry->target, en %
    "regime_strength",    # force du régime au moment de l'entrée
    "entry_hour",         # heure UTC (0-23) — saisonnalité intraday
    "entry_dow",          # jour de semaine (0-6)
]

# Indicateurs techniques calculés au moment de la détection (indicators.py).
# Absents des datasets reconstruits sans OHLCV -> NaN -> imputés par le pipeline.
# stop_dist_atr = stop_dist_pct / atr_pct : « le stop fait-il assez d'ATR ? »
# (variable clé : stop_dist_pct était le coefficient #1 du modèle).
INDICATOR_FEATURES: list[str] = INDICATOR_OHLCV + ["stop_dist_atr"]
NUMERIC_FEATURES = NUMERIC_FEATURES + INDICATOR_FEATURES

CATEGORICAL_FEATURES: list[str] = [
    "pattern_kind",       # type de figure (one-hot ~16 modalités)
    "side",               # LONG / SHORT
    "timeframe_id",       # 15m / 1h / 4h
    "regime_trend",       # BULL / BEAR / RANGE / UNKNOWN
]

BINARY_FEATURES: list[str] = TAG_FEATURES + ["regime_known"]

ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

LABEL = "label"  # 1 = TARGET_HIT (gagnant), 0 = STOPPED (perdant)

# Colonnes conservées pour l'analyse / le split temporel mais NON utilisées
# comme features (éviter toute fuite : pct_gain et outcome dérivent du label).
META_COLUMNS: list[str] = [
    "id", "hypothesis_id", "symbol_id", "entry_timestamp", "pct_gain", "outcome",
]


def parse_tags(raw: object) -> list[str]:
    """Parse une cellule confluence_tags (JSON str ou liste) en liste de tags."""
    if not raw:
        return []
    try:
        v = json.loads(raw) if isinstance(raw, str) else raw
        return list(v) if isinstance(v, (list, tuple)) else []
    except Exception:
        return []


def parse_snapshot(raw: object) -> dict:
    """Parse pattern_snapshot (JSON) en dict ; {} si invalide."""
    if not raw:
        return {}
    try:
        v = json.loads(raw) if isinstance(raw, str) else raw
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}
