"""Générateur de dataset par replay historique (chantier #3).

Rejoue les détecteurs de patterns sur un OHLCV historique, simule chaque trade
avec des stops ATR (#2), enregistre les features (#1) au moment de la détection,
et étiquette l'issue (TARGET_HIT / STOPPED). Produit un dataset multi-régimes —
la donnée diverse qui manque (la DB live ne couvre que 24h d'un seul régime BEAR
et est contaminée par des runs pré-fix).

Sortie compatible avec ``dataset.engineer_features`` / ``model`` / ``evaluate``.

Lancer :  .venv/Scripts/python.exe scripts/ml_replay.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.market_structure.swings import detect_swings
from app.ml.indicators import atr, compute_features
from app.ml.risk import atr_trade_plan
from app.patterns._quality import QualityWrappedDetector
from app.patterns.channels import ChannelDetector
from app.patterns.cup_and_handle import CupHandleDetector
from app.patterns.expanding_triangles import ExpandingTriangleDetector
from app.patterns.flags import FlagDetector
from app.patterns.pennants import PennantDetector
from app.patterns.rectangles import RectangleDetector
from app.patterns.reversal import ReversalDetector
from app.patterns.triangles import TriangleDetector
from app.patterns.triples import TripleDetector
from app.patterns.wedges import WedgeDetector
from app.schemas.patterns import BreakoutDirection

_DETECTOR_CLASSES = (
    TriangleDetector, RectangleDetector, ChannelDetector, WedgeDetector,
    FlagDetector, ReversalDetector, TripleDetector, ExpandingTriangleDetector,
    PennantDetector, CupHandleDetector,
)


def build_detectors() -> list:
    return [QualityWrappedDetector(cls()) for cls in _DETECTOR_CLASSES]


def _scorer():
    """ConfluenceScorer (import paresseux : évite de charger l'engine au module load)."""
    from app.services.hypothesis_engine import ConfluenceScorer
    return ConfluenceScorer()


def _safe_regime(window: pd.DataFrame):
    try:
        from app.services.market_regime import detect_regime
        return detect_regime(window)
    except Exception:
        return None


def _atr_val(window: pd.DataFrame) -> float:
    tail = window.iloc[-120:]
    return atr(tail["high"].to_numpy(float), tail["low"].to_numpy(float),
              tail["close"].to_numpy(float), 14)


def _simulate(df: pd.DataFrame, det_idx: int, side: str, entry: float,
              stop: float, target: float, arm_bars: int, horizon: int):
    """Simule trigger puis SL/TP. Retourne (label, pct_gain, trigger_idx) ou None.

    None = ordre jamais déclenché ou invalidé avant entrée (pas de trade) ou issue
    indécise dans l'horizon. On reste fidèle au lifecycle : ARM -> TRIGGER -> SL/TP,
    et stop prioritaire si une bougie touche les deux (conservateur).
    """
    n = len(df)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    # 1) chercher le trigger (cassure dans le bon sens) dans la fenêtre d'armement
    trig = None
    for j in range(det_idx + 1, min(det_idx + 1 + arm_bars, n)):
        if side == "LONG":
            if low[j] <= stop:          # invalidé avant entrée
                return None
            if high[j] >= entry:        # cassure haussière
                trig = j
                break
        else:
            if high[j] >= stop:
                return None
            if low[j] <= entry:
                trig = j
                break
    if trig is None:
        return None
    # 2) issue SL/TP à partir du trigger
    for k in range(trig, min(trig + horizon, n)):
        if side == "LONG":
            if low[k] <= stop:
                return 0, (stop / entry - 1.0) * 100.0, trig
            if high[k] >= target:
                return 1, (target / entry - 1.0) * 100.0, trig
        else:
            if high[k] >= stop:
                return 0, (entry / stop - 1.0) * 100.0, trig
            if low[k] <= target:
                return 1, (entry / target - 1.0) * 100.0, trig
    return None  # indécis dans l'horizon


def replay_ohlcv(
    df: pd.DataFrame, *, symbol: str, timeframe: str, htf: pd.DataFrame | None = None,
    k_stop: float = 2.0, rr_target: float = 2.0, arm_bars: int = 12, horizon: int = 120,
    step: int = 3, warmup: int = 210, detect_window: int = 300,
) -> list[dict]:
    """Rejoue un OHLCV et retourne une liste de lignes brutes (features + label)."""
    detectors = build_detectors()
    scorer = _scorer()
    n = len(df)
    rows: list[dict] = []
    seen: dict[tuple, int] = {}

    for idx in range(max(warmup, detect_window), n - 2, step):
        w = df.iloc[max(0, idx - detect_window + 1):idx + 1]
        try:
            swings = detect_swings(w, left=2, right=2)
        except Exception:
            continue
        if len(swings) < 3:
            continue
        patterns = []
        for det in detectors:
            try:
                patterns.extend(det.detect(w, swings, symbol=symbol, timeframe=timeframe))
            except Exception:
                pass
        if not patterns:
            continue

        regime = _safe_regime(w)
        atr_v = _atr_val(w)
        feats = None
        for p in patterns:
            if p.breakout_direction == BreakoutDirection.UNDETERMINED:
                continue
            side = "LONG" if p.breakout_direction == BreakoutDirection.UP else "SHORT"
            sig = (p.kind.value, side, round(float(p.breakout_level), 6))
            if sig in seen and (idx - seen[sig]) < arm_bars:
                continue  # déduplication (même pattern re-détecté barre suivante)

            plan = atr_trade_plan(side=side, entry=float(p.breakout_level),
                                  raw_invalidation=float(p.invalidation_level),
                                  atr=atr_v, k_stop=k_stop, rr_target=rr_target)
            sim = _simulate(df, idx, side, float(p.breakout_level), plan.stop,
                            plan.target, arm_bars, horizon)
            if sim is None:
                continue
            label, pct_gain, trig = sim
            seen[sig] = idx
            if feats is None:
                feats = compute_features(df, idx, htf)
            score, tags = scorer.score(p, w, market_regime=regime)
            rows.append({
                **feats,
                "id": f"{symbol}:{timeframe}:{idx}:{p.kind.value}",
                "hypothesis_id": f"{symbol}:{timeframe}:{idx}",
                "symbol_id": symbol,
                "confluence_score": float(score),
                "confluence_tags": list(tags),
                "pattern_snapshot": {
                    "kind": p.kind.value, "confidence": float(p.confidence),
                    "breakout_level": float(p.breakout_level), "height": float(p.height),
                },
                "h_entry": float(p.breakout_level),
                "target_price": float(plan.target),
                "invalidation_price": float(plan.stop),
                "regime_trend": getattr(regime, "trend", None),
                "regime_strength": getattr(regime, "strength", None),
                "entry_timestamp": str(df["timestamp"].iloc[trig]),
                "side": side,
                "pattern_kind": p.kind.value,
                "timeframe_id": timeframe,
                "pct_gain": float(pct_gain),
                "outcome": "TARGET_HIT" if label == 1 else "STOPPED",
            })
    return rows
