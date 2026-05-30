"""Tests du système ML de sélection de trades : parité features, politique EV, gate."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.ml.dataset import engineer_features, live_feature_row
from app.ml.features import ALL_FEATURES, LABEL, NUMERIC_FEATURES
from app.ml.gate import MlTradeGate
from app.ml.indicators import INDICATOR_OHLCV, compute_features
from app.ml.model import save_model, split_xy, train_model
from app.ml.policy import TradePolicy


def _ohlcv_uptrend(n: int = 260, start: float = 100.0, step: float = 0.4) -> pd.DataFrame:
    closes = [start + i * step for i in range(n)]
    return pd.DataFrame({
        "timestamp": [datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(n)],
        "open": closes,
        "high": [c + 1.0 for c in closes],
        "low": [c - 1.0 for c in closes],
        "close": closes,
        "volume": [100.0 + i for i in range(n)],
    })


# ---------------------------------------------------------------------------
# Politique de décision (espérance + Kelly)
# ---------------------------------------------------------------------------
def test_policy_takes_positive_ev():
    p = TradePolicy()
    # RR=2 -> seuil p* = 1/3. prob 0.5 > 1/3 -> EV = 0.5*2 - 0.5 = +0.5R -> prendre
    d = p.decide(0.5, 2.0)
    assert d.take
    assert d.ev_r == pytest.approx(0.5)
    assert d.size > 0.0


def test_policy_rejects_negative_ev():
    p = TradePolicy()
    # prob 0.3 < 1/3 -> EV négative -> rejet
    d = p.decide(0.30, 2.0)
    assert not d.take
    assert d.size == 0.0


def test_policy_rejects_degenerate_rr():
    p = TradePolicy(min_rr=0.3)
    assert not p.decide(0.99, 0.05).take


def test_policy_size_scales_with_edge():
    p = TradePolicy(kelly_fraction=1.0)
    low = p.decide(0.40, 2.0)
    high = p.decide(0.70, 2.0)
    assert high.size > low.size  # plus d'edge -> plus de taille


# ---------------------------------------------------------------------------
# Parité train/serve : live_feature_row calcule comme l'entraînement
# ---------------------------------------------------------------------------
def test_live_feature_row_parity():
    X = live_feature_row(
        pattern_kind="DOUBLE_TOP",
        side="LONG",
        timeframe_id=2,
        confluence_score=0.5,
        confluence_tags=["trend_aligned", "volume_weak"],
        pattern_snapshot={"confidence": 0.7, "breakout_level": 100.0, "height": 5.0},
        entry=100.0,
        target=110.0,
        invalidation=97.0,
        regime_trend="BEAR",
        regime_strength=0.9,
        entry_timestamp="2026-05-28 19:00:00",
    )
    assert list(X.columns) == ALL_FEATURES
    row = X.iloc[0]
    assert row["stop_dist_pct"] == pytest.approx(0.03)      # |100-97|/100
    assert row["tgt_dist_pct"] == pytest.approx(0.10)       # |110-100|/100
    assert row["rr"] == pytest.approx(10.0 / 3.0)           # 0.10 / 0.03
    assert row["height_pct"] == pytest.approx(0.05)         # 5/100
    assert row["geom_confidence"] == pytest.approx(0.7)
    assert row["tag_trend_aligned"] == 1
    assert row["tag_volume_weak"] == 1
    assert row["tag_trend_counter"] == 0
    assert row["regime_known"] == 1


def test_live_feature_row_accepts_json_strings():
    """confluence_tags/pattern_snapshot doivent accepter du JSON comme la DB."""
    X = live_feature_row(
        pattern_kind="CUP_AND_HANDLE", side="SHORT", timeframe_id=3,
        confluence_score=0.6, confluence_tags='["trend_counter"]',
        pattern_snapshot=json.dumps({"confidence": 0.4, "breakout_level": 50.0, "height": 2.0}),
        entry=50.0, target=48.0, invalidation=51.0, regime_trend=None,
        regime_strength=None, entry_timestamp="2026-05-28T19:00:00+00:00",
    )
    row = X.iloc[0]
    assert row["tag_trend_counter"] == 1
    assert row["regime_known"] == 0          # régime inconnu
    assert row["rr"] == pytest.approx(2.0)   # reward 2 / risk 1


# ---------------------------------------------------------------------------
# Modèle + gate end-to-end
# ---------------------------------------------------------------------------
def _synthetic_dataset(n: int = 90) -> pd.DataFrame:
    """Dataset jouet : les LONG trend_aligned gagnent, les SHORT trend_counter perdent
    (signal apprenable) pour vérifier l'apprentissage de bout en bout."""
    rows = []
    for i in range(n):
        win = (i % 2 == 0)
        rows.append({
            "confluence_score": 0.6 if win else 0.4,
            "confluence_tags": ["trend_aligned"] if win else ["trend_counter"],
            "pattern_snapshot": {"confidence": 0.7 if win else 0.5,
                                 "breakout_level": 100.0, "height": 5.0},
            "h_entry": 100.0,
            "target_price": 110.0 if win else 108.0,
            "invalidation_price": 97.0,
            "regime_trend": "BEAR",
            "regime_strength": 0.9,
            "entry_timestamp": "2026-05-28 19:00:00.000000",
            "side": "LONG" if win else "SHORT",
            "pattern_kind": "DOUBLE_TOP" if (i % 3) else "TRIPLE_TOP",
            "timeframe_id": "2",
            "outcome": "TARGET_HIT" if win else "STOPPED",
        })
    df = engineer_features(pd.DataFrame(rows))
    df[LABEL] = (df["outcome"] == "TARGET_HIT").astype(int)
    return df


def test_model_trains_and_predicts():
    df = _synthetic_dataset()
    pipe = train_model(df, "logreg")
    X, _ = split_xy(df)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(df),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_gate_loads_and_decides(tmp_path):
    df = _synthetic_dataset()
    pipe = train_model(df, "logreg")
    path = tmp_path / "m.joblib"
    save_model(pipe, path)

    gate = MlTradeGate.load(path, TradePolicy())
    assert gate is not None
    dec = gate.evaluate(
        pattern_kind="DOUBLE_TOP", side="LONG", timeframe_id=2, confluence_score=0.6,
        confluence_tags=["trend_aligned"],
        pattern_snapshot={"confidence": 0.7, "breakout_level": 100.0, "height": 5.0},
        entry=100.0, target=110.0, invalidation=97.0, regime_trend="BEAR",
        regime_strength=0.9, entry_timestamp="2026-05-28 19:00:00",
    )
    assert 0.0 <= dec.prob <= 1.0
    assert isinstance(dec.take, bool)


def test_gate_try_load_missing_returns_none(tmp_path):
    assert MlTradeGate.try_load(tmp_path / "does_not_exist.joblib") is None


# ---------------------------------------------------------------------------
# Indicateurs techniques (#1)
# ---------------------------------------------------------------------------
def test_compute_features_uptrend():
    f = compute_features(_ohlcv_uptrend(), -1)
    assert set(INDICATOR_OHLCV).issubset(f)
    assert f["atr_pct"] > 0
    assert 0.0 <= f["rsi"] <= 100.0
    assert f["rsi"] > 60.0               # tendance haussière -> RSI élevé
    assert f["ema50_dist_pct"] > 0       # prix au-dessus de l'EMA50
    assert not np.isnan(f["adx"])        # tendance forte -> ADX défini


def test_compute_features_insufficient_history_is_nan():
    f = compute_features(_ohlcv_uptrend(n=12), -1)
    assert all(np.isnan(v) for v in f.values())


def test_indicator_features_registered_in_spec():
    assert "stop_dist_atr" in NUMERIC_FEATURES
    assert "atr_pct" in ALL_FEATURES
    assert "adx" in ALL_FEATURES


def test_stop_dist_atr_derived_when_atr_present():
    """Si atr_pct est fourni, stop_dist_atr = stop_dist_pct / atr_pct."""
    raw = pd.DataFrame([{
        "confluence_score": 0.5, "confluence_tags": [], "pattern_snapshot": {},
        "h_entry": 100.0, "target_price": 110.0, "invalidation_price": 96.0,
        "regime_trend": "BEAR", "regime_strength": 0.9,
        "entry_timestamp": "2026-05-28 19:00:00", "side": "LONG",
        "pattern_kind": "DOUBLE_TOP", "timeframe_id": "2",
        "atr_pct": 0.02,  # fourni par le replay
    }])
    out = engineer_features(raw)
    # stop_dist_pct = |100-96|/100 = 0.04 ; stop_dist_atr = 0.04 / 0.02 = 2.0
    assert out["stop_dist_atr"].iloc[0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Plan de trade ATR (#2)
# ---------------------------------------------------------------------------
def test_atr_plan_widens_tight_stop_long():
    from app.ml.risk import atr_trade_plan
    p = atr_trade_plan(side="LONG", entry=100.0, raw_invalidation=99.5, atr=2.0,
                       k_stop=2.0, rr_target=2.0)
    assert p.stop == pytest.approx(96.0)     # min(99.5, 100-4) -> élargi
    assert p.target == pytest.approx(108.0)  # 100 + 2×(100-96)


def test_atr_plan_keeps_wider_pattern_stop_long():
    from app.ml.risk import atr_trade_plan
    p = atr_trade_plan(side="LONG", entry=100.0, raw_invalidation=90.0, atr=2.0,
                       k_stop=2.0, rr_target=2.0)
    assert p.stop == pytest.approx(90.0)     # le pattern est déjà plus large que 2×ATR
    assert p.target == pytest.approx(120.0)


def test_atr_plan_short():
    from app.ml.risk import atr_trade_plan
    p = atr_trade_plan(side="SHORT", entry=100.0, raw_invalidation=100.5, atr=2.0,
                       k_stop=2.0, rr_target=2.0)
    assert p.stop == pytest.approx(104.0)    # max(100.5, 100+4)
    assert p.target == pytest.approx(92.0)


def test_atr_plan_no_atr_fallback():
    from app.ml.risk import atr_trade_plan
    p = atr_trade_plan(side="LONG", entry=100.0, raw_invalidation=95.0, atr=0.0,
                       rr_target=2.0)
    assert p.stop == pytest.approx(95.0)      # pas d'ATR -> stop du pattern
    assert p.target == pytest.approx(110.0)   # target aligné sur le RR
