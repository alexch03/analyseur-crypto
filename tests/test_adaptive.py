"""Tests de la couche adaptative : features continues + decisions de gestion.

Verifie que :
  - les features sont signees correctement (favorable au trade = positif)
  - une situation franchement favorable -> conviction haute -> EXTEND pres de la cible
  - une situation franchement adverse -> conviction basse -> annule l'ordre / EXIT
  - pas de crash sur series courtes
"""

from __future__ import annotations

import numpy as np

from app.strategy.adaptive import (
    ManageAction,
    TradeFeatures,
    compute_trade_features,
    conviction_score,
    decide_open,
    decide_pending,
)


def _arrays(closes, vol_up=True):
    closes = np.array(closes, dtype=float)
    n = len(closes)
    highs = closes + 0.5
    lows = closes - 0.5
    opens = np.empty(n)
    opens[0] = closes[0]
    opens[1:] = closes[:-1]  # open = close precedent (bougie haussiere si close monte)
    volumes = np.full(n, 100.0)
    return highs, lows, closes, volumes, opens


def test_features_neutral_on_short_series():
    h, l, c, v, o = _arrays([100, 101, 102])
    f = compute_trade_features(h, l, c, v, o, direction=1, entry=100, target=110)
    assert isinstance(f, TradeFeatures)
    # serie trop courte -> features neutres, pas de crash
    assert -1.0 <= conviction_score(f) <= 1.0


def test_strong_uptrend_long_high_conviction():
    # Tendance haussiere reguliere : LONG doit avoir conviction > 0
    closes = list(np.linspace(100, 130, 60))
    h, l, c, v, o = _arrays(closes)
    f = compute_trade_features(h, l, c, v, o, direction=1, entry=100, target=135,
                               running_extreme=130)
    assert f.momentum > 0
    assert f.htf_alignment > 0
    assert f.structure_health > 0
    assert conviction_score(f) > 0.3


def test_strong_uptrend_short_negative_conviction():
    # Meme tendance haussiere mais on est SHORT -> tout doit etre defavorable
    closes = list(np.linspace(100, 130, 60))
    h, l, c, v, o = _arrays(closes)
    f = compute_trade_features(h, l, c, v, o, direction=-1, entry=130, target=110,
                               running_extreme=130)
    assert f.momentum < 0
    assert f.htf_alignment < 0
    assert conviction_score(f) < 0


def test_decide_pending_cancels_when_adverse():
    # LONG arme mais le marche descend fortement -> annuler l'ordre
    closes = list(np.linspace(130, 100, 60))  # downtrend
    h, l, c, v, o = _arrays(closes)
    f = compute_trade_features(h, l, c, v, o, direction=1, entry=130, target=145)
    assert decide_pending(f) is True


def test_decide_pending_keeps_when_favorable():
    closes = list(np.linspace(100, 130, 60))  # uptrend
    h, l, c, v, o = _arrays(closes)
    f = compute_trade_features(h, l, c, v, o, direction=1, entry=100, target=135)
    assert decide_pending(f) is False


def test_decide_open_extends_near_target_with_conviction():
    # LONG fort, proche de la cible (mfe_progress eleve) -> laisser courir
    closes = list(np.linspace(100, 134, 60))
    h, l, c, v, o = _arrays(closes)
    f = compute_trade_features(h, l, c, v, o, direction=1, entry=100, target=135,
                               running_extreme=134)  # ~97% du chemin
    assert f.mfe_progress >= 0.8
    assert decide_open(f) == ManageAction.EXTEND_TARGET


def test_decide_open_exits_when_strongly_adverse():
    # LONG mais effondrement -> couper tot
    closes = list(np.linspace(130, 100, 60))
    h, l, c, v, o = _arrays(closes)
    f = compute_trade_features(h, l, c, v, o, direction=1, entry=130, target=145,
                               running_extreme=130)
    assert decide_open(f) == ManageAction.EXIT_NOW


def test_conviction_bounded():
    closes = list(np.linspace(100, 200, 80))
    h, l, c, v, o = _arrays(closes)
    f = compute_trade_features(h, l, c, v, o, direction=1, entry=100, target=210,
                               running_extreme=200)
    assert -1.0 <= conviction_score(f) <= 1.0


def _trend_df(prices):
    import pandas as pd
    from datetime import UTC, datetime, timedelta
    n = len(prices)
    opens = [prices[0]] + list(prices[:-1])
    return pd.DataFrame({
        "timestamp": [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=15 * i) for i in range(n)],
        "open": opens,
        "high": [p + 0.3 for p in prices],
        "low": [p - 0.3 for p in prices],
        "close": list(prices),
        "volume": [100.0] * n,
    })


def _armed_long(entry, target, invalidation, end_idx):
    from datetime import UTC, datetime
    from app.schemas.hypothesis import HypothesisDTO, HypothesisState
    from app.schemas.patterns import BreakoutDirection, ChartPatternDTO, PatternKind
    from app.schemas.domain import Side
    pat = ChartPatternDTO(
        kind=PatternKind.TRIANGLE_ASC, symbol="X/USDT", timeframe="15m",
        start_index=0, end_index=end_idx,
        start_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        end_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        breakout_level=entry, invalidation_level=invalidation,
        breakout_direction=BreakoutDirection.UP, height=target - entry,
        target=target, confidence=0.7,
    )
    return HypothesisDTO(
        id="adaptive-test", pattern=pat, symbol="X/USDT", timeframe="15m", side=Side.LONG,
        entry_price=entry, target_price=target, invalidation_price=invalidation,
        state=HypothesisState.ARMED,
        created_at=datetime(2026, 1, 1, tzinfo=UTC), updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_engine_adaptive_cancels_armed_in_adverse_market():
    """ARMED + marche franchement adverse + adaptive ON -> ordre annule (INVALIDATED)."""
    import numpy as np
    from app.services.hypothesis_engine import HypothesisEngine
    from app.schemas.hypothesis import HypothesisState

    df = _trend_df(list(np.linspace(105, 95, 60)))  # downtrend (adverse pour un LONG)
    end_idx = len(df) - 1
    # invalidation lointaine (80) pour qu'AUCUNE regle de prix n'invalide : seul l'adaptatif agit
    armed = _armed_long(entry=112, target=130, invalidation=80, end_idx=end_idx)

    eng_on = HypothesisEngine(adaptive_enabled=True, breakeven_trigger_pct=0.0,
                              trailing_stop_atr_mult=0.0, expiry_bars=999)
    res = eng_on.step(df, [], [armed])
    updated = res.updated[0]
    assert updated.state == HypothesisState.INVALIDATED
    assert any("adaptive" in t.reason for _, t in res.transitions)


def test_engine_adaptive_off_keeps_armed():
    """Meme scenario, adaptive OFF -> reste ARMED (aucune regle de prix ne s'applique)."""
    import numpy as np
    from app.services.hypothesis_engine import HypothesisEngine
    from app.schemas.hypothesis import HypothesisState

    df = _trend_df(list(np.linspace(105, 95, 60)))
    end_idx = len(df) - 1
    armed = _armed_long(entry=112, target=130, invalidation=80, end_idx=end_idx)

    eng_off = HypothesisEngine(adaptive_enabled=False, breakeven_trigger_pct=0.0,
                               trailing_stop_atr_mult=0.0, expiry_bars=999)
    res = eng_off.step(df, [], [armed])
    updated = res.updated[0]
    assert updated.state == HypothesisState.ARMED  # pas d'annulation adaptative


def test_features_no_lookahead_only_uses_given_slice():
    # La fonction ne recoit que le slice : par construction pas de look-ahead.
    # On verifie qu'allonger la serie future ne change PAS les features du passe.
    closes_short = list(np.linspace(100, 120, 50))
    closes_long = closes_short + list(np.linspace(120, 90, 30))  # futur baissier
    hs, ls, cs, vs, os_ = _arrays(closes_short)
    f_short = compute_trade_features(hs, ls, cs, vs, os_, direction=1, entry=100, target=125)
    # recalcule sur le meme prefixe extrait de la serie longue
    hl, ll, cl, vl, ol = _arrays(closes_long)
    f_prefix = compute_trade_features(
        hl[:50], ll[:50], cl[:50], vl[:50], ol[:50],
        direction=1, entry=100, target=125,
    )
    assert f_short.momentum == f_prefix.momentum
    assert f_short.htf_alignment == f_prefix.htf_alignment
