"""Tests du moteur de cycle de vie d'hypothèse."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from app.schemas.patterns import BreakoutDirection, ChartPatternDTO, PatternKind
from app.services.hypothesis_engine import HypothesisEngine
from app.schemas.hypothesis import HypothesisState


def _bar_df(prices_close: list[float], highs: list[float] | None = None,
            lows: list[float] | None = None) -> pd.DataFrame:
    n = len(prices_close)
    h = highs or [c + 0.5 for c in prices_close]
    l = lows or [c - 0.5 for c in prices_close]
    return pd.DataFrame({
        "timestamp": [datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(n)],
        "open": prices_close,
        "high": h,
        "low": l,
        "close": prices_close,
        "volume": [100.0] * n,
    })


def _bull_pattern(symbol="TEST/USDT", timeframe="1h", end_idx=10) -> ChartPatternDTO:
    return ChartPatternDTO(
        kind=PatternKind.TRIANGLE_ASC,
        symbol=symbol,
        timeframe=timeframe,
        start_index=0,
        end_index=end_idx,
        start_timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        end_timestamp=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=end_idx),
        breakout_level=110.0,
        invalidation_level=100.0,
        breakout_direction=BreakoutDirection.UP,
        height=10.0,
        target=120.0,
        confidence=0.7,
    )


def _bear_pattern(symbol="TEST/USDT", timeframe="1h", end_idx=10) -> ChartPatternDTO:
    return ChartPatternDTO(
        kind=PatternKind.TRIANGLE_DESC,
        symbol=symbol,
        timeframe=timeframe,
        start_index=0,
        end_index=end_idx,
        start_timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        end_timestamp=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=end_idx),
        breakout_level=90.0,
        invalidation_level=100.0,
        breakout_direction=BreakoutDirection.DOWN,
        height=10.0,
        target=80.0,
        confidence=0.7,
    )


class TestHypothesisLifecycle:
    def test_forming_from_new_pattern(self):
        engine = HypothesisEngine()
        df = _bar_df([105.0] * 11)  # close=105, breakout=110 → loin, donc FORMING
        result = engine.step(df, [_bull_pattern()], [])

        assert len(result.created) == 1
        h = result.created[0]
        assert h.state == HypothesisState.FORMING
        assert h.side.value == "LONG"
        assert h.entry_price == 110.0
        assert h.target_price == 120.0
        assert h.invalidation_price == 100.0

    def test_forming_to_armed_on_proximity(self):
        engine = HypothesisEngine(arm_proximity_pct=0.01)  # 1% proximity
        # Première itération : pattern détecté à close=105
        df1 = _bar_df([105.0] * 11)
        r1 = engine.step(df1, [_bull_pattern()], [])
        h = r1.created[0]

        # Deuxième itération : close=109.5, dans la zone 1% de 110 → ARMED
        df2 = _bar_df([105.0] * 11 + [109.5])
        r2 = engine.step(df2, [], [h])
        updated = r2.updated[0]
        assert updated.state == HypothesisState.ARMED

    def test_armed_to_triggered_on_breakout(self):
        engine = HypothesisEngine(arm_proximity_pct=0.01)
        df1 = _bar_df([109.5] * 11)
        r1 = engine.step(df1, [_bull_pattern()], [])
        h = r1.created[0]
        assert h.state == HypothesisState.ARMED  # arm direct car proche

        # Cassure : close=111 > 110
        df2 = _bar_df([109.5] * 11 + [111.0])
        r2 = engine.step(df2, [], [h])
        updated = r2.updated[0]
        assert updated.state == HypothesisState.TRIGGERED
        assert updated.triggered_price == 111.0

    def test_armed_invalidated_cancels_order(self):
        """KEY TEST : si le prix casse l'invalidation AVANT trigger, l'hypothèse passe
        INVALIDATED (= ordre annulé) sans STOPPED."""
        engine = HypothesisEngine(arm_proximity_pct=0.01)
        df1 = _bar_df([109.5] * 11)
        r1 = engine.step(df1, [_bull_pattern()], [])
        h = r1.created[0]
        assert h.state == HypothesisState.ARMED

        # Cassure baissière : close=99 ≤ 100 (invalidation)
        df2 = _bar_df([109.5] * 11 + [99.0])
        r2 = engine.step(df2, [], [h])
        updated = r2.updated[0]
        assert updated.state == HypothesisState.INVALIDATED
        assert updated.triggered_at is None, "ne doit jamais avoir été triggered"

    def test_triggered_to_target_hit(self):
        engine = HypothesisEngine(arm_proximity_pct=0.01)
        df1 = _bar_df([109.5] * 11)
        r1 = engine.step(df1, [_bull_pattern()], [])
        h0 = r1.created[0]

        df2 = _bar_df([109.5] * 11 + [111.0])
        r2 = engine.step(df2, [], [h0])
        h1 = r2.updated[0]
        assert h1.state == HypothesisState.TRIGGERED

        # Bougie suivante : high=120.5 touche le target=120
        df3 = _bar_df([109.5] * 11 + [111.0, 120.0],
                      highs=[None] * 11 + [111.5, 120.5],
                      lows=[None] * 11 + [110.5, 119.5])
        # Reconstruire les None
        for col in ["high", "low"]:
            df3[col] = df3[col].where(df3[col].notna(),
                                       df3["close"] + (0.5 if col == "high" else -0.5))
        r3 = engine.step(df3, [], [h1])
        h2 = r3.updated[0]
        assert h2.state == HypothesisState.TARGET_HIT
        assert h2.outcome_price == 120.0
        # gain % = (120 / 111 - 1) * 100 ≈ 8.11%
        assert abs((h2.realized_pct or 0) - 8.108) < 0.01

    def test_triggered_to_stopped(self):
        engine = HypothesisEngine(arm_proximity_pct=0.01)
        df1 = _bar_df([109.5] * 11)
        r1 = engine.step(df1, [_bull_pattern()], [])

        df2 = _bar_df([109.5] * 11 + [111.0])
        r2 = engine.step(df2, [], r1.created)
        h_triggered = r2.updated[0]

        # Bougie suivante : low=99 touche invalidation=100
        df3 = _bar_df([109.5] * 11 + [111.0, 99.5],
                      highs=[c + 0.5 for c in [109.5] * 11 + [111.5, 100.0]],
                      lows=[c - 0.5 for c in [109.5] * 11 + [110.5, 99.0]])
        r3 = engine.step(df3, [], [h_triggered])
        h_stopped = r3.updated[0]
        assert h_stopped.state == HypothesisState.STOPPED
        assert h_stopped.outcome_price == 100.0

    def test_expiry_without_trigger(self):
        engine = HypothesisEngine(arm_proximity_pct=0.005, expiry_bars=5)
        df1 = _bar_df([105.0] * 11)  # pattern à end_index=10
        r1 = engine.step(df1, [_bull_pattern(end_idx=10)], [])
        h = r1.created[0]
        assert h.state == HypothesisState.FORMING

        # 6 bougies plus tard, toujours à 105 → expiré
        prices = [105.0] * 17  # last_idx=16, bars_since=16-10=6 > 5
        df2 = _bar_df(prices)
        r2 = engine.step(df2, [], [h])
        assert r2.updated[0].state == HypothesisState.EXPIRED

    def test_short_bear_pattern_triggers_down(self):
        engine = HypothesisEngine(arm_proximity_pct=0.015)
        # close=90.5 proche de 90 (breakout DOWN) — 0.56% au-dessus
        df1 = _bar_df([90.5] * 11)
        r1 = engine.step(df1, [_bear_pattern()], [])
        h = r1.created[0]
        assert h.side.value == "SHORT"
        assert h.state == HypothesisState.ARMED

        # close=89 < 90 → triggered
        df2 = _bar_df([90.5] * 11 + [89.0])
        r2 = engine.step(df2, [], [h])
        assert r2.updated[0].state == HypothesisState.TRIGGERED

    def test_no_duplicate_hypothesis_on_repeated_detection(self):
        """Si le même pattern est re-détecté tick suivant, on n'en crée pas un 2e."""
        engine = HypothesisEngine()
        df = _bar_df([105.0] * 11)
        r1 = engine.step(df, [_bull_pattern()], [])
        h = r1.created[0]

        # Re-détection du même pattern (même symbol/tf/kind/breakout)
        r2 = engine.step(df, [_bull_pattern()], [h])
        assert r2.created == []

    def test_symmetrical_pattern_does_not_spawn(self):
        engine = HypothesisEngine()
        df = _bar_df([105.0] * 11)
        sym_pattern = ChartPatternDTO(
            kind=PatternKind.TRIANGLE_SYM,
            symbol="TEST/USDT",
            timeframe="1h",
            start_index=0,
            end_index=10,
            start_timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            end_timestamp=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=10),
            breakout_level=110.0,
            invalidation_level=100.0,
            breakout_direction=BreakoutDirection.UNDETERMINED,
            height=10.0,
            target=None,
            confidence=0.7,
        )
        r = engine.step(df, [sym_pattern], [])
        assert r.created == [], "SYM non directionnel ne devrait pas spawner d'hypothèse"
