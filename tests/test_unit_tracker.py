"""Tests pour le tracker paper unit-based."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.paper.unit_tracker import (
    UnitTracker,
    UnitTradeDTO,
    compute_pct_gain,
    reconcile_with_engine_step,
)
from app.schemas.domain import Side
from app.schemas.hypothesis import HypothesisDTO, HypothesisState
from app.schemas.patterns import BreakoutDirection, ChartPatternDTO, PatternKind


def _make_pattern(side_up: bool = True) -> ChartPatternDTO:
    return ChartPatternDTO(
        kind=PatternKind.TRIANGLE_ASC if side_up else PatternKind.TRIANGLE_DESC,
        symbol="BTC/USDT",
        timeframe="1h",
        start_index=0,
        end_index=10,
        start_timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        end_timestamp=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=10),
        breakout_level=100.0,
        invalidation_level=90.0 if side_up else 110.0,
        breakout_direction=BreakoutDirection.UP if side_up else BreakoutDirection.DOWN,
        height=10.0,
        target=110.0 if side_up else 90.0,
        confidence=0.7,
    )


def _make_hypothesis(state: HypothesisState, *, triggered=None, outcome=None, side_up=True):
    p = _make_pattern(side_up=side_up)
    return HypothesisDTO(
        id="h-1",
        pattern=p,
        symbol=p.symbol,
        timeframe=p.timeframe,
        side=Side.LONG if side_up else Side.SHORT,
        entry_price=100.0,
        target_price=110.0 if side_up else 90.0,
        invalidation_price=90.0 if side_up else 110.0,
        state=state,
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
        updated_at=datetime(2025, 1, 1, tzinfo=UTC),
        triggered_at=triggered[0] if triggered else None,
        triggered_price=triggered[1] if triggered else None,
        closed_at=outcome[0] if outcome else None,
        outcome_price=outcome[1] if outcome else None,
    )


class TestPctGain:
    def test_long_gain(self):
        assert abs(compute_pct_gain(Side.LONG, 100.0, 110.0) - 10.0) < 1e-9

    def test_long_loss(self):
        assert abs(compute_pct_gain(Side.LONG, 100.0, 95.0) - -5.0) < 1e-9

    def test_short_gain(self):
        # Entry 100, exit 90 : SHORT gagne 10/90 = 11.111...%
        assert abs(compute_pct_gain(Side.SHORT, 100.0, 90.0) - 11.1111) < 0.01

    def test_short_loss(self):
        # Entry 100, exit 110 : SHORT perd 10/110 ≈ -9.09%
        assert abs(compute_pct_gain(Side.SHORT, 100.0, 110.0) - -9.0909) < 0.01

    def test_invalid_prices_return_zero(self):
        assert compute_pct_gain(Side.LONG, 0.0, 100.0) == 0.0
        assert compute_pct_gain(Side.LONG, 100.0, 0.0) == 0.0


class TestUnitTracker:
    def test_open_from_triggered_hypothesis(self):
        h = _make_hypothesis(
            HypothesisState.TRIGGERED,
            triggered=(datetime(2025, 1, 2, tzinfo=UTC), 101.0),
        )
        t = UnitTracker.open_from_hypothesis(h)
        assert t is not None
        assert t.entry_price == 101.0
        assert t.side == Side.LONG
        assert t.symbol == "BTC/USDT"
        assert t.pattern_kind == PatternKind.TRIANGLE_ASC
        assert not t.is_closed

    def test_open_returns_none_if_not_triggered(self):
        h = _make_hypothesis(HypothesisState.ARMED)
        assert UnitTracker.open_from_hypothesis(h) is None

    def test_close_on_target_hit(self):
        h_trigger = _make_hypothesis(
            HypothesisState.TRIGGERED,
            triggered=(datetime(2025, 1, 2, tzinfo=UTC), 101.0),
        )
        t = UnitTracker.open_from_hypothesis(h_trigger)
        assert t is not None

        h_done = _make_hypothesis(
            HypothesisState.TARGET_HIT,
            triggered=(datetime(2025, 1, 2, tzinfo=UTC), 101.0),
            outcome=(datetime(2025, 1, 3, tzinfo=UTC), 110.0),
        )
        closed = UnitTracker.close_from_hypothesis(t, h_done)
        assert closed is not None
        assert closed.is_closed
        # (110/101 - 1)*100 = 8.910891...
        assert closed.pct_gain is not None
        assert abs(closed.pct_gain - 8.9109) < 0.01
        assert closed.outcome == "TARGET_HIT"

    def test_close_on_stopped(self):
        t = UnitTracker.open_from_hypothesis(_make_hypothesis(
            HypothesisState.TRIGGERED,
            triggered=(datetime(2025, 1, 2, tzinfo=UTC), 101.0),
        ))
        assert t is not None
        h_done = _make_hypothesis(
            HypothesisState.STOPPED,
            triggered=(datetime(2025, 1, 2, tzinfo=UTC), 101.0),
            outcome=(datetime(2025, 1, 3, tzinfo=UTC), 90.0),
        )
        closed = UnitTracker.close_from_hypothesis(t, h_done)
        assert closed is not None
        assert closed.pct_gain is not None
        assert closed.pct_gain < 0
        assert closed.outcome == "STOPPED"


class TestCumulativeStats:
    def _trade(self, pct: float, *, side=Side.LONG, opened=True) -> UnitTradeDTO:
        return UnitTradeDTO(
            id=f"t-{pct}",
            hypothesis_id="h",
            symbol="BTC/USDT",
            timeframe="1h",
            side=side,
            pattern_kind=PatternKind.TRIANGLE_ASC,
            entry_price=100.0,
            entry_timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            exit_price=100.0 * (1 + pct / 100.0) if opened else None,
            exit_timestamp=datetime(2025, 1, 2, tzinfo=UTC) if opened else None,
            pct_gain=pct if opened else None,
            outcome="TARGET_HIT" if pct > 0 else "STOPPED" if pct < 0 else None,
        )

    def test_empty_trades(self):
        stats = UnitTracker.compute_cumulative([])
        assert stats.total_trades == 0
        assert stats.cumulative_simple_pct == 0.0
        assert stats.cumulative_compound_pct == 0.0

    def test_cumulative_simple_is_sum(self):
        trades = [self._trade(10.0), self._trade(-5.0), self._trade(20.0)]
        stats = UnitTracker.compute_cumulative(trades)
        assert stats.cumulative_simple_pct == 25.0
        assert stats.win_count == 2
        assert stats.loss_count == 1
        assert stats.win_rate == round(2 / 3, 4)
        assert stats.best_pct == 20.0
        assert stats.worst_pct == -5.0

    def test_cumulative_compound(self):
        # +10% puis -5% puis +20% : (1.10)(0.95)(1.20) = 1.254
        # → 25.4% en compound
        trades = [self._trade(10.0), self._trade(-5.0), self._trade(20.0)]
        stats = UnitTracker.compute_cumulative(trades)
        assert abs(stats.cumulative_compound_pct - 25.4) < 0.01

    def test_open_trades_excluded_from_stats(self):
        trades = [self._trade(10.0), self._trade(20.0, opened=False)]
        stats = UnitTracker.compute_cumulative(trades)
        assert stats.closed_trades == 1
        assert stats.open_trades == 1
        assert stats.cumulative_simple_pct == 10.0


class TestReconcile:
    def test_opens_trade_for_newly_triggered(self):
        h = _make_hypothesis(
            HypothesisState.TRIGGERED,
            triggered=(datetime(2025, 1, 2, tzinfo=UTC), 101.0),
        )
        still, closed, opened = reconcile_with_engine_step([], [h])
        assert len(opened) == 1
        assert opened[0].hypothesis_id == h.id
        assert closed == []
        assert still == []

    def test_closes_trade_when_hypothesis_completes(self):
        t_existing = UnitTracker.open_from_hypothesis(_make_hypothesis(
            HypothesisState.TRIGGERED,
            triggered=(datetime(2025, 1, 2, tzinfo=UTC), 101.0),
        ))
        assert t_existing is not None
        h_done = _make_hypothesis(
            HypothesisState.TARGET_HIT,
            triggered=(datetime(2025, 1, 2, tzinfo=UTC), 101.0),
            outcome=(datetime(2025, 1, 3, tzinfo=UTC), 110.0),
        )
        still, closed, opened = reconcile_with_engine_step([t_existing], [h_done])
        assert len(closed) == 1
        assert closed[0].is_closed
        assert opened == []
        assert still == []
