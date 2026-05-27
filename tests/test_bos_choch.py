"""Tests for BOS / CHOCH structure detection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from app.market_structure.bos_choch import detect_bos_choch
from app.market_structure.swings import detect_swings
from app.schemas.domain import StructureEventType, SwingKind, SwingPoint, Trend


class TestBOSCHOCH:
    def test_bullish_trend_produces_bos(self, tiny_ohlcv_bull):
        swings = detect_swings(tiny_ohlcv_bull, left=2, right=2)
        closes = tiny_ohlcv_bull["close"]
        events = detect_bos_choch(swings, closes)

        bos_events = [e for e in events if e.event_type == StructureEventType.BOS]
        assert len(bos_events) >= 1, "Bullish trend should produce at least one BOS"
        for b in bos_events:
            assert b.direction in (Trend.BULLISH, Trend.BEARISH)

    def test_choch_detected_on_lower_low(self, tiny_ohlcv_choch):
        swings = detect_swings(tiny_ohlcv_choch, left=2, right=2)
        closes = tiny_ohlcv_choch["close"]
        events = detect_bos_choch(swings, closes)

        choch_events = [e for e in events if e.event_type == StructureEventType.CHOCH]
        assert len(choch_events) >= 1, "CHOCH should fire when lower low breaks bull sequence"
        assert choch_events[0].direction == Trend.BEARISH

    def test_too_few_swings_returns_empty(self):
        swings = [
            SwingPoint(
                index=2,
                timestamp=datetime(2025, 1, 1, tzinfo=UTC),
                price=100.0,
                kind=SwingKind.HIGH,
            )
        ]
        closes = pd.Series([100.0] * 5)
        assert detect_bos_choch(swings, closes) == []

    def test_manual_choch_sequence(self):
        """Manually build a swing sequence that should trigger a CHOCH."""
        base = datetime(2025, 1, 1, tzinfo=UTC)
        swings = [
            SwingPoint(index=1, timestamp=base + timedelta(hours=1), price=90.0, kind=SwingKind.LOW),
            SwingPoint(index=3, timestamp=base + timedelta(hours=3), price=110.0, kind=SwingKind.HIGH),
            SwingPoint(index=5, timestamp=base + timedelta(hours=5), price=95.0, kind=SwingKind.LOW),
            SwingPoint(index=7, timestamp=base + timedelta(hours=7), price=115.0, kind=SwingKind.HIGH),
            # This LOW is below the previous LOW (95) → CHOCH
            SwingPoint(index=9, timestamp=base + timedelta(hours=9), price=88.0, kind=SwingKind.LOW),
        ]
        closes = pd.Series(
            [100, 90, 100, 110, 105, 95, 100, 115, 110, 88, 85],
            index=pd.date_range(base, periods=11, freq="h"),
        )

        events = detect_bos_choch(swings, closes)
        choch = [e for e in events if e.event_type == StructureEventType.CHOCH]
        assert len(choch) >= 1
        assert choch[0].direction == Trend.BEARISH
        assert choch[0].swing_ref.price == 88.0
