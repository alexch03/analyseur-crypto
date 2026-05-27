"""Tests for calendar period → bar count helpers."""

from app.services.period_utils import (
    bars_for_calendar_days,
    bars_for_period,
    calendar_bar_count_uncapped,
    preset_to_days,
    timeframe_bar_seconds,
)


def test_timeframe_bar_seconds():
    assert timeframe_bar_seconds("5m") == 300
    assert timeframe_bar_seconds("15m") == 900
    assert timeframe_bar_seconds("1h") == 3600
    assert timeframe_bar_seconds("4h") == 14400
    assert timeframe_bar_seconds("1d") == 86400


def test_preset_to_days():
    assert preset_to_days("30d") == 30.0
    assert preset_to_days("CUSTOM") is None
    assert preset_to_days("custom") is None


def test_bars_for_period_1h_7d():
    n = bars_for_period("7d", "1h", min_bars=10, max_bars=10000)
    assert n == 168


def test_bars_for_period_respects_max():
    n = bars_for_period("365d", "15m", min_bars=10, max_bars=500)
    assert n == 500


def test_bars_for_period_custom_min():
    n = bars_for_period("custom", "1h", min_bars=300, max_bars=20000)
    assert n == 300


def test_calendar_bar_count_uncapped_1h_90d():
    assert calendar_bar_count_uncapped("90d", "1h", min_bars=300) == 2160


def test_calendar_bar_count_uncapped_custom_is_none():
    assert calendar_bar_count_uncapped("custom", "1h") is None


def test_bars_for_calendar_days_50d_1h():
    assert bars_for_calendar_days(50, "1h", min_bars=100, max_bars=20000) == 1200


def test_bars_for_calendar_days_respects_max():
    assert bars_for_calendar_days(400, "15m", min_bars=100, max_bars=500) == 500
