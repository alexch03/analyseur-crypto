"""Tests for local OHLCV CSV helpers."""

from datetime import UTC, datetime

import pandas as pd

import app.services.ohlcv_data_store as store


def test_save_load_roundtrip_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(store.settings, "ohlcv_data_dir", str(tmp_path))
    ts = pd.Timestamp("2024-01-01T00:00:00", tz=UTC)
    df = pd.DataFrame(
        {
            "timestamp": [ts, ts + pd.Timedelta(hours=1)],
            "open": [1.0, 1.1],
            "high": [1.2, 1.3],
            "low": [0.9, 1.0],
            "close": [1.1, 1.2],
            "volume": [10.0, 11.0],
        }
    )
    path = store.save_ohlcv_csv(df, "BTC/USDT", "1h")
    assert "BTC_USDT__1h.csv" in path
    loaded = store.load_ohlcv_csv("BTC/USDT", "1h", limit=None)
    assert len(loaded) == 2
    tail = store.load_ohlcv_csv("BTC/USDT", "1h", limit=1)
    assert len(tail) == 1


def test_file_dataset_status(monkeypatch, tmp_path):
    monkeypatch.setattr(store.settings, "ohlcv_data_dir", str(tmp_path))
    st = store.file_dataset_status("ETH/USDT", "4h")
    assert st["exists"] is False
    df = pd.DataFrame(
        {
            "timestamp": [datetime(2024, 6, 1, tzinfo=UTC)],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
        }
    )
    store.save_ohlcv_csv(df, "ETH/USDT", "4h")
    st2 = store.file_dataset_status("ETH/USDT", "4h")
    assert st2["exists"] is True
    assert st2["bars"] == 1
    assert st2["last_candle_open_utc"] is not None
    assert st2["age_seconds_since_last_candle"] is not None
