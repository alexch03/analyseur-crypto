"""Tests workspace CSV helpers."""

from __future__ import annotations

import pandas as pd

import app.services.dashboard_workspace as dw


def test_suggest_workspace_dataset_basename():
    assert dw.suggest_workspace_dataset_basename("BTC/USDT", "1h", 50) == "BTC_USDT__1h__50d.csv"


def test_prepare_workspace_ohlcv_full_file():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-01"], utc=True),
            "open": [1.0, 1.0],
        }
    )
    out, meta = dw.prepare_workspace_ohlcv_for_analysis(df, {"rows_loaded": 2})
    assert len(out) == 2
    assert meta["workspace_load_policy"] == "full_file"
    assert meta["analysis_bars_load_limit"] == 2
    assert out["timestamp"].iloc[0] <= out["timestamp"].iloc[1]


def test_prepare_workspace_ohlcv_safety_cap(monkeypatch):
    monkeypatch.setattr(dw, "WORKSPACE_CSV_SAFETY_MAX_ROWS", 3)
    ts = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "open": [1.0] * 5})
    out, meta = dw.prepare_workspace_ohlcv_for_analysis(df, {"rows_loaded": 5})
    assert len(out) == 3
    assert meta["workspace_load_policy"] == "truncated_safety_max"
    assert meta["analysis_bars_load_limit"] == 3


def test_is_workspace_csv_dataset_selected():
    assert dw.is_workspace_csv_dataset_selected({"dashboard_dataset_file": "x.csv"}) is True
    assert dw.is_workspace_csv_dataset_selected({"dashboard_dataset_file": "__live__"}) is False
    assert dw.is_workspace_csv_dataset_selected({"dashboard_dataset_file": ""}) is False


def test_infer_timeframe_from_workspace_dataset_basename():
    assert dw.infer_timeframe_from_workspace_dataset_basename("BTC_USDT__1d__700d.csv") == "1d"
    assert dw.infer_timeframe_from_workspace_dataset_basename("X__5m__10d.csv") == "5m"
    assert dw.infer_timeframe_from_workspace_dataset_basename("weird.csv") is None
    assert dw.infer_timeframe_from_workspace_dataset_basename("") is None


def test_save_workspace_dataset_df(monkeypatch, tmp_path):
    monkeypatch.setattr(dw, "workspace_datasets_dir", lambda: tmp_path)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
            "open": [1, 1, 1],
            "high": [2, 2, 2],
            "low": [0.5, 0.5, 0.5],
            "close": [1.5, 1.5, 1.5],
            "volume": [10, 10, 10],
        }
    )
    meta = dw.save_workspace_dataset_df(df, "test__1h__1d.csv")
    assert meta["ok"] is True
    assert (tmp_path / "test__1h__1d.csv").is_file()
    assert meta.get("merged_with_existing") is False


def test_save_workspace_dataset_merge(monkeypatch, tmp_path):
    monkeypatch.setattr(dw, "workspace_datasets_dir", lambda: tmp_path)
    name = "merge__1h__1d.csv"
    df1 = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
            "open": [1, 1],
            "high": [2, 2],
            "low": [0.5, 0.5],
            "close": [1.5, 1.5],
            "volume": [10, 10],
        }
    )
    assert dw.save_workspace_dataset_df(df1, name)["ok"] is True
    df2 = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01 01:00:00", periods=2, freq="h", tz="UTC"),
            "open": [9, 1],
            "high": [9, 2],
            "low": [9, 0.5],
            "close": [9, 1.5],
            "volume": [9, 11],
        }
    )
    meta = dw.save_workspace_dataset_df(df2, name, merge_with_existing=True)
    assert meta["ok"] is True
    assert meta["merged_with_existing"] is True
    assert meta["rows"] == 3
    reread = pd.read_csv(tmp_path / name, parse_dates=["timestamp"])
    assert len(reread) == 3
    # keep=last sur timestamp : la dernière ligne du concat [ancien, nouveau] l’emporte (ici 01:00 vient du nouveau lot).
    row_01 = reread[reread["timestamp"] == pd.Timestamp("2024-01-01 01:00:00+00:00")]
    assert float(row_01["volume"].iloc[0]) == 9.0
    row_02 = reread[reread["timestamp"] == pd.Timestamp("2024-01-01 02:00:00+00:00")]
    assert float(row_02["volume"].iloc[0]) == 11.0


def test_merge_method_overlay_symbols_timeframes_scan_limit():
    base = {"symbols": ["ETH/USDT"], "timeframes": ["4h"], "scan_limit": 500, "best_engine_params": {}}
    method = {"symbols": ["BTC/USDT"], "timeframes": ["1h"], "scan_limit": 800}
    out = dw.merge_method_overlay(base, method)
    assert out["symbols"] == ["BTC/USDT"]
    assert out["timeframes"] == ["1h"]
    assert out["scan_limit"] == 800


def test_method_trace_for_runs_empty():
    assert dw.method_trace_for_runs({})["method_overlay_applied"] is False


def test_method_trace_for_runs_with_file(monkeypatch, tmp_path):
    monkeypatch.setattr(dw, "workspace_methods_dir", lambda: tmp_path)
    (tmp_path / "trace.json").write_text('{"best_engine_params": {"rr_min": 1}}', encoding="utf-8")
    st = {"dashboard_method_file": "trace.json"}
    tr = dw.method_trace_for_runs(st)
    assert tr["method_file"] == "trace.json"
    assert tr["method_overlay_applied"] is True
    assert tr["method_sha256"] and len(tr["method_sha256"]) == 64
