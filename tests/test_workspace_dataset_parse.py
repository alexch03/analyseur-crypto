"""Parsing des noms de CSV workspace (symbole + timeframe)."""

from __future__ import annotations

from app.services.dashboard_workspace import (
    infer_symbol_from_workspace_dataset_basename,
    infer_timeframe_from_workspace_dataset_basename,
    parse_workspace_dataset_basename_parts,
)


def test_parse_eth_5m():
    p = parse_workspace_dataset_basename_parts("ETH_USDT__5m__15d.csv")
    assert p == ("ETH_USDT", "5m")


def test_infer_symbol_eth():
    assert infer_symbol_from_workspace_dataset_basename("ETH_USDT__5m__15d.csv") == "ETH/USDT"


def test_infer_timeframe_matches():
    assert infer_timeframe_from_workspace_dataset_basename("ETH_USDT__5m__15d.csv") == "5m"


def test_infer_symbol_fdusd_suffix():
    assert infer_symbol_from_workspace_dataset_basename("FOO_FDUSD__1h__30d.csv") == "FOO/FDUSD"


def test_non_conventional_name_returns_none():
    assert infer_symbol_from_workspace_dataset_basename("random.csv") is None
    assert infer_timeframe_from_workspace_dataset_basename("random.csv") is None
