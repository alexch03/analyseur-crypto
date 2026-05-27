"""Détection des événements Telegram paper (entrée / clôture), sans appel réseau."""

from __future__ import annotations

from app.services.paper_live_service import paper_telegram_trade_events


def test_entry_filled_no_close() -> None:
    prev: dict = {"sim_position": None, "trade_log": []}
    patch = {"sim_position": {"side": "LONG"}, "sim_replay_open": {"setup_dict": {"symbol": "X"}}}
    e, c = paper_telegram_trade_events(prev, patch)
    assert e is True
    assert c is False


def test_close_only() -> None:
    prev = {
        "sim_position": {"side": "LONG"},
        "trade_log": [
            {
                "opened_at_utc": "a",
                "exit_bar_utc": "b",
                "net_pnl_quote": 1.0,
                "outcome": "TP",
            }
        ],
    }
    patch = {
        "sim_position": None,
        "trade_log": [
            {
                "opened_at_utc": "c",
                "exit_bar_utc": "d",
                "net_pnl_quote": -2.0,
                "outcome": "SL",
            }
        ],
    }
    e, c = paper_telegram_trade_events(prev, patch)
    assert e is False
    assert c is True


def test_no_event_pending_only() -> None:
    prev: dict = {"sim_position": None, "trade_log": []}
    patch = {"sim_replay_pending": {"x": 1}, "sim_position": None}
    e, c = paper_telegram_trade_events(prev, patch)
    assert e is False
    assert c is False


def test_no_event_same_trade_log_head() -> None:
    row = {"opened_at_utc": "a", "exit_bar_utc": "b", "net_pnl_quote": 1.0, "outcome": "TP"}
    prev = {"sim_position": {"open": True}, "trade_log": [row]}
    patch = {"sim_position": {"open": True}, "trade_log": [row]}
    e, c = paper_telegram_trade_events(prev, patch)
    assert e is False
    assert c is False
