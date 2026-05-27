"""Simulation paper live alignée sur le moteur de replay (backtest).

- **Signal** : dernier setup SMC (top) sur la **dernière bougie** clôturée, comme une barre ``i``
  du walk-forward.
- **Entrée** : première bougie **après** le signal où ``low <= entry <= high`` (identique à
  ``ReplayBacktestEngine._simulate_one_trade``).
- **Sortie** : SL / TP / trailing / TIMEOUT avec les mêmes règles intrabar et les mêmes
  frais + funding que ``_finalize_trade``.

Une seule position ou un seul pending à la fois ; état JSON dans ``paper_live``.
Le **routeur** (:mod:`app.services.paper_execution`) choisit le backend (replay / préparation Bitget) sans changer ce module.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.paper.engine_replay import ReplayBacktestEngine, _timeframe_to_hours
from app.schemas.domain import Side, TradeSetupDTO

_TRADE_LOG_MAX = 40


def _ts_series(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["timestamp"], utc=True)


def _norm_ts_key(ts: Any) -> str:
    return pd.Timestamp(ts).isoformat()


def _find_index_for_ts(ts_series: pd.Series, key: str) -> int | None:
    target = pd.Timestamp(key)
    for i in range(len(ts_series)):
        if ts_series.iloc[i] == target:
            return i
    return None


def setup_to_dict(s: TradeSetupDTO) -> dict[str, Any]:
    ts = s.timestamp
    ts_s = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
    return {
        "symbol": s.symbol,
        "timeframe": s.timeframe,
        "side": s.side.value,
        "entry": float(s.entry),
        "stop_loss": float(s.stop_loss),
        "take_profits": [float(x) for x in s.take_profits],
        "risk_reward": float(s.risk_reward),
        "confidence": float(s.confidence),
        "setup_type": s.setup_type,
        "timestamp": ts_s,
        "rationale": s.rationale,
        "payload": dict(s.payload) if isinstance(s.payload, dict) else {},
    }


def setup_from_dict(d: dict[str, Any]) -> TradeSetupDTO:
    ts_raw = d.get("timestamp", "")
    ts_parsed = pd.Timestamp(ts_raw).to_pydatetime() if ts_raw else pd.Timestamp.utcnow().to_pydatetime()
    side_raw = d.get("side", "LONG")
    side = Side(str(side_raw).upper()) if str(side_raw).upper() in ("LONG", "SHORT") else Side.LONG
    tps = d.get("take_profits") or []
    return TradeSetupDTO(
        symbol=str(d.get("symbol", "")),
        timeframe=str(d.get("timeframe", "")),
        side=side,
        entry=float(d.get("entry", 0.0)),
        stop_loss=float(d.get("stop_loss", 0.0)),
        take_profits=[float(x) for x in tps],
        risk_reward=float(d.get("risk_reward", 0.0)),
        confidence=float(d.get("confidence", 0.0)),
        setup_type=str(d.get("setup_type", "")),
        timestamp=ts_parsed,
        rationale=str(d.get("rationale", "")),
        payload=d.get("payload") if isinstance(d.get("payload"), dict) else {},
    )


def _trade_to_log_row(tr: Any, tick_wall_utc_iso: str) -> dict[str, Any]:
    return {
        "utc": tick_wall_utc_iso,
        "exit_bar_utc": tr.closed_at,
        "opened_at_utc": tr.opened_at,
        "side": tr.side.value,
        "setup_type": tr.setup_type,
        "entry": round(float(tr.entry), 8),
        "exit": round(float(tr.close_price), 8),
        "outcome": tr.outcome,
        "bars_held": tr.bars_held,
        "gross_pnl_quote": round(float(tr.gross_pnl_quote), 8),
        "net_pnl_quote": round(float(tr.net_pnl_quote), 8),
        "fees_open_quote": round(float(tr.fees_open_quote), 8),
        "fees_close_quote": round(float(tr.fees_close_quote), 8),
        "funding_quote": round(float(tr.funding_quote), 8),
        "replay_aligned": True,
    }


def _fix_trade_row_symbol(row: dict[str, Any], symbol: str, timeframe: str) -> dict[str, Any]:
    row = dict(row)
    row["symbol"] = symbol
    row["timeframe"] = timeframe
    return row


def _open_position_ui(
    setup: TradeSetupDTO,
    exit_state: dict[str, Any],
    ts_series: pd.Series,
) -> dict[str, Any]:
    ei = int(exit_state["entry_idx"])
    return {
        "side": setup.side.value,
        "entry": float(setup.entry),
        "stop_loss": float(setup.stop_loss),
        "take_profit": float(setup.take_profits[0]) if setup.take_profits else None,
        "setup_type": setup.setup_type,
        "opened_bar_ts": _norm_ts_key(ts_series.iloc[ei]),
        "replay_mode": True,
    }


def step_paper_simulation(
    df: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    tick_wall_utc_iso: str,
    top: TradeSetupDTO | None,
    prev_pl: dict[str, Any],
    engine: ReplayBacktestEngine,
) -> dict[str, Any]:
    trade_log: list[dict[str, Any]] = list(prev_pl.get("trade_log") or [])
    skip_entry_bar: str | None = prev_pl.get("sim_skip_entry_until_bar_ts")
    cum = float(prev_pl.get("sim_cumulative_net_pnl_quote") or 0.0)
    note_fr = "—"

    ts_series = _ts_series(df)
    n = len(df)
    if n < 3:
        return {
            "sim_position": None,
            "sim_replay_pending": prev_pl.get("sim_replay_pending"),
            "sim_replay_open": prev_pl.get("sim_replay_open"),
            "trade_log": trade_log,
            "sim_skip_entry_until_bar_ts": skip_entry_bar,
            "sim_cumulative_net_pnl_quote": cum,
            "sim_execution_note_fr": "Pas assez de bougies pour la simulation replay.",
            "sim_replay_last_bar_ts": prev_pl.get("sim_replay_last_bar_ts"),
        }

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    opens = df["open"].to_numpy(dtype=float) if "open" in df.columns else None
    timestamps = df["timestamp"]
    last_key = _norm_ts_key(ts_series.iloc[-1])
    bar_hours = _timeframe_to_hours(timeframe)

    if skip_entry_bar and _norm_ts_key(skip_entry_bar) != last_key:
        skip_entry_bar = None

    pending: dict[str, Any] | None = prev_pl.get("sim_replay_pending")
    open_wrap: dict[str, Any] | None = prev_pl.get("sim_replay_open")
    sim_position: dict[str, Any] | None = None
    if open_wrap is not None:
        pending = None

    if top is None:
        pending = None

    if pending is not None and top is not None and pending.get("signal_bar_ts") == last_key:
        pending = {**pending, "setup_dict": setup_to_dict(top)}

    # --- Pending : recherche entrée (avant la gestion de position ouverte pour pouvoir ouvrir puis avancer au même tick) ---
    if open_wrap is None and pending is not None:
        setup = setup_from_dict(pending["setup_dict"])
        sig_ts = str(pending.get("signal_bar_ts") or "")
        sig_idx = _find_index_for_ts(ts_series, sig_ts)
        if sig_idx is None:
            pending = None
            note_fr = "Signal replay expiré (bougie de signal hors fenêtre)."
        elif skip_entry_bar and _norm_ts_key(skip_entry_bar) == last_key:
            note_fr = "Pas de nouvelle entrée replay sur la bougie de sortie précédente."
        else:
            entry_idx = engine.seek_entry_after_signal(setup, sig_idx, highs, lows, n)
            if entry_idx is not None:
                sig_i = int(pending.get("signal_index", entry_idx))
                ex0 = engine.paper_live_init_exit_state(
                    setup,
                    entry_idx,
                    highs,
                    lows,
                    closes,
                    signal_index=sig_i,
                    n_bars=n,
                )
                if ex0 is None:
                    pending = None
                    note_fr = "Setup invalide pour le replay (TP ou risque)."
                else:
                    open_wrap = {
                        "setup_dict": pending["setup_dict"],
                        "exit_state": ex0,
                        "signal_index": sig_idx,
                        "entry_bar_ts": _norm_ts_key(ts_series.iloc[entry_idx]),
                    }
                    pending = None
                    sim_position = _open_position_ui(setup, ex0, ts_series)
                    note_fr = "Entrée replay exécutée (touch du prix sur bougie après signal)."
            else:
                note_fr = "En attente : aucune bougie après le signal ne touche encore le prix d'entrée."

    # --- Position ouverte (état de sortie partagé avec le replay) ---
    if open_wrap is not None:
        setup = setup_from_dict(open_wrap["setup_dict"])
        exit_state: dict[str, Any] = dict(open_wrap["exit_state"])
        entry_bar_ts = str(open_wrap.get("entry_bar_ts") or "")
        entry_idx = _find_index_for_ts(ts_series, entry_bar_ts) if entry_bar_ts else int(
            exit_state.get("entry_idx", n - 1),
        )

        if entry_idx is None:
            ei = min(int(exit_state.get("entry_idx", n - 1)), n - 1)
            tr = engine.paper_live_timeout_at_bar(
                setup, ei, n - 1, highs, lows, closes, timestamps, bar_hours
            )
            cum += float(tr.net_pnl_quote)
            trade_log.insert(0, _fix_trade_row_symbol(_trade_to_log_row(tr, tick_wall_utc_iso), symbol, timeframe))
            trade_log[0]["outcome"] = "DATA_GAP"
            trade_log[0]["note_fr"] = "Bougie d'entrée absente de la fenêtre OHLCV."
            trade_log = trade_log[:_TRADE_LOG_MAX]
            open_wrap = None
            skip_entry_bar = _norm_ts_key(ts_series.iloc[n - 1])
            note_fr = "Position replay fermée (DATA_GAP fenêtre)."
        else:
            exit_state["entry_idx"] = entry_idx
            sig_idx = int(open_wrap.get("signal_index", entry_idx))
            max_bar = n - 1
            tr_done, exit_state2 = engine.paper_live_advance_exit_state(
                setup,
                exit_state,
                highs,
                lows,
                closes,
                timestamps,
                bar_hours,
                max_bar_idx_inclusive=max_bar,
                opens=opens,
            )
            if tr_done is not None:
                cum += float(tr_done.net_pnl_quote)
                trade_log.insert(
                    0,
                    _fix_trade_row_symbol(_trade_to_log_row(tr_done, tick_wall_utc_iso), symbol, timeframe),
                )
                trade_log = trade_log[:_TRADE_LOG_MAX]
                open_wrap = None
                ci = int(tr_done.closed_index)
                skip_entry_bar = _norm_ts_key(ts_series.iloc[ci]) if 0 <= ci < len(ts_series) else last_key
                note_fr = f"Trade replay clôturé ({tr_done.outcome})."
            elif exit_state2 is not None and int(exit_state2["j"]) > int(exit_state2.get("deadline_idx", max_bar)):
                dln = int(exit_state2["deadline_idx"])
                tr_to = engine.paper_live_timeout_at_bar(
                    setup, entry_idx, dln, highs, lows, closes, timestamps, bar_hours
                )
                cum += float(tr_to.net_pnl_quote)
                trade_log.insert(
                    0,
                    _fix_trade_row_symbol(_trade_to_log_row(tr_to, tick_wall_utc_iso), symbol, timeframe),
                )
                trade_log = trade_log[:_TRADE_LOG_MAX]
                open_wrap = None
                skip_entry_bar = _norm_ts_key(ts_series.iloc[dln]) if 0 <= dln < len(ts_series) else last_key
                note_fr = "Clôture TIMEOUT (max holding ou fin de prolongation)."
            else:
                open_wrap = {**open_wrap, "exit_state": exit_state2 or exit_state}
                sim_position = _open_position_ui(setup, open_wrap["exit_state"], ts_series)
                note_fr = "Position replay ouverte — sorties SL/TP/trail comme le backtest."

    if (
        pending is None
        and open_wrap is None
        and top is not None
        and not (skip_entry_bar and _norm_ts_key(skip_entry_bar) == last_key)
    ):
        pending = {
            "signal_bar_ts": last_key,
            "signal_index": n - 1,
            "setup_dict": setup_to_dict(top),
        }
        note_fr = "Signal replay enregistré (top setup sur dernière bougie)."

    if open_wrap is not None and sim_position is None:
        setup = setup_from_dict(open_wrap["setup_dict"])
        sim_position = _open_position_ui(setup, open_wrap["exit_state"], ts_series)

    return {
        "sim_position": sim_position,
        "sim_replay_pending": pending,
        "sim_replay_open": open_wrap,
        "trade_log": trade_log,
        "sim_skip_entry_until_bar_ts": skip_entry_bar,
        "sim_cumulative_net_pnl_quote": round(cum, 8),
        "sim_execution_note_fr": note_fr,
        "sim_replay_last_bar_ts": last_key,
    }
