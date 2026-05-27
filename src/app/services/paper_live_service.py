"""Boucle paper « live » : poll CCXT, analyse SMC (même logique que scan) avec l’état + méthode JSON.

Contrôle via ``start`` / ``stop`` ; statut persisté dans ``.runtime_control.json`` (``paper_live``, ``paper_live_running``).

En plus du **journal des ticks** (résumé d’analyse à chaque cycle), une **simulation replay**
(:mod:`app.services.paper_live_simulation`) enregistre des **trades fictifs** avec les **mêmes règles**
que le backtest (entrée après le signal si le prix touche le niveau, SL/TP/break-even/trailing,
frais et funding selon l’état). Ce n’est pas un compte exchange.
"""

from __future__ import annotations

import asyncio
import html as html_module
import logging
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from app.config import settings
from app.ingestion.ccxt_fetcher import CCXTFetcher
from app.paper.engine_replay import replay_engine_from_bt_cfg
from app.services.analysis_pipeline import run_analysis
from app.services.control_state import load_state, patch_state
from app.services.dashboard_workspace import effective_run_state, method_trace_for_runs
from app.services.model_registry import get_active_model
from app.services.parameter_policy import resolve_smc_parameters
from app.services.paper_execution import run_paper_execution_step
from app.services.paper_live_simulation import setup_from_dict
from app.services.replay_runtime import build_replay_bt_config, resolve_trade_cost_rates
from app.schemas.domain import Side, TradeSetupDTO
from app.telegram.notifier import TelegramNotifier

logger = logging.getLogger(__name__)


def _trade_log_head_sig(log: list[Any]) -> tuple[Any, ...]:
    if not log or not isinstance(log[0], dict):
        return ()
    row = log[0]
    return (
        row.get("opened_at_utc"),
        row.get("exit_bar_utc"),
        row.get("net_pnl_quote"),
        row.get("outcome"),
    )


def paper_telegram_trade_events(prev_pl: dict[str, Any], sim_patch: dict[str, Any]) -> tuple[bool, bool]:
    """Détecte (entrée_replay_remplie, trade_clôturé_loggué) pour notifications Telegram ciblées."""
    merged = {**prev_pl, **sim_patch}
    prev_sig = _trade_log_head_sig(list(prev_pl.get("trade_log") or []))
    new_sig = _trade_log_head_sig(list(merged.get("trade_log") or []))
    prev_pos = prev_pl.get("sim_position")
    new_pos = merged.get("sim_position")
    entry_filled = not prev_pos and bool(new_pos)
    trade_closed = bool(new_sig) and new_sig != prev_sig
    return entry_filled, trade_closed


def _format_paper_entry_telegram(setup: TradeSetupDTO) -> str:
    direction = "LONG" if setup.side == Side.LONG else "SHORT"
    tps = " / ".join(f"{float(tp):.2f}" for tp in setup.take_profits)
    rat = html_module.escape((setup.rationale or "")[:220])
    return (
        "<b>Paper sim — Entrée</b>\n"
        f"<b>{html_module.escape(setup.symbol)}</b> · {html_module.escape(setup.timeframe)}\n"
        f"{direction} — <b>{html_module.escape(setup.setup_type)}</b>\n"
        f"Entry <b>{setup.entry:.2f}</b>  SL <b>{setup.stop_loss:.2f}</b>  TP {tps}\n"
        f"R:R {setup.risk_reward:.1f}  conf {setup.confidence:.0%}\n\n"
        f"<i>{rat}</i>"
    )


def _format_paper_close_telegram(row: dict[str, Any], symbol: str, timeframe: str) -> str:
    sym = html_module.escape(str(row.get("symbol") or symbol))
    tf = html_module.escape(str(row.get("timeframe") or timeframe))
    side = html_module.escape(str(row.get("side", "")))
    st = html_module.escape(str(row.get("setup_type", "")))
    oc = html_module.escape(str(row.get("outcome", "")))
    net = row.get("net_pnl_quote")
    bars = row.get("bars_held")
    return (
        "<b>Paper sim — Clôture</b>\n"
        f"<b>{sym}</b> · {tf}\n"
        f"{side} — <b>{st}</b>\n"
        f"Sortie: <b>{oc}</b>   PnL net: <b>{net}</b> quote   Bars: {bars}"
    )


async def _notify_paper_trade_events_if_needed(
    *,
    send_tg: bool,
    prev_pl: dict[str, Any],
    sim_patch: dict[str, Any],
    symbol: str,
    timeframe: str,
) -> None:
    """Telegram uniquement sur entrée / clôture trade simulé (pas sur chaque analyse)."""
    if not send_tg:
        return
    entry_filled, trade_closed = paper_telegram_trade_events(prev_pl, sim_patch)
    if not entry_filled and not trade_closed:
        return
    merged = {**prev_pl, **sim_patch}
    token = settings.telegram_bot_token
    chat = settings.telegram_chat_id
    if not token or not chat:
        return
    notifier = TelegramNotifier(token, chat)
    try:
        if entry_filled:
            ow = merged.get("sim_replay_open") or {}
            sd = ow.get("setup_dict")
            if isinstance(sd, dict):
                setup = setup_from_dict(sd)
                await notifier.send_plain_html(_format_paper_entry_telegram(setup))
        if trade_closed:
            row0 = (merged.get("trade_log") or [None])[0]
            if isinstance(row0, dict):
                await notifier.send_plain_html(_format_paper_close_telegram(row0, symbol, timeframe))
    except Exception:
        logger.exception("paper_live: notification Telegram (trade simulé)")


_lock = asyncio.Lock()
_stop_event = asyncio.Event()
_worker_task: asyncio.Task[None] | None = None


async def reconcile_paper_live_task_with_state() -> None:
    """Réaligne la tâche asyncio sur l'état disque (``.runtime_control.json``).

    Après un **redémarrage** du serveur, ``paper_live_running`` peut rester ``true`` alors que
    ``_worker_task`` est ``None`` : le dashboard affiche « Transition » (run sans tâche).
    Si ``enabled`` est encore vrai, on **recrée** le worker ; sinon on remet le flag à false.
    """
    global _worker_task
    async with _lock:
        st = load_state()
        running_flag = bool(st.get("paper_live_running"))
        enabled = bool(st.get("enabled"))
        t = _worker_task
        task_alive = t is not None and not t.done()

        if running_flag and not task_alive:
            if enabled:
                _stop_event.clear()
                loop = asyncio.get_running_loop()
                _worker_task = loop.create_task(_paper_live_worker(), name="paper_live_worker")
                logger.warning(
                    "paper_live: worker recréé (paper_live_running=true, enabled=true, sans tâche vivante).",
                )
            else:
                patch_state({"paper_live_running": False})
                logger.warning(
                    "paper_live: paper_live_running forcé à false (pas de tâche et enabled=false).",
                )


async def _fetch_df(fetcher: CCXTFetcher, symbol: str, tf: str, limit: int):
    rows = await fetcher.fetch_ohlcv(symbol, tf, limit=limit)
    if not rows:
        return None
    return pd.DataFrame(
        [
            {
                "timestamp": r.ts_open,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in rows
        ]
    )


_DEFAULT_ENGINE_PARAMS: dict[str, Any] = {
    "rr_min": 2.0,
    "fvg_proximity_pct": 0.003,
    "ob_proximity_pct": 0.003,
    "max_setups": 5,
    "swing_left": 3,
    "swing_right": 3,
}


def resolve_paper_live_market(state: dict[str, Any]) -> dict[str, Any]:
    """Symbole, timeframe et profondeur OHLCV utilisés par le worker (règles uniques)."""
    eff = effective_run_state(state)
    sym_raw = state.get("paper_live_symbol")
    tf_raw = state.get("paper_live_timeframe")
    syms = list(eff.get("symbols") or state.get("symbols") or [])
    tfs = list(eff.get("timeframes") or state.get("timeframes") or [])
    sym = (str(sym_raw).strip() if sym_raw else None) or (syms[0] if syms else None)
    tf = (str(tf_raw).strip() if tf_raw else None) or (tfs[0] if tfs else None)
    limit = min(int(eff.get("scan_limit") or state.get("scan_limit") or 500), 1500)
    sym_src = "paper_live_symbol" if sym_raw and str(sym_raw).strip() else ("symbols[0]" if sym else None)
    tf_src = "paper_live_timeframe" if tf_raw and str(tf_raw).strip() else ("timeframes[0]" if tf else None)
    if sym and tf:
        data_fr = (
            f"Chaque cycle : CCXT ({settings.exchange_id}) charge {limit} bougies {tf} "
            f"(fenêtre glissante incluant la dernière clôture), puis analyse SMC sur tout le DataFrame."
        )
    else:
        data_fr = (
            "Symbole ou timeframe manquant : renseigne les champs paper, ou symbols + timeframes dans la méthode."
        )
    return {
        "effective_symbol": sym,
        "effective_timeframe": tf,
        "bars_fetched_per_tick": limit,
        "symbol_resolution": sym_src,
        "timeframe_resolution": tf_src,
        "data_pipeline_fr": data_fr,
    }


async def _paper_live_worker() -> None:
    global _worker_task
    fetcher: CCXTFetcher | None = None
    last_ccxt_exchange: str | None = None
    try:
        while not _stop_event.is_set():
            state = load_state()
            if not state.get("enabled"):
                patch_state(
                    {
                        "paper_live_running": False,
                        "paper_live": {
                            **(state.get("paper_live") or {}),
                            "last_tick_utc": datetime.now(tz=UTC).isoformat(),
                            "detail": "Système désactivé (Contrôle). Paper live arrêté.",
                        },
                    }
                )
                break

            eff = effective_run_state(state)
            ex_ccxt = str(eff.get("paper_ohlcv_exchange_id") or "").strip().lower() or str(
                settings.exchange_id
            ).strip().lower()
            if fetcher is None or last_ccxt_exchange != ex_ccxt:
                if fetcher is not None:
                    await fetcher.close()
                fetcher = CCXTFetcher(ex_ccxt)
                last_ccxt_exchange = ex_ccxt
            meta = resolve_paper_live_market(state)
            sym = meta["effective_symbol"]
            tf = meta["effective_timeframe"]
            limit = int(meta["bars_fetched_per_tick"])
            interval = int(state.get("paper_live_interval_sec") or 90)
            interval = max(15, min(600, interval))

            tick_utc = datetime.now(tz=UTC).isoformat()
            trace = method_trace_for_runs(state)

            if not sym or not tf:
                patch_state(
                    {
                        "paper_live": {
                            **(state.get("paper_live") or {}),
                            "last_tick_utc": tick_utc,
                            "error": "Définis paper_live_symbol / paper_live_timeframe ou symbols+timeframes dans la méthode.",
                            **trace,
                        },
                    }
                )
                try:
                    await asyncio.wait_for(_stop_event.wait(), timeout=float(interval))
                except TimeoutError:
                    pass
                continue

            active_model = get_active_model()
            manual_params = (
                active_model["params"] if active_model else eff.get("best_engine_params") or _DEFAULT_ENGINE_PARAMS
            )
            auto_enabled = bool(eff.get("auto_parameters", True)) and active_model is None

            try:
                df = await _fetch_df(fetcher, sym, tf, limit)
                if df is None or df.empty:
                    patch_state(
                        {
                            "paper_live": {
                                **(state.get("paper_live") or {}),
                                "last_tick_utc": tick_utc,
                                "symbol": sym,
                                "timeframe": tf,
                                "setups_count": 0,
                                "detail": "Pas de bougies (exchange ou limite).",
                                **trace,
                            },
                        }
                    )
                else:
                    smc_params = resolve_smc_parameters(
                        timeframe=tf,
                        ohlcv_df=df,
                        auto_enabled=auto_enabled,
                        manual_params=manual_params,
                    )
                    engine_params = {
                        "rr_min": smc_params["rr_min"],
                        "fvg_proximity_pct": smc_params["fvg_proximity_pct"],
                        "ob_proximity_pct": smc_params["ob_proximity_pct"],
                        "max_setups": smc_params["max_setups"],
                    }
                    send_tg = bool(
                        settings.telegram_bot_token
                        and settings.telegram_chat_id
                        and eff.get("paper_live_send_telegram", True)
                    )
                    # Pas de Telegram sur chaque poll : uniquement entrée / clôture trade simulé (voir ci-dessous).
                    out = await run_analysis(
                        df,
                        sym,
                        tf,
                        swing_left=int(smc_params["swing_left"]),
                        swing_right=int(smc_params["swing_right"]),
                        send_telegram=False,
                        render_chart_img=False,
                        engine_params=engine_params,
                    )
                    top = out.setups[0] if out.setups else None
                    st_pl = load_state()
                    prev_pl: dict[str, Any] = dict(st_pl.get("paper_live") or {})
                    costs = await resolve_trade_cost_rates(eff, sym, exchange_id=ex_ccxt)
                    bt_cfg = build_replay_bt_config(eff, costs)
                    replay_engine = replay_engine_from_bt_cfg(bt_cfg)
                    sim_patch = run_paper_execution_step(
                        df,
                        symbol=sym,
                        timeframe=tf,
                        tick_wall_utc_iso=tick_utc,
                        top=top,
                        prev_pl=prev_pl,
                        engine=replay_engine,
                        eff=eff,
                    )
                    await _notify_paper_trade_events_if_needed(
                        send_tg=send_tg,
                        prev_pl=prev_pl,
                        sim_patch=sim_patch,
                        symbol=sym,
                        timeframe=tf,
                    )
                    prev_pl.update(sim_patch)
                    tick_log: list[dict[str, Any]] = list(prev_pl.get("tick_log") or [])
                    row: dict[str, Any] = {
                        "utc": tick_utc,
                        "symbol": sym,
                        "timeframe": tf,
                        "bars": len(df),
                        "trend": out.context.trend.value,
                        "setups": len(out.setups),
                    }
                    if top:
                        row["top_type"] = top.setup_type
                        row["top_side"] = top.side.value
                        row["top_conf"] = round(float(top.confidence), 4)
                        row["top_rr"] = round(float(top.risk_reward), 4)
                        row["top_entry"] = round(float(top.entry), 6)
                        row["top_sl"] = round(float(top.stop_loss), 6)
                    tick_log.insert(0, row)
                    tick_log = tick_log[:40]
                    prev_pl.update(
                        {
                            "last_tick_utc": tick_utc,
                            "symbol": sym,
                            "timeframe": tf,
                            "bars": len(df),
                            "trend": out.context.trend.value,
                            "setups_count": len(out.setups),
                            "top_setup": (
                                {
                                    "type": top.setup_type,
                                    "side": top.side.value,
                                    "confidence": top.confidence,
                                    "rr": top.risk_reward,
                                }
                                if top
                                else None
                            ),
                            "tick_log": tick_log,
                            **trace,
                        }
                    )
                    patch_state({"paper_live": prev_pl})
            except Exception as e:  # noqa: BLE001
                logger.exception("paper_live tick")
                patch_state(
                    {
                        "paper_live": {
                            **(load_state().get("paper_live") or {}),
                            "last_tick_utc": tick_utc,
                            "symbol": sym,
                            "timeframe": tf,
                            "last_error": str(e),
                            **trace,
                        },
                    }
                )

            try:
                await asyncio.wait_for(_stop_event.wait(), timeout=float(interval))
            except TimeoutError:
                pass
    finally:
        if fetcher is not None:
            await fetcher.close()
        st = load_state()
        patch_state(
            {
                "enabled": False,
                "paper_live_running": False,
                "paper_live": {
                    **(st.get("paper_live") or {}),
                    "stopped_at_utc": datetime.now(tz=UTC).isoformat(),
                },
            }
        )
        _worker_task = None


async def start_paper_live() -> dict[str, Any]:
    """Démarre la boucle ; active ``enabled`` (même rôle que l’ancien interrupteur dashboard)."""
    global _worker_task
    async with _lock:
        state = load_state()
        if _worker_task is not None and not _worker_task.done():
            return {"ok": False, "detail": "Paper live déjà en cours."}
        meta = resolve_paper_live_market(state)
        sym = meta["effective_symbol"]
        tf = meta["effective_timeframe"]
        persist: dict[str, Any] = {"enabled": True}
        if sym and not (state.get("paper_live_symbol") and str(state.get("paper_live_symbol")).strip()):
            persist["paper_live_symbol"] = sym
        if tf and not (state.get("paper_live_timeframe") and str(state.get("paper_live_timeframe")).strip()):
            persist["paper_live_timeframe"] = tf
        patch_state(persist)
        _stop_event.clear()
        loop = asyncio.get_running_loop()
        _worker_task = loop.create_task(_paper_live_worker(), name="paper_live_worker")
        patch_state({"paper_live_running": True})
    return {
        "ok": True,
        "paper_live_running": True,
        "paper_live_market": meta,
    }


async def stop_paper_live() -> dict[str, Any]:
    global _worker_task
    async with _lock:
        _stop_event.set()
        t = _worker_task
        if t is not None and not t.done():
            try:
                await asyncio.wait_for(t, timeout=12.0)
            except TimeoutError:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        _worker_task = None
    patch_state({"enabled": False})
    return {"ok": True, "paper_live_running": False}


def paper_live_status() -> dict[str, Any]:
    st = load_state()
    eff = effective_run_state(st)
    running = bool(st.get("paper_live_running"))
    t = _worker_task
    task_alive = t is not None and not t.done()
    meta = resolve_paper_live_market(st)
    return {
        "paper_live_running": running,
        "task_alive": task_alive,
        "enabled": bool(st.get("enabled")),
        "paper_live": st.get("paper_live"),
        "paper_live_symbol": st.get("paper_live_symbol"),
        "paper_live_timeframe": st.get("paper_live_timeframe"),
        "paper_live_interval_sec": st.get("paper_live_interval_sec"),
        "paper_execution_backend": eff.get("paper_execution_backend", "sim_replay"),
        "paper_ohlcv_exchange_id": eff.get("paper_ohlcv_exchange_id"),
        "effective_symbol": meta["effective_symbol"],
        "effective_timeframe": meta["effective_timeframe"],
        "bars_fetched_per_tick": meta["bars_fetched_per_tick"],
        "symbol_resolution": meta["symbol_resolution"],
        "timeframe_resolution": meta["timeframe_resolution"],
        "data_pipeline_fr": meta["data_pipeline_fr"],
    }
