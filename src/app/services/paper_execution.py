"""Point d'entrée unique pour l'exécution « paper » : replay aujourd'hui, Bitget futures demain.

Le worker paper live appelle toujours ``run_paper_execution_step`` : aujourd'hui les backends
``sim_replay`` et ``bitget_futures_sim`` délèguent au même moteur replay (mêmes entrées/sorties).
Le mode ``bitget_futures_sim`` sert de **crochet** : mêmes données d'état et métadonnées pour
brancher plus tard ``create_order`` / WebSocket sur Bitget sans changer la forme du flux
(signal → pending → position → trade_log).
"""

from __future__ import annotations

from typing import Any

from app.config import settings
from app.paper.engine_replay import ReplayBacktestEngine
from app.services.paper_live_simulation import step_paper_simulation

PAPER_EXECUTION_BACKENDS = frozenset({"sim_replay", "bitget_futures_sim"})


def resolve_paper_execution_backend(eff: dict[str, Any]) -> str:
    raw = str(eff.get("paper_execution_backend") or "sim_replay").strip().lower()
    return raw if raw in PAPER_EXECUTION_BACKENDS else "sim_replay"


def resolve_paper_ohlcv_exchange_id(eff: dict[str, Any]) -> str:
    o = str(eff.get("paper_ohlcv_exchange_id") or "").strip().lower()
    return o if o else str(settings.exchange_id).strip().lower()


def run_paper_execution_step(
    df: Any,
    *,
    symbol: str,
    timeframe: str,
    tick_wall_utc_iso: str,
    top: Any,
    prev_pl: dict[str, Any],
    engine: ReplayBacktestEngine,
    eff: dict[str, Any],
) -> dict[str, Any]:
    backend = resolve_paper_execution_backend(eff)
    patch = step_paper_simulation(
        df,
        symbol=symbol,
        timeframe=timeframe,
        tick_wall_utc_iso=tick_wall_utc_iso,
        top=top,
        prev_pl=prev_pl,
        engine=engine,
    )
    ex_used = resolve_paper_ohlcv_exchange_id(eff)
    patch["paper_execution_backend_resolved"] = backend
    patch["paper_ohlcv_exchange_id_resolved"] = ex_used
    if backend == "bitget_futures_sim":
        patch["sim_broker_plan_fr"] = (
            "Mode préparation Bitget futures (simulation replay) : "
            "entrées/sorties = moteur backtest ; bougies CCXT = paper_ohlcv_exchange_id ; "
            "prochaine étape = envoi d'ordres réels (CCXT create_order / private WS) derrière ce backend."
        )
    return patch
