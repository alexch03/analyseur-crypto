"""Runtime control state for local dashboard.

The state is persisted to a local JSON file so the dashboard can keep
its toggles/configuration across restarts without editing `.env`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import settings

STATE_FILE = Path(".runtime_control.json")

# Défaut dashboard : fenêtre en nombre de bougies (backtest_limit / optimize_limit), pas un calendrier implicite.
DEFAULT_ANALYSIS_PERIOD_PRESET = "custom"

# Révision des défauts moteur / grille (incrémenter pour ré-appliquer best_engine + optimization_grid au chargement).
ENGINE_DEFAULTS_REVISION = 4

DEFAULT_BEST_ENGINE_PARAMS: dict[str, Any] = {
    "rr_min": 2.0,
    "fvg_proximity_pct": 0.004,
    "ob_proximity_pct": 0.004,
    "max_setups": 5,
    "swing_left": 3,
    "swing_right": 3,
}

DEFAULT_OPTIMIZATION_GRID: dict[str, Any] = {
    "rr_min_values": [1.8, 2.0, 2.5],
    "fvg_proximity_values": [0.003, 0.005],
    "ob_proximity_values": [0.003, 0.005],
    "swing_left_values": [2, 3],
    "swing_right_values": [2, 3],
    "max_setups_values": [3, 5],
}


def _default_state() -> dict[str, Any]:
    return {
        "enabled": True,
        "auto_parameters": True,
        "telegram_enabled": False,
        "symbols": list(settings.symbols),
        "timeframes": list(settings.timeframes),
        "scan_limit": 500,
        "backtest_limit": 1200,
        "optimize_limit": 1200,
        # Période : « custom » = limites explicites ; les presets 7d/30d/… servent surtout live / DB / CSV auto.
        "analysis_period_preset": DEFAULT_ANALYSIS_PERIOD_PRESET,
        "period_max_bars": 20000,
        # Données pour backtest / optimisation : live (CCXT), file (CSV local), database (PostgreSQL).
        "backtest_ohlcv_source": "live",
        "top_optimization": 5,
        "training_bars": 120,
        "max_holding_bars": 120,
        # Combinaisons de setups évaluées par bougie replay (1 = seulement la meilleure confiance).
        "max_setups_per_bar": 1,
        "unit_size": 1.0,
        "entry_fee_rate": 0.0004,
        "exit_fee_rate": 0.0004,
        "funding_rate_8h": 0.0,
        # Frais / funding : si True, surcharge les champs manuels avec l'exchange (CCXT).
        "auto_fee_from_exchange": False,
        "fee_market_type": "swap",
        "auto_funding_from_exchange": False,
        "optimization_objective": "net_pnl_quote",
        "optimization_strategy": "exhaustive",
        "optimization_max_trials": 200,
        "optimization_grid": dict(DEFAULT_OPTIMIZATION_GRID),
        "best_engine_params": dict(DEFAULT_BEST_ENGINE_PARAMS),
        "engine_defaults_revision": ENGINE_DEFAULTS_REVISION,
        "last_scan": None,
        "last_backtest": None,
        "last_optimization": None,
        "last_optimization_batch": None,
        "last_ohlcv_fetch": None,
        "last_wf_oos": None,
        "wf_splits": 3,
        # Replay : trailing (None ou 0 = désactivé). Break-even retiré du produit.
        "replay_break_even_r": None,
        "replay_trail_after_r": None,
        "replay_trail_atr_mult": None,
        "replay_trail_atr_period": 14,
        # TIMEOUT : si PnL latent > 0 + Bollinger + tendance (SMA) favorables, prolonger (pas de sortie forcée).
        "replay_timeout_smart_extend": True,
        "replay_timeout_grace_bars": None,
        "replay_timeout_max_extensions": 3,
        "replay_timeout_bb_period": 20,
        "replay_timeout_sma_fast": 10,
        "replay_timeout_sma_slow": 20,
        # Filtres entrée (pipeline setups).
        "require_ifvg_confluence": False,
        "ifvg_confluence_pct": 0.008,
        "require_rsi_divergence": False,
        # Workspace : listes déroulantes (fichiers sur disque), persistées.
        "dashboard_dataset_file": "__live__",
        "dashboard_method_file": None,
        # Paper live (boucle CCXT + analyse ; pas d’ordres DB dans cette version).
        "paper_live_running": False,
        "paper_live_interval_sec": 90,
        "paper_live_symbol": None,
        "paper_live_timeframe": None,
        # Exécution paper : même flux pour replay ; ``bitget_futures_sim`` = crochet export Bitget (ordres réels à brancher).
        "paper_execution_backend": "sim_replay",
        # CCXT exchange_id pour les bougies paper uniquement (ex. bitget) ; vide = .env EXCHANGE_ID.
        "paper_ohlcv_exchange_id": None,
        "paper_live": None,
        "updated_at": datetime.now(tz=UTC).isoformat(),
    }


def load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        state = _default_state()
        save_state(state)
        return state
    with STATE_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Forward-compat default merge (sans écraser la sélection workspace si absente du JSON).
    defaults = dict(_default_state())
    dash_ds = defaults.pop("dashboard_dataset_file", "__live__")
    dash_meth = defaults.pop("dashboard_method_file", None)
    for k, v in defaults.items():
        data.setdefault(k, v)

    rev = int(data.get("engine_defaults_revision", 0))
    if rev < ENGINE_DEFAULTS_REVISION:
        if rev < 2:
            data["best_engine_params"] = dict(DEFAULT_BEST_ENGINE_PARAMS)
            data["optimization_grid"] = dict(DEFAULT_OPTIMIZATION_GRID)
        if rev < 3:
            data["replay_break_even_r"] = None
        if rev < 4:
            data["best_engine_params"] = dict(DEFAULT_BEST_ENGINE_PARAMS)
            data["optimization_grid"] = dict(DEFAULT_OPTIMIZATION_GRID)
        data["engine_defaults_revision"] = ENGINE_DEFAULTS_REVISION
        save_state(data)
    else:
        # Shallow merge nested dictionaries to keep backward compatibility.
        if isinstance(data.get("best_engine_params"), dict):
            merged = dict(DEFAULT_BEST_ENGINE_PARAMS)
            merged.update(data["best_engine_params"])
            data["best_engine_params"] = merged
        if isinstance(data.get("optimization_grid"), dict):
            merged_grid = dict(DEFAULT_OPTIMIZATION_GRID)
            merged_grid.update(data["optimization_grid"])
            data["optimization_grid"] = merged_grid
    # Backward compatibility for old key name.
    if "funding_rate_8h" not in data and "funding_rate_per_bar" in data:
        data["funding_rate_8h"] = data["funding_rate_per_bar"]
    if "dashboard_dataset_file" not in data or not data.get("dashboard_dataset_file"):
        legacy = str(data.get("backtest_ohlcv_source", "live")).lower().strip()
        data["dashboard_dataset_file"] = {
            "live": "__live__",
            "file": "__auto_csv__",
            "database": "__database__",
        }.get(legacy, dash_ds)
    data.setdefault("dashboard_method_file", dash_meth)
    # Ajoute les TF définis dans settings/.env sans retirer les TF déjà persistés (ex. nouveau 5m).
    cfg_tfs = [str(t).strip() for t in settings.timeframes if str(t).strip()]
    cur_tf = data.get("timeframes")
    if isinstance(cur_tf, list) and cfg_tfs:
        seen = {str(x).strip().lower() for x in cur_tf if str(x).strip()}
        to_add = [t for t in cfg_tfs if t.lower() not in seen]
        if to_add:
            data["timeframes"] = to_add + [str(x).strip() for x in cur_tf if str(x).strip()]
    return data


def save_state(state: dict[str, Any]) -> dict[str, Any]:
    state["updated_at"] = datetime.now(tz=UTC).isoformat()
    with STATE_FILE.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=True, indent=2)
    return state


def patch_state(patch: dict[str, Any]) -> dict[str, Any]:
    state = load_state()
    state.update(patch)
    return save_state(state)
