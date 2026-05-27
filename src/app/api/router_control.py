"""Control plane API for local dashboard (state + actions + monitoring)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from statistics import median
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import SessionDep
from app.config import settings
from app.ingestion.candle_metadata import ensure_symbol_timeframe_ids
from app.ingestion.candle_writer import CandleWriter
from app.ingestion.ccxt_fetcher import CCXTFetcher
from app.paper.engine_replay import replay_engine_from_bt_cfg
from app.services.analysis_pipeline import run_analysis
from app.services.control_state import (
    DEFAULT_ANALYSIS_PERIOD_PRESET,
    DEFAULT_BEST_ENGINE_PARAMS,
    DEFAULT_OPTIMIZATION_GRID,
    load_state,
    patch_state,
)
from app.services.model_registry import (
    add_model,
    get_active_model,
    load_registry,
    set_active_model,
    update_model,
)
from app.services.optimization_batch import (
    allocate_autoopt_output_filename,
    build_effective_state_for_batch_job,
    save_autoopt_method_variant,
    slug_for_autoopt_filename,
)
from app.services.optimizer import optimize_setup_parameters, report_optimization_export_metrics
from app.services.walk_forward_out_of_sample import run_walk_forward_oos
from app.services.parameter_policy import resolve_smc_parameters
from app.services.replay_runtime import build_replay_bt_config, resolve_trade_cost_rates
from app.services.period_utils import (
    bars_for_calendar_days,
    bars_for_period,
    calendar_bar_count_uncapped,
    preset_to_days,
    timeframe_bar_seconds,
)
from app.services.exchange_market_quotes import fetch_market_info_bundle
from app.services import paper_live_service
from app.services.dashboard_workspace import (
    apply_optimized_params_to_workspace_method,
    effective_run_state,
    infer_symbol_from_workspace_dataset_basename,
    infer_timeframe_from_workspace_dataset_basename,
    is_workspace_csv_dataset_selected,
    load_workspace_dataset_csv,
    method_json_path,
    method_trace_for_runs,
    prepare_workspace_ohlcv_for_analysis,
    safe_dataset_basename,
    safe_method_basename,
    save_workspace_dataset_df,
    suggest_workspace_dataset_basename,
    workspace_snapshot,
)
from app.services.ohlcv_data_store import (
    db_dataset_status,
    file_dataset_status,
    load_ohlcv_csv,
    load_ohlcv_from_db,
    ohlcv_dir,
    save_ohlcv_csv,
    ohlcv_csv_path,
)

router = APIRouter(prefix="/control", tags=["control"])


def _normalize_exchange_symbol(raw: str) -> str:
    """Accepte ``BTCUSDT`` ou ``btc/usdt`` et renvoie une forme CCXT ``BASE/QUOTE``."""
    s = (raw or "").strip()
    if not s or "/" in s:
        return s
    up = s.upper()
    for quote in ("USDT", "USDC", "BUSD", "EUR", "USD"):
        if up.endswith(quote) and len(up) > len(quote) + 1:
            return f"{up[: -len(quote)]}/{quote}"
    return s


_ALLOWED_PERIOD_PRESETS = frozenset({"7d", "30d", "90d", "180d", "365d", "custom"})
_OHLCV_SOURCES = frozenset({"live", "file", "database"})
_FEE_MARKET_TYPES = frozenset({"spot", "swap"})
_ALLOWED_OPTIMIZATION_STRATEGIES = frozenset(
    {"exhaustive", "random", "coordinate_descent"},
)

_OPTIMIZATION_RANKING_FR: dict[str, str] = {
    "net_pnl_quote": "Rang 1 = meilleur PnL net cumulé (quote), après frais et funding.",
    "net_r": "Rang 1 = meilleure somme des R (PnL net / risque par trade).",
    "profit_factor": "Rang 1 = meilleur profit factor (somme gains nets ÷ somme pertes nets).",
    "expectancy_r": "Rang 1 = meilleure espérance en R par trade.",
    "sharpe_like": "Rang 1 = meilleur ratio espérance R / drawdown R (indicateur simplifié).",
    "composite": (
        "Rang 1 = meilleur score composite (PnL, profit factor, espérance R, somme R, "
        "drawdown pénalisé, volume de trades modéré) — plusieurs métriques à la fois."
    ),
    "penalized_pnl_quote": (
        "Rang 1 = meilleur PnL net quote après pénalités explicites : drawdown quote, "
        "trop peu de trades, profit factor sous 1, sur-trading (même unité que le PnL)."
    ),
    "penalized_net_pnl": (
        "Alias de penalized_pnl_quote : même score pénalisé."
    ),
}


def _optimization_run_diagnostics(
    ranked: list[Any],
    *,
    rr_vals: list[float],
    fvg_vals: list[float],
    ob_vals: list[float],
    sl_vals: list[int],
    sr_vals: list[int],
    ms_vals: list[int],
    strategy: str,
) -> dict[str, Any]:
    """Résumé interprétable : taille de grille, nombre d'exécutions, signal grossier de robustesse (trades)."""

    def nz_len(xs: list[Any]) -> int:
        return max(1, len(xs)) if xs else 1

    cart = (
        nz_len(rr_vals)
        * nz_len(fvg_vals)
        * nz_len(ob_vals)
        * nz_len(sl_vals)
        * nz_len(sr_vals)
        * nz_len(ms_vals)
    )
    n_exec = len(ranked)
    out: dict[str, Any] = {
        "grid_cartesian_size": cart,
        "combinations_executed": n_exec,
        "strategy": strategy,
    }
    if not ranked:
        out["top1_total_trades"] = None
        out["median_total_trades_top5"] = None
        out["interpretation_fr"] = (
            f"Grille cartésienne ≈ {cart} combinaisons, mais aucun backtest n'a produit de résultat exploitable."
        )
        return out

    k = min(5, len(ranked))
    trades_slice = [int(ranked[i].report.total_trades) for i in range(k)]
    med_top = int(round(median(trades_slice)))
    t0 = int(ranked[0].report.total_trades)
    out["top1_total_trades"] = t0
    out["median_total_trades_top5"] = med_top

    if strategy == "exhaustive" and n_exec == cart:
        strat_note = f"{n_exec} backtests exécutés (grille pleine)."
    else:
        strat_note = (
            f"{n_exec} backtests exécutés ; la grille théorique compte {cart} points — en mode « {strategy} », "
            "l'exploration peut être partielle."
        )
    rob = (
        f"Robustesse (grossier) : le n°1 totalise {t0} trades ; médiane trades sur les {k} meilleurs scores : {med_top}. "
        "Un n°1 avec très peu de trades par rapport au reste invite à la prudence (échantillon faible / sur-ajustement)."
    )
    out["interpretation_fr"] = f"{strat_note} {rob}"
    return out


def _analysis_fetch_limit(state: dict[str, Any], timeframe: str, which: Literal["backtest", "optimize"]) -> int:
    preset = str(state.get("analysis_period_preset", DEFAULT_ANALYSIS_PERIOD_PRESET)).lower().strip()
    if preset not in _ALLOWED_PERIOD_PRESETS:
        preset = DEFAULT_ANALYSIS_PERIOD_PRESET
    max_cap = int(state.get("period_max_bars", 5000))
    max_cap = max(300, min(max_cap, 20000))
    if preset == "custom":
        key = "backtest_limit" if which == "backtest" else "optimize_limit"
        raw = int(state.get(key, 1200))
        floor = 300 if which == "backtest" else 500
        return max(floor, min(raw, 20000))
    return bars_for_period(preset, timeframe, min_bars=300, max_bars=max_cap)


def _effective_ohlcv_load_limit(
    state: dict[str, Any],
    timeframe: str,
    which: Literal["backtest", "optimize"],
) -> int:
    """Limite demandée pour charger les bougies (live / DB / CSV auto / presets).

    Pour un CSV **workspace**, cette valeur est ignorée par ``_load_backtest_df`` : tout le fichier
    est lu (troncature seulement au-delà du plafond technique ``WORKSPACE_CSV_SAFETY_MAX_ROWS``).
    """
    if is_workspace_csv_dataset_selected(state):
        return 0
    return _analysis_fetch_limit(state, timeframe, which)


def _response_bars_load_limit(ohlcv_meta: dict[str, Any], requested_limit: int) -> int:
    """Bougies réellement analysées (ex. fichier workspace entier)."""
    v = ohlcv_meta.get("analysis_bars_load_limit")
    if v is None:
        return int(requested_limit)
    return int(v)


def _analysis_window_preview(state: dict[str, Any], timeframe: str) -> dict[str, Any]:
    """Aide dashboard : bougies demandées vs plafond vs durée calendaire théorique."""
    preset = str(state.get("analysis_period_preset", DEFAULT_ANALYSIS_PERIOD_PRESET)).lower().strip()
    if preset not in _ALLOWED_PERIOD_PRESETS:
        preset = DEFAULT_ANALYSIS_PERIOD_PRESET
    max_cap = int(state.get("period_max_bars", 5000))
    max_cap = max(300, min(max_cap, 20000))
    if is_workspace_csv_dataset_selected(state):
        return {
            "preset": preset,
            "timeframe": timeframe,
            "period_max_bars": max_cap,
            "bars_backtest": None,
            "bars_optimize": None,
            "approx_days_of_bars_backtest": None,
            "truncated_by_cap": False,
            "hint_fr": (
                "Dataset workspace actif : le backtest et l’optimisation utilisent **toutes les bougies** du CSV "
                "(tri chronologique), sans appliquer « backtest_limit », « optimize_limit » ni « period_max_bars ». "
                "Seul un plafond technique (~350k lignes) peut tronquer un fichier anormalement volumineux."
            ),
        }
    bt = _analysis_fetch_limit(state, timeframe, "backtest")
    opt = _analysis_fetch_limit(state, timeframe, "optimize")
    bar_sec = timeframe_bar_seconds(timeframe)
    approx_days_loaded = (bt * bar_sec / 86400.0) if bar_sec > 0 else 0.0

    if preset == "custom":
        return {
            "preset": preset,
            "timeframe": timeframe,
            "period_max_bars": max_cap,
            "bars_backtest": bt,
            "bars_optimize": opt,
            "approx_days_of_bars_backtest": round(approx_days_loaded, 2),
            "truncated_by_cap": False,
            "hint_fr": (
                "Mode personnalisé : les bougies viennent des champs « Backtest » et « Optimisation » "
                "(minimum 300 pour backtest, 500 pour optim, maximum 20000). Pense à sauver l’état si besoin."
            ),
        }

    uncapped = calendar_bar_count_uncapped(preset, timeframe, min_bars=300)
    truncated = uncapped is not None and uncapped > max_cap
    hint = (
        f"Preset {preset} sur {timeframe} équivaut à ~{uncapped} bougies calendaires ; "
        f"avec plafond {max_cap}, le moteur charge {bt} bougies (~{approx_days_loaded:.1f} jours de bougies)."
    )
    if truncated:
        hint += (
            " Pour viser toute la période (ex. 365j en 1h ≈ 8760 bougies), augmente « Plafond bougies » "
            "jusqu’à au moins ce nombre."
        )
    return {
        "preset": preset,
        "timeframe": timeframe,
        "period_max_bars": max_cap,
        "bars_calendar_uncapped": uncapped,
        "bars_backtest": bt,
        "bars_optimize": opt,
        "approx_days_of_bars_backtest": round(approx_days_loaded, 2),
        "truncated_by_cap": truncated,
        "hint_fr": hint,
    }


def _period_run_audit(
    state: dict[str, Any],
    *,
    timeframe: str,
    which: Literal["backtest", "optimize"],
    limit: int,
    bars_loaded: int,
    df: pd.DataFrame,
    ohlcv_meta: dict[str, Any],
) -> dict[str, Any]:
    """Expose la différence entre preset / plafond / limite demandée et la série réellement chargée."""
    preset = str(state.get("analysis_period_preset", DEFAULT_ANALYSIS_PERIOD_PRESET)).lower().strip()
    if preset not in _ALLOWED_PERIOD_PRESETS:
        preset = DEFAULT_ANALYSIS_PERIOD_PRESET
    max_cap = int(state.get("period_max_bars", 5000))
    max_cap = max(300, min(max_cap, 20000))
    src = str(ohlcv_meta.get("ohlcv_source_used") or "")
    tf_sec = timeframe_bar_seconds(timeframe)
    approx_days_tf = (
        round((bars_loaded * tf_sec / 86400.0), 4) if bars_loaded > 0 and tf_sec > 0 else None
    )
    preset_days = None if preset == "custom" else preset_to_days(preset)
    uncapped = (
        calendar_bar_count_uncapped(preset, timeframe, min_bars=300)
        if preset != "custom"
        else None
    )
    span_days: float | None = None
    range_lbl: str | None = None
    if bars_loaded > 0 and "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna()
        if len(ts) >= 2:
            span_days = round(float((ts.max() - ts.min()).total_seconds() / 86400.0), 4)
            range_lbl = f"{ts.min().isoformat()} → {ts.max().isoformat()}"

    if src == "workspace_csv":
        policy = str(ohlcv_meta.get("workspace_load_policy") or "full_file")
        if policy == "truncated_safety_max":
            head_cfg = (
                f"Config : dataset workspace CSV — {bars_loaded} bougies analysées sur ce run ({which}) "
                f"(troncature de sécurité après {ohlcv_meta.get('workspace_safety_max_rows', '?')} lignes)."
            )
        else:
            head_cfg = (
                f"Config : dataset workspace CSV — {bars_loaded} bougies analysées ({which}), soit tout le fichier "
                "(les champs period_max_bars, backtest_limit et optimize_limit ne réduisent pas ce chargement)."
            )
    else:
        head_cfg = (
            f"Config : preset « {preset} », plafond période (max bougies analyse) = {max_cap}, "
            f"limite de chargement calculée pour ce run ({which}) = {limit} bougies."
        )
    parts: list[str] = [
        head_cfg,
        f"Série utilisée : {bars_loaded} bougies au pas {timeframe}"
        + (
            f" (~{approx_days_tf:.2f} jours cumulés au rythme du timeframe)."
            if approx_days_tf is not None
            else "."
        ),
    ]
    if preset_days is not None and uncapped is not None:
        parts.append(
            f"Le preset calendaire correspond à ~{preset_days:.0f} j ; sans plafond il faudrait ~{uncapped} bougies à ce TF."
        )
    if span_days is not None and range_lbl:
        parts.append(f"Fenêtre réelle (1er → dernier timestamp) : ~{span_days} j calendaires ({range_lbl}).")
    if bars_loaded < limit:
        if src == "workspace_csv":
            parts.append(
                "Moins de bougies que la limite : l'historique est surtout celui du CSV workspace "
                "(nombre de lignes) ; le preset seul n'allonge pas le fichier."
            )
        else:
            parts.append("Moins de bougies que la limite : historique disponible (exchange / DB / fichier) plus court.")
    elif preset != "custom" and uncapped is not None and limit < uncapped:
        parts.append(
            f"Troncature : la limite effective ({limit}) est inférieure aux ~{uncapped} bougies théoriques du preset "
            f"(plafond {max_cap})."
        )
    rows_disk = ohlcv_meta.get("rows_on_disk")
    if (
        src == "workspace_csv"
        and str(ohlcv_meta.get("workspace_load_policy")) == "truncated_safety_max"
        and rows_disk is not None
        and int(rows_disk) > int(bars_loaded)
    ):
        parts.append(
            f"CSV workspace : environ {int(rows_disk)} bougies sur disque ; analyse limitée aux {bars_loaded} "
            "dernières (plafond technique de sécurité)."
        )

    cap_out = int(bars_loaded) if src == "workspace_csv" else max_cap
    return {
        "period_analysis_max_bars_cap": cap_out,
        "bars_load_limit": limit,
        "bars_loaded": bars_loaded,
        "loaded_span_calendar_days": span_days,
        "loaded_timestamp_range_utc": range_lbl,
        "approx_days_from_timeframe_math": approx_days_tf,
        "bars_truncated_vs_load_limit": bars_loaded < limit,
        "period_context_fr": " ".join(parts),
    }


def _engine_params_filters(state: dict[str, Any]) -> dict[str, Any]:
    out = {
        "require_ifvg_confluence": bool(state.get("require_ifvg_confluence", False)),
        "ifvg_confluence_pct": float(state.get("ifvg_confluence_pct", 0.008)),
        "require_rsi_divergence": bool(state.get("require_rsi_divergence", False)),
    }
    if "chart_focus_last_bars" in state:
        out["chart_focus_last_bars"] = state["chart_focus_last_bars"]
    if "chart_compact_overlays" in state:
        out["chart_compact_overlays"] = state["chart_compact_overlays"]
    return out


def _normalize_ohlcv_source(state: dict[str, Any]) -> str:
    s = str(state.get("backtest_ohlcv_source", "live")).lower().strip()
    return s if s in _OHLCV_SOURCES else "live"


def _effective_timeframe_after_workspace_load(
    timeframe_requested: str,
    ohlcv_meta: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Aligne le TF du run sur le CSV workspace (convention ``…__1d__700d.csv``) si ça diffère du sélecteur."""
    req = (timeframe_requested or "").strip() or "1h"
    extra: dict[str, Any] = {}
    if ohlcv_meta.get("ohlcv_source_used") != "workspace_csv":
        return req, extra
    bn = ohlcv_meta.get("workspace_dataset_basename") or ohlcv_meta.get("dashboard_dataset_file")
    inf = infer_timeframe_from_workspace_dataset_basename(str(bn) if bn else None)
    if not inf:
        extra["timeframe_workspace_basename_note_fr"] = (
            f"Nom de CSV sans segment __tf__Nd : impossible d'inférer le pas ; utilisation du TF requête « {req} »."
        )
        return req, extra
    if inf.lower() != req.lower():
        extra["timeframe_requested"] = req
        extra["timeframe_inferred_from_workspace_file"] = inf
        extra["timeframe_alignment_note_fr"] = (
            f"Fichier « {bn} » : pas « {inf} » (nom), requête « {req} » (sélecteur). "
            f"SMC, funding par bougie et replay utilisent « {inf} »."
        )
        return inf, extra
    return req, extra


def _effective_symbol_after_workspace_load(
    symbol_requested: str,
    ohlcv_meta: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Aligne le symbole du run sur le CSV workspace (convention STEM__tf__Nd.csv) si ça diffère du sélecteur."""
    req_raw = (symbol_requested or "").strip() or "BTC/USDT"
    req = _normalize_exchange_symbol(req_raw)
    extra: dict[str, Any] = {}
    if ohlcv_meta.get("ohlcv_source_used") != "workspace_csv":
        return req, extra
    bn = ohlcv_meta.get("workspace_dataset_basename") or ohlcv_meta.get("dashboard_dataset_file")
    inf = infer_symbol_from_workspace_dataset_basename(str(bn) if bn else None)
    if not inf:
        extra["symbol_workspace_basename_note_fr"] = (
            f"Nom de CSV « {bn} » : impossible d'inférer la paire (attendu : SYMBOLE__tf__Nd.csv) ; "
            f"symbole requête « {req} » utilisé."
        )
        return req, extra
    inf_n = _normalize_exchange_symbol(inf)
    if inf_n.upper().replace(" ", "") != req.upper().replace(" ", ""):
        extra["symbol_requested"] = req
        extra["symbol_inferred_from_workspace_file"] = inf_n
        extra["symbol_alignment_note_fr"] = (
            f"Fichier « {bn} » : paire « {inf_n} » (nom), requête « {req} » (sélecteur). "
            f"Analyse, frais et replay utilisent « {inf_n} »."
        )
    return inf_n, extra


def _tag_ohlcv_analysis_limit(meta: dict[str, Any], df: pd.DataFrame) -> None:
    """Nombre de bougies réellement passées au moteur (réponse API + audit)."""
    meta["analysis_bars_load_limit"] = int(len(df))


async def _load_backtest_df(
    session: AsyncSession | None,
    state: dict[str, Any],
    symbol: str,
    timeframe: str,
    limit: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Charge les bougies : priorité à ``dashboard_dataset_file`` (workspace), sinon ``backtest_ohlcv_source``."""
    ds = str(state.get("dashboard_dataset_file") or "").strip()
    meta: dict[str, Any] = {
        "dashboard_dataset_file": ds or None,
        "legacy_backtest_ohlcv_source": state.get("backtest_ohlcv_source", "live"),
    }

    if ds == "__database__":
        if session is None:
            raise HTTPException(status_code=500, detail="Database OHLCV source requires a DB session")
        df = await load_ohlcv_from_db(session, symbol, timeframe, limit)
        meta["ohlcv_source_used"] = "database"
        meta["rows_loaded"] = len(df)
        _tag_ohlcv_analysis_limit(meta, df)
        return df, meta

    if ds == "__auto_csv__":
        df = load_ohlcv_csv(symbol, timeframe, limit)
        meta["ohlcv_source_used"] = "file_auto"
        meta["ohlcv_file_path"] = str(ohlcv_csv_path(symbol, timeframe).resolve())
        meta["rows_loaded"] = len(df)
        _tag_ohlcv_analysis_limit(meta, df)
        return df, meta

    if ds and ds not in ("__live__", ""):
        df, wmeta = load_workspace_dataset_csv(ds, None)
        df, wmeta = prepare_workspace_ohlcv_for_analysis(df, wmeta)
        meta.update(wmeta)
        meta["ohlcv_source_used"] = "workspace_csv"
        _tag_ohlcv_analysis_limit(meta, df)
        return df, meta

    if ds == "__live__":
        fetcher = CCXTFetcher(settings.exchange_id)
        try:
            df = await _fetch_df(fetcher, symbol, timeframe, limit)
        finally:
            await fetcher.close()
        meta["ohlcv_source_used"] = "live"
        meta["rows_loaded"] = len(df)
        _tag_ohlcv_analysis_limit(meta, df)
        return df, meta

    src = _normalize_ohlcv_source(state)
    meta["ohlcv_source_configured"] = state.get("backtest_ohlcv_source", "live")
    meta["ohlcv_source_used"] = src
    if src == "file":
        df = load_ohlcv_csv(symbol, timeframe, limit)
        meta["ohlcv_file_path"] = str(ohlcv_csv_path(symbol, timeframe).resolve())
        meta["rows_loaded"] = len(df)
        _tag_ohlcv_analysis_limit(meta, df)
        return df, meta
    if src == "database":
        if session is None:
            raise HTTPException(status_code=500, detail="Database OHLCV source requires a DB session")
        df = await load_ohlcv_from_db(session, symbol, timeframe, limit)
        meta["rows_loaded"] = len(df)
        _tag_ohlcv_analysis_limit(meta, df)
        return df, meta
    fetcher = CCXTFetcher(settings.exchange_id)
    try:
        df = await _fetch_df(fetcher, symbol, timeframe, limit)
    finally:
        await fetcher.close()
    meta["rows_loaded"] = len(df)
    _tag_ohlcv_analysis_limit(meta, df)
    return df, meta


class ControlStatePatch(BaseModel):
    enabled: bool | None = None
    auto_parameters: bool | None = None
    telegram_enabled: bool | None = None
    symbols: list[str] | None = None
    timeframes: list[str] | None = None
    scan_limit: int | None = Field(default=None, ge=100, le=5000)
    backtest_limit: int | None = Field(default=None, ge=300, le=20000)
    optimize_limit: int | None = Field(default=None, ge=500, le=20000)
    analysis_period_preset: str | None = None
    period_max_bars: int | None = Field(default=None, ge=300, le=20000)
    backtest_ohlcv_source: str | None = None
    top_optimization: int | None = Field(default=None, ge=1, le=20)
    wf_splits: int | None = Field(default=None, ge=1, le=12)
    training_bars: int | None = Field(default=None, ge=20, le=3000)
    max_holding_bars: int | None = Field(default=None, ge=5, le=3000)
    max_setups_per_bar: int | None = Field(default=None, ge=1, le=10)
    unit_size: float | None = Field(default=None, gt=0)
    entry_fee_rate: float | None = Field(default=None, ge=0)
    exit_fee_rate: float | None = Field(default=None, ge=0)
    funding_rate_8h: float | None = None
    auto_fee_from_exchange: bool | None = None
    fee_market_type: str | None = None
    auto_funding_from_exchange: bool | None = None
    optimization_objective: str | None = None
    optimization_strategy: str | None = None
    optimization_max_trials: int | None = Field(default=None, ge=10, le=10000)
    optimization_grid: dict[str, Any] | None = None
    best_engine_params: dict[str, Any] | None = None
    replay_trail_after_r: float | None = Field(default=None, ge=0, le=10)
    replay_trail_atr_mult: float | None = Field(default=None, ge=0, le=20)
    replay_trail_atr_period: int | None = Field(default=None, ge=2, le=500)
    replay_timeout_smart_extend: bool | None = None
    replay_timeout_grace_bars: int | None = Field(default=None, ge=0, le=3000)
    replay_timeout_max_extensions: int | None = Field(default=None, ge=0, le=20)
    replay_timeout_bb_period: int | None = Field(default=None, ge=5, le=100)
    replay_timeout_sma_fast: int | None = Field(default=None, ge=2, le=100)
    replay_timeout_sma_slow: int | None = Field(default=None, ge=2, le=200)
    paper_live_send_telegram: bool | None = None
    require_ifvg_confluence: bool | None = None
    ifvg_confluence_pct: float | None = Field(default=None, ge=0, le=0.2)
    require_rsi_divergence: bool | None = None
    dashboard_dataset_file: str | None = Field(default=None, max_length=240)
    dashboard_method_file: str | None = Field(default=None, max_length=240)
    paper_live_running: bool | None = None
    paper_live_interval_sec: int | None = Field(default=None, ge=15, le=600)
    paper_live_symbol: str | None = Field(default=None, max_length=80)
    paper_live_timeframe: str | None = Field(default=None, max_length=32)
    paper_execution_backend: str | None = Field(default=None, max_length=40)
    paper_ohlcv_exchange_id: str | None = Field(default=None, max_length=32)
    chart_focus_last_bars: int | None = None
    chart_compact_overlays: bool | None = None

    @field_validator("chart_focus_last_bars")
    @classmethod
    def _chart_focus_last_bars(cls, v: int | None) -> int | None:
        if v is None:
            return None
        return max(15, min(8000, int(v)))

    @field_validator("paper_execution_backend")
    @classmethod
    def _paper_execution_backend(cls, v: str | None) -> str | None:
        if v is None:
            return v
        s = v.strip().lower()
        if s not in ("sim_replay", "bitget_futures_sim"):
            return "sim_replay"
        return s

    @field_validator("paper_ohlcv_exchange_id")
    @classmethod
    def _paper_ohlcv_exchange_id(cls, v: str | None) -> str | None:
        if v is None:
            return v
        s = v.strip().lower()
        return s[:32] if s else None

    @field_validator("analysis_period_preset")
    @classmethod
    def _normalize_period_preset(cls, v: str | None) -> str | None:
        if v is None:
            return v
        p = v.strip().lower()
        if p not in _ALLOWED_PERIOD_PRESETS:
            raise ValueError("analysis_period_preset must be one of 7d,30d,90d,180d,365d,custom")
        return p

    @field_validator("backtest_ohlcv_source")
    @classmethod
    def _normalize_ohlcv_source_field(cls, v: str | None) -> str | None:
        if v is None:
            return v
        s = v.strip().lower()
        if s not in _OHLCV_SOURCES:
            raise ValueError("backtest_ohlcv_source must be one of live,file,database")
        return s

    @field_validator("fee_market_type")
    @classmethod
    def _fee_market_type(cls, v: str | None) -> str | None:
        if v is None:
            return v
        s = v.strip().lower()
        if s not in _FEE_MARKET_TYPES:
            raise ValueError("fee_market_type must be spot or swap")
        return s

    @field_validator("optimization_strategy")
    @classmethod
    def _optimization_strategy(cls, v: str | None) -> str | None:
        if v is None:
            return v
        s = v.strip().lower()
        if s not in _ALLOWED_OPTIMIZATION_STRATEGIES:
            raise ValueError(
                "optimization_strategy must be one of exhaustive, random, coordinate_descent"
            )
        return s


class ModelCreate(BaseModel):
    name: str
    params: dict[str, Any]


class ModelFromOptimization(BaseModel):
    rank: int = Field(ge=1, le=20)
    name: str | None = None


_MAX_OPTIM_BATCH_JOBS = 24


class OptimizeBatchJobIn(BaseModel):
    """Un job dans la file : même OHLCV pour tous ; overlay méthode + overrides optionnels."""

    source_method: str | None = Field(
        default=None,
        max_length=200,
        description="Basename du JSON dans data/dashboard_methods/ ; si absent, dashboard_method_file.",
    )
    optimization_objective: str | None = None
    optimization_strategy: str | None = None
    optimization_max_trials: int | None = Field(default=None, ge=10, le=10000)
    optimization_grid: dict[str, Any] | None = None
    label: str | None = Field(default=None, max_length=64)

    @field_validator("source_method")
    @classmethod
    def _normalize_source_method(cls, v: str | None) -> str | None:
        if v is None:
            return None
        bare = safe_method_basename(v.strip())
        if bare is None:
            raise ValueError("source_method doit être un nom .json sûr (basename seul)")
        return bare


class OptimizeBatchBody(BaseModel):
    jobs: list[OptimizeBatchJobIn] = Field(..., min_length=1)

    @field_validator("jobs")
    @classmethod
    def _max_jobs(cls, v: list[OptimizeBatchJobIn]) -> list[OptimizeBatchJobIn]:
        if len(v) > _MAX_OPTIM_BATCH_JOBS:
            raise ValueError(f"maximum {_MAX_OPTIM_BATCH_JOBS} jobs par file")
        return v


@router.get("/state")
async def get_control_state():
    await paper_live_service.reconcile_paper_live_task_with_state()
    state = load_state()
    state["model_registry"] = load_registry()
    # État effectif (fichier méthode fusionné) pour le dashboard : un seul lieu d’édition (Méthodes) + miroir caché.
    state["effective_run"] = effective_run_state(state)
    # Aligné sur /paper-live/status : évite un UI « Transition » si le client ne lit que GET /state.
    snap = paper_live_service.paper_live_status()
    state["task_alive"] = snap["task_alive"]
    return state


@router.get("/workspace")
async def get_workspace():
    """Dossiers datasets / méthodes + fichiers ; sélection courante dans l’état."""
    snap = workspace_snapshot()
    st = load_state()
    snap["selected_dataset_file"] = st.get("dashboard_dataset_file")
    snap["selected_method_file"] = st.get("dashboard_method_file")
    return snap


@router.post("/workspace/fetch-dataset")
async def workspace_fetch_dataset(
    symbol: str = Query(..., min_length=1, max_length=80),
    timeframe: str = Query(..., min_length=1, max_length=32),
    calendar_days: int = Query(..., ge=1, le=730),
    filename: str | None = Query(
        default=None,
        max_length=200,
        description="Nom du fichier CSV dans data/dashboard_datasets/ (optionnel, sinon auto).",
    ),
    set_active_dataset: bool = Query(
        True,
        description="Si vrai : sélectionne ce CSV comme source workspace et mode fichier.",
    ),
    sync_analysis_window: bool = Query(
        True,
        description="Si vrai : aligne backtest_limit / optimize_limit sur le nombre de bougies récupéré.",
    ),
    target_candles: int | None = Query(
        default=None,
        ge=300,
        le=100_000,
        description=(
            "Optionnel : nombre exact de bougies à récupérer (multi-requêtes CCXT jusqu’à concurrence). "
            "Sinon le nombre est dérivé de calendar_days (plafonné comme en period_utils)."
        ),
    ),
    merge_existing: bool = Query(
        False,
        description="Si vrai : fusionne avec le CSV existant du même nom (dédoublonnage par timestamp).",
    ),
):
    """Télécharge l’historique depuis l’exchange et enregistre un CSV dans ``data/dashboard_datasets/``."""
    sym = _normalize_exchange_symbol(symbol)
    if target_candles is not None:
        limit = int(target_candles)
    else:
        limit = bars_for_calendar_days(calendar_days, timeframe)
    limit = max(1, min(limit, 100_000))
    fetcher = CCXTFetcher(settings.exchange_id)
    try:
        rows, fetch_meta = await fetcher.fetch_ohlcv_with_meta(sym, timeframe, limit=limit)
    finally:
        await fetcher.close()

    fetched_at = datetime.now(tz=UTC).isoformat()
    if not rows:
        payload = {
            "ok": False,
            "source": "exchange_ccxt_live",
            "exchange_id": settings.exchange_id,
            "fetched_at_utc": fetched_at,
            "symbol_requested": symbol,
            "symbol_normalized": sym,
            "timeframe": timeframe,
            "calendar_days": calendar_days,
            "limit_requested": limit,
            "bars": 0,
            "detail": "Aucune bougie renvoyée par l’exchange.",
        }
        patch_state({"last_ohlcv_fetch": payload})
        return payload

    df_save = pd.DataFrame(
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
    raw_fn = (filename or "").strip()
    auto = suggest_workspace_dataset_basename(sym, timeframe, calendar_days)
    filename_note_fr: str | None = None
    if raw_fn:
        safe = safe_dataset_basename(raw_fn)
        if safe:
            basename = safe
        else:
            basename = auto
            filename_note_fr = (
                f"Le nom demandé « {raw_fn} » n'est pas un nom CSV valide ; "
                f"fichier enregistré sous « {basename} »."
            )
    else:
        basename = auto
    meta_save = save_workspace_dataset_df(df_save, basename, merge_with_existing=merge_existing)
    if not meta_save.get("ok"):
        raise HTTPException(status_code=400, detail=meta_save)

    first_ts = rows[0].ts_open
    last_ts = rows[-1].ts_open
    bars = len(rows)
    last_fetch: dict[str, Any] = {
        "ok": True,
        "source": "workspace_dataset_csv",
        "data_storage_fr": (
            "Fichier CSV enregistré sous data/dashboard_datasets/ ; sélectionnable dans l’onglet Dataset."
        ),
        "exchange_id": settings.exchange_id,
        "fetched_at_utc": fetched_at,
        "symbol_requested": symbol,
        "symbol_normalized": sym,
        "timeframe": timeframe,
        "calendar_days": calendar_days,
        "limit_requested": limit,
        "bars": bars,
        "ohlcv_multi_fetch": fetch_meta,
        "merge_existing_csv": merge_existing,
        "first_candle_open_utc": first_ts.isoformat(),
        "last_candle_open_utc": last_ts.isoformat(),
        "workspace_csv_basename": meta_save.get("basename"),
        "workspace_csv_path": meta_save.get("path"),
    }
    if filename_note_fr:
        last_fetch["filename_invalid_note_fr"] = filename_note_fr
    patch: dict[str, Any] = {"last_ohlcv_fetch": last_fetch}
    if set_active_dataset:
        patch["dashboard_dataset_file"] = str(meta_save.get("basename"))
        patch["backtest_ohlcv_source"] = "file"
    if sync_analysis_window:
        patch["analysis_period_preset"] = "custom"
        patch["backtest_limit"] = max(300, min(20000, bars))
        patch["optimize_limit"] = max(500, min(20000, bars))
    patch_state(patch)
    return {"ok": True, **last_fetch, "saved": meta_save}


@router.get("/workspace/method-file")
async def get_workspace_method_file(name: str = Query(..., min_length=1, max_length=240)):
    """Lit un JSON méthode du dossier workspace (éditeur dashboard)."""
    path = method_json_path(name)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Fichier méthode introuvable ou nom invalide.")
    with path.open("r", encoding="utf-8") as f:
        content: Any = json.load(f)
    if not isinstance(content, dict):
        raise HTTPException(status_code=400, detail="Le JSON racine doit être un objet.")
    return {"name": path.name, "path": str(path.resolve()), "content": content}


@router.put("/workspace/method-file")
async def put_workspace_method_file(
    name: str = Query(..., min_length=1, max_length=240),
    body: dict[str, Any] = Body(...),
):
    """Enregistre un JSON méthode (clés hors overlay ignorées au run mais conservées ici)."""
    path = method_json_path(name)
    if path is None:
        raise HTTPException(status_code=400, detail="Nom de fichier invalide.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(body, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return {"ok": True, "name": path.name, "path": str(path.resolve())}


@router.post("/paper-live/start")
async def paper_live_start():
    """Démarre la boucle paper live (poll CCXT + analyse ; requiert système activé)."""
    return await paper_live_service.start_paper_live()


@router.post("/paper-live/stop")
async def paper_live_stop():
    """Arrête la boucle paper live."""
    return await paper_live_service.stop_paper_live()


@router.get("/paper-live/status")
async def paper_live_status_endpoint():
    """Statut : tâche asyncio, dernier tick, symbole/timeframe, trace méthode."""
    await paper_live_service.reconcile_paper_live_task_with_state()
    return paper_live_service.paper_live_status()


@router.get("/options")
async def get_options():
    state = load_state()
    return {
        "symbols": state.get("symbols", []),
        "timeframes": state.get("timeframes", []),
        "objectives": [
            "net_pnl_quote",
            "penalized_pnl_quote",
            "net_r",
            "profit_factor",
            "expectancy_r",
            "sharpe_like",
            "composite",
        ],
        "optimization_strategies": sorted(_ALLOWED_OPTIMIZATION_STRATEGIES),
        "period_presets": sorted(_ALLOWED_PERIOD_PRESETS),
        "ohlcv_sources": sorted(_OHLCV_SOURCES),
        "ohlcv_data_dir": str(ohlcv_dir().resolve()),
    }


@router.get("/analysis-window")
async def get_analysis_window(
    timeframe: str,
    analysis_period_preset: str | None = Query(
        default=None,
        description="Optionnel : valeur formulaire (sinon état persisté).",
    ),
    period_max_bars: int | None = Query(default=None, ge=300, le=20000),
    backtest_limit: int | None = Query(default=None, ge=300, le=20000),
    optimize_limit: int | None = Query(default=None, ge=500, le=20000),
):
    """Prévisualise le nombre de bougies chargées (preset + plafond + TF).

    Si des query params sont fournis, ils surchargent l’état disque pour refléter le formulaire dashboard.
    """
    state: dict[str, Any] = dict(load_state())
    if analysis_period_preset is not None:
        p = analysis_period_preset.strip().lower()
        if p in _ALLOWED_PERIOD_PRESETS:
            state["analysis_period_preset"] = p
    if period_max_bars is not None:
        state["period_max_bars"] = period_max_bars
    if backtest_limit is not None:
        state["backtest_limit"] = backtest_limit
    if optimize_limit is not None:
        state["optimize_limit"] = optimize_limit
    return _analysis_window_preview(state, timeframe)


@router.put("/state")
async def update_control_state(payload: ControlStatePatch):
    raw = payload.model_dump(exclude_unset=True)
    patch: dict[str, Any] = {}
    for k, v in raw.items():
        if v is not None:
            patch[k] = v
        elif k in ("replay_trail_after_r", "replay_trail_atr_mult"):
            # Permet de désactiver explicitement le trailing (le filtre ``v is not None`` ignorait null avant).
            patch[k] = None
    return patch_state(patch)


@router.get("/exchange-market-info")
async def exchange_market_info(symbol: str):
    """Aperçu CCXT : taker spot vs swap + funding 8h sur le perp (ex. Bitget)."""
    return await fetch_market_info_bundle(settings.exchange_id, symbol)


@router.get("/ohlcv-data-status")
async def get_ohlcv_data_status(symbol: str, timeframe: str):
    """Fichier local + DB : existence, dernière bougie, âge. Ne plante pas si Postgres local est coupé."""
    from app.db.session import async_session_factory

    state = load_state()
    db_payload: dict[str, Any]
    try:
        async with async_session_factory() as session:
            db_payload = await db_dataset_status(session, symbol, timeframe)
    except Exception as e:  # noqa: BLE001 — connexion locale asyncpg / réseau
        db_payload = {
            "ok": False,
            "error": str(e),
            "has_symbol": None,
            "has_timeframe": None,
            "bars": 0,
            "last_candle_open_utc": None,
            "age_seconds_since_last_candle": None,
            "hint_fr": (
                "PostgreSQL local : service Windows « postgresql-x64-… » démarré ? "
                "DATABASE_URL dans .env = postgresql+asyncpg://USER:PASS@127.0.0.1:5432/NOM_BASE "
                "(pas besoin de Docker). Crée la base puis alembic upgrade."
            ),
        }
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "configured_backtest_source": _normalize_ohlcv_source(state),
        "ohlcv_data_dir": str(ohlcv_dir().resolve()),
        "exchange_id": settings.exchange_id,
        "file": file_dataset_status(symbol, timeframe),
        "database": db_payload,
        "notes_fr": {
            "live": "Chaque backtest re-télécharge depuis l'exchange (CCXT).",
            "file": "Lit le CSV sous ohlcv_data_dir (rempli via « Enregistrer sur disque » au téléchargement).",
            "database": "Lit la table candles (remplie par ingestion ou case PostgreSQL au téléchargement).",
        },
    }


async def _fetch_df(fetcher: CCXTFetcher, symbol: str, tf: str, limit: int) -> pd.DataFrame:
    rows = await fetcher.fetch_ohlcv(symbol, tf, limit=limit)
    if not rows:
        return pd.DataFrame()
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


@router.post("/actions/fetch-ohlcv")
async def action_fetch_ohlcv(
    session: SessionDep,
    symbol: str,
    timeframe: str,
    persist: bool = Query(False, description="Upsert candles into PostgreSQL."),
    save_to_file: bool = Query(False, description="Write fetched OHLCV to CSV under ohlcv_data_dir."),
    limit: int | None = Query(
        None,
        ge=100,
        le=20000,
        description="Bars to request; omit to use the same window as backtest (état sauvegardé).",
    ),
):
    """Téléchargement explicite avec métadonnées (source = exchange live; option DB)."""
    state = load_state()
    eff_limit = int(limit) if limit is not None else _analysis_fetch_limit(state, timeframe, "backtest")

    fetcher = CCXTFetcher(settings.exchange_id)
    try:
        rows = await fetcher.fetch_ohlcv(symbol, timeframe, limit=eff_limit)
    finally:
        await fetcher.close()

    fetched_at = datetime.now(tz=UTC).isoformat()
    base_payload: dict[str, Any] = {
        "source": "exchange_ccxt_live",
        "data_storage_fr": (
            "Données demandées en direct à l'exchange (CCXT). Les listes Symbol / Timeframe du dashboard "
            "choisissent le marché, pas un chemin de fichier local."
        ),
        "postgresql_hint_fr": (
            "PostgreSQL (table candles) : en mode « base » sur le dashboard, le téléchargement upsert "
            "automatiquement (ou POST /api/v1/ingestion/run). Les modes « fichier » / « base » lisent ces données ; "
            "« live » ignore fichier/DB."
        ),
        "exchange_id": settings.exchange_id,
        "fetched_at_utc": fetched_at,
        "symbol": symbol,
        "timeframe": timeframe,
        "limit_requested": eff_limit,
    }

    if not rows:
        payload = {
            **base_payload,
            "ok": False,
            "bars": 0,
            "persisted_to_postgres": False,
            "detail": "Aucune bougie renvoyée par l'exchange (symbole ou timeframe invalide ?).",
        }
        patch_state({"last_ohlcv_fetch": payload})
        return payload

    first_ts = rows[0].ts_open
    last_ts = rows[-1].ts_open
    payload: dict[str, Any] = {
        **base_payload,
        "ok": True,
        "bars": len(rows),
        "first_candle_open_utc": first_ts.isoformat(),
        "last_candle_open_utc": last_ts.isoformat(),
        "persisted_to_postgres": False,
        "saved_to_file": False,
    }

    if persist:
        symbol_id, tf_id = await ensure_symbol_timeframe_ids(
            session, settings.exchange_id, symbol, timeframe
        )
        writer = CandleWriter(session)
        rows_upserted = await writer.upsert_candles(symbol_id, tf_id, rows)
        payload["persisted_to_postgres"] = True
        payload["rows_upserted"] = rows_upserted

    if save_to_file:
        df_save = pd.DataFrame(
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
        payload["saved_to_file"] = True
        payload["saved_file_path"] = save_ohlcv_csv(df_save, symbol, timeframe)

    patch_state({"last_ohlcv_fetch": payload})
    return payload


@router.post("/actions/scan")
async def action_scan():
    state = load_state()
    eff = effective_run_state(state)
    symbols = list(eff.get("symbols") or [])
    timeframes = list(eff.get("timeframes") or [])
    limit = int(eff.get("scan_limit") or state.get("scan_limit") or 500)
    send_telegram = bool(state["telegram_enabled"]) and bool(settings.telegram_bot_token) and bool(settings.telegram_chat_id)
    active_model = get_active_model()
    manual_params = active_model["params"] if active_model else eff["best_engine_params"]
    auto_enabled = bool(eff.get("auto_parameters", True)) and active_model is None

    fetcher = CCXTFetcher(settings.exchange_id)
    results: list[dict[str, Any]] = []
    try:
        for symbol in symbols:
            for tf in timeframes:
                df = await _fetch_df(fetcher, symbol, tf, limit)
                if df.empty:
                    continue
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
                    **_engine_params_filters(eff),
                }
                out = await run_analysis(
                    df,
                    symbol,
                    tf,
                    swing_left=int(smc_params["swing_left"]),
                    swing_right=int(smc_params["swing_right"]),
                    send_telegram=send_telegram,
                    render_chart_img=False,
                    engine_params=engine_params,
                )
                results.append(
                    {
                        "symbol": symbol,
                        "timeframe": tf,
                        "trend": out.context.trend.value,
                        "setups_count": len(out.setups),
                        "top_setup": (
                            {
                                "type": out.setups[0].setup_type,
                                "side": out.setups[0].side.value,
                                "entry": out.setups[0].entry,
                                "sl": out.setups[0].stop_loss,
                                "tp1": out.setups[0].take_profits[0] if out.setups[0].take_profits else None,
                                "rr": out.setups[0].risk_reward,
                                "confidence": out.setups[0].confidence,
                            }
                            if out.setups
                            else None
                        ),
                    }
                )
    finally:
        await fetcher.close()

    trace = method_trace_for_runs(state)
    patch_state({"last_scan": {"count": len(results), "results": results[:50], **trace}})
    return {"count": len(results), "results": results, **trace}


@router.post("/actions/backtest")
async def action_backtest(symbol: str, timeframe: str, session: SessionDep):
    state = load_state()
    eff = effective_run_state(state)
    limit = _effective_ohlcv_load_limit(state, timeframe, "backtest")
    active_model = get_active_model()
    manual_params = active_model["params"] if active_model else eff["best_engine_params"]

    df, ohlcv_meta = await _load_backtest_df(session, state, symbol, timeframe, limit)
    if df.empty:
        if ohlcv_meta.get("ohlcv_source_used") == "workspace_csv":
            detail = (
                "Dataset workspace vide ou introuvable. Vérifie le CSV dans data/dashboard_datasets/ "
                f"(détail: {ohlcv_meta.get('error', ohlcv_meta.get('path', ''))})."
            )
        else:
            src = _normalize_ohlcv_source(state)
            if src == "file":
                detail = (
                    f"Fichier OHLCV absent ou vide (attendu: {ohlcv_csv_path(symbol, timeframe)}). "
                    "Télécharge avec « Enregistrer sur disque » ou repasse en mode live."
                )
            elif src == "database":
                detail = (
                    "Aucune bougie en base pour cette paire/timeframe sur l'exchange configuré. "
                    "Lance l'ingestion ou un téléchargement avec option PostgreSQL."
                )
            else:
                detail = "No market data"
        raise HTTPException(status_code=404, detail=detail)
    tf_eff, tf_align = _effective_timeframe_after_workspace_load(timeframe, ohlcv_meta)
    sym_eff, sym_align = _effective_symbol_after_workspace_load(symbol, ohlcv_meta)
    smc_params = resolve_smc_parameters(
        timeframe=tf_eff,
        ohlcv_df=df,
        auto_enabled=bool(eff.get("auto_parameters", True)) and active_model is None,
        manual_params=manual_params,
    )
    engine_params = {
        "rr_min": smc_params["rr_min"],
        "fvg_proximity_pct": smc_params["fvg_proximity_pct"],
        "ob_proximity_pct": smc_params["ob_proximity_pct"],
        "max_setups": smc_params["max_setups"],
        **_engine_params_filters(eff),
    }

    costs = await resolve_trade_cost_rates(eff, sym_eff)
    bt_cfg = build_replay_bt_config(eff, costs)
    report = replay_engine_from_bt_cfg(bt_cfg).run_walkforward(
        df,
        symbol=sym_eff,
        timeframe=tf_eff,
        swing_left=int(smc_params["swing_left"]),
        swing_right=int(smc_params["swing_right"]),
        engine_params=engine_params,
    )
    trace = method_trace_for_runs(state)
    audit_lim = _response_bars_load_limit(ohlcv_meta, limit)
    period_audit = _period_run_audit(
        state,
        timeframe=tf_eff,
        which="backtest",
        limit=audit_lim,
        bars_loaded=len(df),
        df=df,
        ohlcv_meta=ohlcv_meta,
    )
    out = {
        "symbol": sym_eff,
        "timeframe": tf_eff,
        **tf_align,
        **sym_align,
        "trade_costs": costs,
        "ohlcv_load": ohlcv_meta,
        **trace,
        "workspace": {
            "dataset_file": state.get("dashboard_dataset_file"),
        },
        "analysis_period_preset": state.get("analysis_period_preset") or DEFAULT_ANALYSIS_PERIOD_PRESET,
        "period_max_bars": state.get("period_max_bars"),
        "bars_load_limit": audit_lim,
        "bars": len(df),
        **period_audit,
        "total_trades": report.total_trades,
        "wins": report.wins,
        "losses": report.losses,
        "win_rate": report.win_rate,
        "profit_factor": report.profit_factor,
        "expectancy_r": report.expectancy_r,
        "net_r": report.net_r,
        "max_drawdown_r": report.max_drawdown_r,
        "gross_pnl_quote": report.gross_pnl_quote,
        "net_pnl_quote": report.net_pnl_quote,
        "realized_gains_quote": round(report.realized_gains_quote, 4),
        "realized_losses_quote": round(report.realized_losses_quote, 4),
        "total_fees_quote": report.total_fees_quote,
        "total_funding_quote": report.total_funding_quote,
        "avg_trade_duration_bars": report.avg_trade_duration_bars,
        "avg_time_in_negative_pct": report.avg_time_in_negative_pct,
        "max_drawdown_quote": report.max_drawdown_quote,
        "trade_report": [
            {
                "setup_type": t.setup_type,
                "side": t.side.value,
                "opened_at": t.opened_at,
                "closed_at": t.closed_at,
                "bars_held": t.bars_held,
                "entry": t.entry,
                "exit_price": t.close_price,
                "quantity": t.quantity,
                "gross_pnl_quote": t.gross_pnl_quote,
                "net_pnl_quote": t.net_pnl_quote,
                "fees_quote": t.fees_open_quote + t.fees_close_quote,
                "funding_quote": t.funding_quote,
                "pnl_r": t.pnl_r,
                "time_in_negative_pct": t.time_in_negative_pct,
                "max_trade_drawdown_quote": t.max_drawdown_quote,
                "outcome": t.outcome,
            }
            for t in report.trades[:100]
        ],
    }
    patch_state({"last_backtest": out})
    if active_model:
        update_model(
            active_model["id"],
            {"stats": {"last_backtest": out}},
        )
    return out


@router.post("/actions/optimize")
async def action_optimize(symbol: str, timeframe: str, session: SessionDep):
    state = load_state()
    eff = effective_run_state(state)
    limit = _effective_ohlcv_load_limit(state, timeframe, "optimize")
    top_n = int(state.get("top_optimization", 5))

    df, ohlcv_meta = await _load_backtest_df(session, state, symbol, timeframe, limit)
    if df.empty:
        if ohlcv_meta.get("ohlcv_source_used") == "workspace_csv":
            detail = (
                "Dataset workspace vide ou introuvable. Vérifie data/dashboard_datasets/ "
                f"({ohlcv_meta.get('error', ohlcv_meta.get('path', ''))})."
            )
        else:
            src = _normalize_ohlcv_source(state)
            if src == "file":
                detail = (
                    f"Fichier OHLCV absent ou vide (attendu: {ohlcv_csv_path(symbol, timeframe)}). "
                    "Télécharge avec « Enregistrer sur disque » ou repasse en mode live."
                )
            elif src == "database":
                detail = (
                    "Aucune bougie en base pour cette paire/timeframe. "
                    "Lance l'ingestion ou un téléchargement avec option PostgreSQL."
                )
            else:
                detail = "No market data"
        raise HTTPException(status_code=404, detail=detail)
    tf_eff, tf_align = _effective_timeframe_after_workspace_load(timeframe, ohlcv_meta)
    sym_eff, sym_align = _effective_symbol_after_workspace_load(symbol, ohlcv_meta)
    smc_params = resolve_smc_parameters(
        timeframe=tf_eff,
        ohlcv_df=df,
        auto_enabled=bool(eff.get("auto_parameters", True)),
        manual_params=eff.get("best_engine_params", dict(DEFAULT_BEST_ENGINE_PARAMS)),
    )

    grid = eff.get("optimization_grid", {})
    rr_vals = [float(x) for x in grid.get("rr_min_values", DEFAULT_OPTIMIZATION_GRID["rr_min_values"])]
    fvg_vals = [float(x) for x in grid.get("fvg_proximity_values", DEFAULT_OPTIMIZATION_GRID["fvg_proximity_values"])]
    ob_vals = [float(x) for x in grid.get("ob_proximity_values", DEFAULT_OPTIMIZATION_GRID["ob_proximity_values"])]
    sl_vals = [int(x) for x in grid.get("swing_left_values", DEFAULT_OPTIMIZATION_GRID["swing_left_values"])]
    sr_vals = [int(x) for x in grid.get("swing_right_values", DEFAULT_OPTIMIZATION_GRID["swing_right_values"])]
    ms_vals = [int(x) for x in grid.get("max_setups_values", DEFAULT_OPTIMIZATION_GRID["max_setups_values"])]
    costs = await resolve_trade_cost_rates(eff, sym_eff)
    opt_strat = str(eff.get("optimization_strategy", "exhaustive")).strip().lower()
    if opt_strat not in _ALLOWED_OPTIMIZATION_STRATEGIES:
        opt_strat = "exhaustive"
    raw_mt = eff.get("optimization_max_trials")
    opt_max_trials = int(raw_mt) if raw_mt is not None else None
    ranked = optimize_setup_parameters(
        df,
        symbol=sym_eff,
        timeframe=tf_eff,
        objective=str(eff.get("optimization_objective", "net_pnl_quote")),
        backtest_config=build_replay_bt_config(eff, costs),
        rr_min_values=rr_vals,
        fvg_proximity_values=fvg_vals,
        ob_proximity_values=ob_vals,
        swing_left_values=sl_vals,
        swing_right_values=sr_vals,
        max_setups_values=ms_vals,
        strategy=opt_strat,
        max_trials=opt_max_trials,
    )
    optimization_diagnostics = _optimization_run_diagnostics(
        ranked,
        rr_vals=rr_vals,
        fvg_vals=fvg_vals,
        ob_vals=ob_vals,
        sl_vals=sl_vals,
        sr_vals=sr_vals,
        ms_vals=ms_vals,
        strategy=opt_strat,
    )
    best = ranked[:top_n]
    obj_key = str(eff.get("optimization_objective", "net_pnl_quote"))
    mf_raw = state.get("dashboard_method_file")
    mf_bare = safe_method_basename(str(mf_raw)) if mf_raw else None
    method_file_update: dict[str, Any]
    if best and mf_bare:
        method_file_update = apply_optimized_params_to_workspace_method(
            mf_bare,
            best[0].params,
            objective=obj_key,
            symbol=sym_eff,
            timeframe=tf_eff,
        )
    elif best and not mf_bare:
        method_file_update = {
            "ok": False,
            "skipped": True,
            "hint_fr": "Sélectionne un fichier méthode JSON (workspace) pour enregistrer le Top-1 sur disque.",
        }
    else:
        method_file_update = {
            "ok": False,
            "skipped": True,
            "hint_fr": "Aucune combinaison évaluée ou classement vide.",
        }
    payload = []
    for i, item in enumerate(best):
        r = item.report
        pf = r.profit_factor
        payload.append(
            {
                "rank": i + 1,
                "params": item.params,
                "trades": r.total_trades,
                "wins": r.wins,
                "losses": r.losses,
                "win_rate": round(r.win_rate, 4),
                "profit_factor": round(pf, 4) if pf != float("inf") else None,
                "expectancy_r": round(r.expectancy_r, 4),
                "net_r": round(r.net_r, 4),
                "net_pnl_quote": round(r.net_pnl_quote, 4),
                "realized_gains_quote": round(r.realized_gains_quote, 4),
                "realized_losses_quote": round(r.realized_losses_quote, 4),
                "total_fees_quote": round(r.total_fees_quote, 4),
                "total_funding_quote": round(r.total_funding_quote, 4),
                "max_drawdown_r": round(r.max_drawdown_r, 4),
                "max_drawdown_quote": round(r.max_drawdown_quote, 4),
            }
        )
    opt_audit_lim = _response_bars_load_limit(ohlcv_meta, limit)
    opt_period_audit = _period_run_audit(
        state,
        timeframe=tf_eff,
        which="optimize",
        limit=opt_audit_lim,
        bars_loaded=len(df),
        df=df,
        ohlcv_meta=ohlcv_meta,
    )
    if best:
        patch_state(
            {
                "best_engine_params": best[0].params,
                "last_optimization": {
                    "symbol": sym_eff,
                    "timeframe": tf_eff,
                    **tf_align,
                    **sym_align,
                    "objective": obj_key,
                    "optimization_strategy": opt_strat,
                    "optimization_max_trials": opt_max_trials,
                    "ranking_explanation_fr": _OPTIMIZATION_RANKING_FR.get(
                        obj_key,
                        _OPTIMIZATION_RANKING_FR["net_pnl_quote"],
                    ),
                    "trade_costs": costs,
                    "analysis_period_preset": state.get("analysis_period_preset")
                    or DEFAULT_ANALYSIS_PERIOD_PRESET,
                    "period_max_bars": state.get("period_max_bars"),
                    "bars_load_limit": opt_audit_lim,
                    "combinations_tested": len(ranked),
                    "optimization_diagnostics": optimization_diagnostics,
                    "top": payload,
                    "auto_reference": smc_params,
                    "method_file_update": method_file_update,
                    "ohlcv_load": ohlcv_meta,
                    **opt_period_audit,
                    **method_trace_for_runs(state),
                },
            }
        )
    return {
        "symbol": sym_eff,
        "timeframe": tf_eff,
        **tf_align,
        **sym_align,
        "ohlcv_load": ohlcv_meta,
        "trade_costs": costs,
        "analysis_period_preset": state.get("analysis_period_preset") or DEFAULT_ANALYSIS_PERIOD_PRESET,
        "period_max_bars": state.get("period_max_bars"),
        "bars_load_limit": opt_audit_lim,
        **opt_period_audit,
        "objective": obj_key,
        "optimization_strategy": opt_strat,
        "optimization_max_trials": opt_max_trials,
        "ranking_explanation_fr": _OPTIMIZATION_RANKING_FR.get(
            obj_key,
            _OPTIMIZATION_RANKING_FR["net_pnl_quote"],
        ),
        "tested": len(ranked),
        "optimization_diagnostics": optimization_diagnostics,
        "top": payload,
        "method_file_update": method_file_update,
    }


@router.post("/actions/optimize-batch")
async def action_optimize_batch(
    symbol: str,
    timeframe: str,
    session: SessionDep,
    body: OptimizeBatchBody,
):
    """Enchaîne plusieurs optimisations sur les **mêmes** bougies ; chaque job écrit un **nouveau** JSON méthode."""
    state = load_state()
    sym_req = _normalize_exchange_symbol(symbol)
    limit = _effective_ohlcv_load_limit(state, timeframe, "optimize")

    df, ohlcv_meta = await _load_backtest_df(session, state, sym_req, timeframe, limit)
    if df.empty:
        if ohlcv_meta.get("ohlcv_source_used") == "workspace_csv":
            detail = (
                "Dataset workspace vide ou introuvable. Vérifie data/dashboard_datasets/ "
                f"({ohlcv_meta.get('error', ohlcv_meta.get('path', ''))})."
            )
        else:
            src = _normalize_ohlcv_source(state)
            if src == "file":
                detail = (
                    f"Fichier OHLCV absent ou vide (attendu: {ohlcv_csv_path(sym_req, timeframe)}). "
                    "Télécharge avec « Enregistrer sur disque » ou repasse en mode live."
                )
            elif src == "database":
                detail = (
                    "Aucune bougie en base pour cette paire/timeframe. "
                    "Lance l'ingestion ou un téléchargement avec option PostgreSQL."
                )
            else:
                detail = "No market data"
        raise HTTPException(status_code=404, detail=detail)

    tf_eff, tf_align = _effective_timeframe_after_workspace_load(timeframe, ohlcv_meta)
    sym_eff, sym_align = _effective_symbol_after_workspace_load(symbol, ohlcv_meta)
    batch_audit_lim = _response_bars_load_limit(ohlcv_meta, limit)
    batch_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S") + "Z"
    results: list[dict[str, Any]] = []

    for i, job in enumerate(body.jobs, start=1):
        src = job.source_method
        if not src:
            mf = state.get("dashboard_method_file")
            src = safe_method_basename(str(mf)) if mf else None
        if not src:
            results.append(
                {
                    "job_index": i,
                    "ok": False,
                    "error": "source_method absent et aucun fichier méthode sélectionné dans l'état.",
                }
            )
            continue
        src_path = method_json_path(src)
        if src_path is None or not src_path.is_file():
            results.append(
                {
                    "job_index": i,
                    "ok": False,
                    "error": f"fichier méthode introuvable: {src}",
                }
            )
            continue

        eff = build_effective_state_for_batch_job(
            state,
            source_method=src,
            optimization_objective=job.optimization_objective,
            optimization_strategy=job.optimization_strategy,
            optimization_max_trials=job.optimization_max_trials,
            optimization_grid=job.optimization_grid,
        )
        grid = eff.get("optimization_grid", {}) or {}
        rr_vals = [float(x) for x in grid.get("rr_min_values", DEFAULT_OPTIMIZATION_GRID["rr_min_values"])]
        fvg_vals = [float(x) for x in grid.get("fvg_proximity_values", DEFAULT_OPTIMIZATION_GRID["fvg_proximity_values"])]
        ob_vals = [float(x) for x in grid.get("ob_proximity_values", DEFAULT_OPTIMIZATION_GRID["ob_proximity_values"])]
        sl_vals = [int(x) for x in grid.get("swing_left_values", DEFAULT_OPTIMIZATION_GRID["swing_left_values"])]
        sr_vals = [int(x) for x in grid.get("swing_right_values", DEFAULT_OPTIMIZATION_GRID["swing_right_values"])]
        ms_vals = [int(x) for x in grid.get("max_setups_values", DEFAULT_OPTIMIZATION_GRID["max_setups_values"])]

        obj_key = str(eff.get("optimization_objective", "net_pnl_quote")).strip().lower()
        opt_strat = str(eff.get("optimization_strategy", "exhaustive")).strip().lower()
        if opt_strat not in _ALLOWED_OPTIMIZATION_STRATEGIES:
            opt_strat = "exhaustive"
        raw_mt = eff.get("optimization_max_trials")
        opt_max_trials = int(raw_mt) if raw_mt is not None else None

        costs = await resolve_trade_cost_rates(eff, sym_eff)
        ranked = optimize_setup_parameters(
            df,
            symbol=sym_eff,
            timeframe=tf_eff,
            objective=obj_key,
            backtest_config=build_replay_bt_config(eff, costs),
            rr_min_values=rr_vals,
            fvg_proximity_values=fvg_vals,
            ob_proximity_values=ob_vals,
            swing_left_values=sl_vals,
            swing_right_values=sr_vals,
            max_setups_values=ms_vals,
            strategy=opt_strat,
            max_trials=opt_max_trials,
        )
        if not ranked:
            results.append(
                {
                    "job_index": i,
                    "ok": False,
                    "source_method": src,
                    "error": "Aucune combinaison évaluée ou classement vide.",
                }
            )
            continue

        best = ranked[0]
        metrics = report_optimization_export_metrics(best.report, objective=obj_key)
        label_slug = slug_for_autoopt_filename(job.label, obj_key, opt_strat, i)
        try:
            out_name = allocate_autoopt_output_filename(src, label_slug)
        except ValueError as exc:
            results.append({"job_index": i, "ok": False, "source_method": src, "error": str(exc)})
            continue

        ranking_fr = _OPTIMIZATION_RANKING_FR.get(
            obj_key,
            _OPTIMIZATION_RANKING_FR["net_pnl_quote"],
        )
        saved = save_autoopt_method_variant(
            src,
            best.params,
            output_filename=out_name,
            batch_id=batch_id,
            job_index=i,
            label_slug=label_slug,
            objective=obj_key,
            strategy=opt_strat,
            max_trials=opt_max_trials,
            symbol=sym_eff,
            timeframe=tf_eff,
            metrics=metrics,
            combinations_tested=len(ranked),
            ranking_explanation_fr=ranking_fr,
        )
        results.append(
            {
                "job_index": i,
                "ok": bool(saved.get("ok")),
                "source_method": src,
                "output_method": saved,
                "label": label_slug,
                "objective": obj_key,
                "optimization_strategy": opt_strat,
                "optimization_max_trials": opt_max_trials,
                "top1_metrics": metrics,
                "best_params": best.params,
                "combinations_tested": len(ranked),
                "error": saved.get("error"),
            }
        )

    batch_ok = bool(results) and all(bool(r.get("ok")) for r in results)
    out_payload = {
        "ok": batch_ok,
        "batch_id": batch_id,
        "symbol": sym_eff,
        "timeframe": tf_eff,
        **tf_align,
        **sym_align,
        "ohlcv_load": ohlcv_meta,
        "bars_load_limit": batch_audit_lim,
        "jobs": results,
    }
    patch_state({"last_optimization_batch": {**out_payload, "finished_utc": datetime.now(tz=UTC).isoformat()}})
    return out_payload


@router.post("/actions/optimize-walk-forward")
async def action_optimize_walk_forward(
    symbol: str,
    timeframe: str,
    session: SessionDep,
    wf_splits: int | None = Query(None, ge=1, le=12, description="Nombre de fenêtres OOS (train croissant)."),
):
    """Optimisation in-sample sur le passé, puis backtest des meilleurs paramètres sur la tranche suivante (OOS)."""
    state = load_state()
    eff = effective_run_state(state)
    limit = _effective_ohlcv_load_limit(state, timeframe, "optimize")
    eff_splits = int(wf_splits if wf_splits is not None else state.get("wf_splits", 3))

    df, ohlcv_meta = await _load_backtest_df(session, state, symbol, timeframe, limit)
    if df.empty:
        raise HTTPException(status_code=404, detail="No market data")

    tf_eff, wf_tf_align = _effective_timeframe_after_workspace_load(timeframe, ohlcv_meta)
    sym_eff, sym_align = _effective_symbol_after_workspace_load(symbol, ohlcv_meta)
    grid = eff.get("optimization_grid", {})
    rr_vals = [float(x) for x in grid.get("rr_min_values", DEFAULT_OPTIMIZATION_GRID["rr_min_values"])]
    fvg_vals = [float(x) for x in grid.get("fvg_proximity_values", DEFAULT_OPTIMIZATION_GRID["fvg_proximity_values"])]
    ob_vals = [float(x) for x in grid.get("ob_proximity_values", DEFAULT_OPTIMIZATION_GRID["ob_proximity_values"])]
    sl_vals = [int(x) for x in grid.get("swing_left_values", DEFAULT_OPTIMIZATION_GRID["swing_left_values"])]
    sr_vals = [int(x) for x in grid.get("swing_right_values", DEFAULT_OPTIMIZATION_GRID["swing_right_values"])]
    ms_vals = [int(x) for x in grid.get("max_setups_values", DEFAULT_OPTIMIZATION_GRID["max_setups_values"])]
    costs = await resolve_trade_cost_rates(eff, sym_eff)
    obj_key = str(eff.get("optimization_objective", "net_pnl_quote"))
    bt_cfg = build_replay_bt_config(eff, costs)
    wf_strat = str(eff.get("optimization_strategy", "exhaustive")).strip().lower()
    if wf_strat not in _ALLOWED_OPTIMIZATION_STRATEGIES:
        wf_strat = "exhaustive"
    raw_wf_mt = eff.get("optimization_max_trials")
    wf_max_trials = int(raw_wf_mt) if raw_wf_mt is not None else None

    result = run_walk_forward_oos(
        df,
        symbol=sym_eff,
        timeframe=tf_eff,
        objective=obj_key,
        backtest_config=bt_cfg,
        rr_min_values=rr_vals,
        fvg_proximity_values=fvg_vals,
        ob_proximity_values=ob_vals,
        swing_left_values=sl_vals,
        swing_right_values=sr_vals,
        max_setups_values=ms_vals,
        n_splits=eff_splits,
        optimization_strategy=wf_strat,
        optimization_max_trials=wf_max_trials,
    )
    wf_audit_lim = _response_bars_load_limit(ohlcv_meta, limit)
    wf_period_audit = _period_run_audit(
        state,
        timeframe=tf_eff,
        which="optimize",
        limit=wf_audit_lim,
        bars_loaded=len(df),
        df=df,
        ohlcv_meta=ohlcv_meta,
    )
    payload = {
        "symbol": sym_eff,
        "timeframe": tf_eff,
        **wf_tf_align,
        **sym_align,
        "objective": obj_key,
        "optimization_strategy": wf_strat,
        "optimization_max_trials": wf_max_trials,
        "wf_splits": eff_splits,
        "ohlcv_load": ohlcv_meta,
        "trade_costs": costs,
        "analysis_period_preset": state.get("analysis_period_preset") or DEFAULT_ANALYSIS_PERIOD_PRESET,
        "period_max_bars": state.get("period_max_bars"),
        "bars_load_limit": wf_audit_lim,
        **wf_period_audit,
        **result,
        **method_trace_for_runs(state),
    }
    patch_state({"last_wf_oos": payload})
    return payload


@router.get("/models")
async def get_models():
    return load_registry()


@router.post("/models")
async def create_model(payload: ModelCreate):
    item = add_model(payload.name, payload.params)
    return {"created": item}


@router.post("/models/from-last-optimization")
async def create_model_from_last_optimization(payload: ModelFromOptimization):
    state = load_state()
    last_opt = state.get("last_optimization") or {}
    top = last_opt.get("top") or []
    idx = payload.rank - 1
    if idx < 0 or idx >= len(top):
        raise HTTPException(status_code=400, detail="Rank not available in last optimization")
    row = top[idx]
    params = row["params"]
    name = payload.name or f"Model rank {payload.rank} ({last_opt.get('symbol', 'N/A')} {last_opt.get('timeframe', '')})"
    item = add_model(name, params, stats={"from_optimization": row})
    return {"created": item}


@router.post("/models/{model_id}/activate")
async def activate_model(model_id: str):
    item = set_active_model(model_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Model not found")
    patch_state({"best_engine_params": item["params"]})
    return {"active_model": item}


@router.post("/models/{model_id}/paper")
async def set_model_paper(model_id: str, enabled: bool):
    item = update_model(model_id, {"paper_enabled": enabled})
    if item is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"model": item}


@router.post("/models/{model_id}/backtest")
async def backtest_model(
    model_id: str,
    symbol: str,
    timeframe: str,
    session: SessionDep,
):
    reg = load_registry()
    model = next((m for m in reg["items"] if m["id"] == model_id), None)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    state = load_state()
    eff = effective_run_state(state)
    limit = _effective_ohlcv_load_limit(state, timeframe, "backtest")
    df, ohlcv_meta = await _load_backtest_df(session, state, symbol, timeframe, limit)
    if df.empty:
        if ohlcv_meta.get("ohlcv_source_used") == "workspace_csv":
            detail = "Dataset workspace vide ou introuvable (voir data/dashboard_datasets/)."
        else:
            src = _normalize_ohlcv_source(state)
            if src == "file":
                detail = (
                    f"Fichier OHLCV absent ou vide (attendu: {ohlcv_csv_path(symbol, timeframe)}). "
                    "Télécharge avec « Enregistrer sur disque » ou repasse en mode live."
                )
            elif src == "database":
                detail = "Aucune bougie en base pour cette paire/timeframe."
            else:
                detail = "No market data"
        raise HTTPException(status_code=404, detail=detail)

    tf_eff, tf_align = _effective_timeframe_after_workspace_load(timeframe, ohlcv_meta)
    sym_eff, sym_align = _effective_symbol_after_workspace_load(symbol, ohlcv_meta)
    p = model["params"]
    engine_params = {
        "rr_min": p.get("rr_min", 2.0),
        "fvg_proximity_pct": p.get("fvg_proximity_pct", 0.003),
        "ob_proximity_pct": p.get("ob_proximity_pct", 0.003),
        "max_setups": p.get("max_setups", 5),
        **_engine_params_filters(eff),
    }
    costs = await resolve_trade_cost_rates(eff, sym_eff)
    bt_cfg = build_replay_bt_config(eff, costs)
    report = replay_engine_from_bt_cfg(bt_cfg).run_walkforward(
        df,
        symbol=sym_eff,
        timeframe=tf_eff,
        swing_left=int(p.get("swing_left", 3)),
        swing_right=int(p.get("swing_right", 3)),
        engine_params=engine_params,
    )
    model_audit_lim = _response_bars_load_limit(ohlcv_meta, limit)
    model_period_audit = _period_run_audit(
        state,
        timeframe=tf_eff,
        which="backtest",
        limit=model_audit_lim,
        bars_loaded=len(df),
        df=df,
        ohlcv_meta=ohlcv_meta,
    )
    stats = {
        "symbol": sym_eff,
        "timeframe": tf_eff,
        **tf_align,
        **sym_align,
        "trade_costs": costs,
        "ohlcv_load": ohlcv_meta,
        "analysis_period_preset": state.get("analysis_period_preset") or DEFAULT_ANALYSIS_PERIOD_PRESET,
        "period_max_bars": state.get("period_max_bars"),
        "bars_load_limit": model_audit_lim,
        "bars": len(df),
        **model_period_audit,
        "total_trades": report.total_trades,
        "wins": report.wins,
        "losses": report.losses,
        "win_rate": report.win_rate,
        "profit_factor": report.profit_factor,
        "expectancy_r": report.expectancy_r,
        "net_r": report.net_r,
        "max_drawdown_r": report.max_drawdown_r,
        "net_pnl_quote": report.net_pnl_quote,
        "max_drawdown_quote": report.max_drawdown_quote,
    }
    update_model(model_id, {"stats": {"last_backtest": stats}})
    return {"model_id": model_id, "stats": stats}
