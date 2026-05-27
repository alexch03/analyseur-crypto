"""Workspace dashboard : CSV OHLCV et profils JSON sur disque, listables et persistants.

Dossiers (créés au besoin, **racine du dépôt** = parent de ``src``, pas le CWD du processus) ::
  - ``data/dashboard_datasets/`` : fichiers ``*.csv`` (colonnes timestamp, open, high, low, close, volume).
  - ``data/dashboard_methods/`` : fichiers ``*.json`` (paramètres / méthode, voir ``METHOD_OVERLAY_KEYS``).

La sélection active est stockée dans ``.runtime_control.json`` (``dashboard_dataset_file``,
``dashboard_method_file``) et n'est pas écrasée par un simple rechargement de page.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

_DATASET_NAME_RE = re.compile(r"^[\w.\-]+\.csv$", re.IGNORECASE)
_METHOD_NAME_RE = re.compile(r"^[\w.\-]+\.json$", re.IGNORECASE)
# Convention ``suggest_workspace_dataset_basename`` : STEM__{tf}__{N}d.csv (STEM = symbole CCXT avec / → _)
_DATASET_FULL_RE = re.compile(
    r"^(?P<stem>[\w.\-]+)__(?P<tf>[0-9]+[mhdw])__(?P<days>\d+)d\.csv$",
    re.IGNORECASE,
)
# Quotes « stables » les plus longues en premier (FDUSD avant USD).
_WS_KNOWN_QUOTES: tuple[str, ...] = (
    "FDUSD",
    "USDT",
    "USDC",
    "BUSD",
    "TUSD",
    "DAI",
    "USDE",
    "EUR",
    "GBP",
    "USD",
    "BTC",
    "ETH",
    "BNB",
    "SOL",
)

_PERIOD_PRESETS = frozenset({"7d", "30d", "90d", "180d", "365d", "custom"})


def workspace_data_root() -> Path:
    """Racine du dépôt (dossier qui contient ``src/`` et ``data/``).

    Ancré sur l'emplacement de ce module pour que liste + écriture des méthodes/CSV
    restent alignées même si uvicorn est lancé avec ``--app-dir src`` ou un autre CWD.
    """
    # .../src/app/services/dashboard_workspace.py -> parents[3] = racine repo
    return Path(__file__).resolve().parents[3]


def workspace_datasets_dir() -> Path:
    p = workspace_data_root() / "data" / "dashboard_datasets"
    p.mkdir(parents=True, exist_ok=True)
    return p


def workspace_methods_dir() -> Path:
    p = workspace_data_root() / "data" / "dashboard_methods"
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_dataset_basename(name: str) -> str | None:
    n = (name or "").strip()
    if not n or not _DATASET_NAME_RE.match(n) or ".." in n or "/" in n or "\\" in n:
        return None
    return n


def safe_method_basename(name: str) -> str | None:
    n = (name or "").strip()
    if not n or not _METHOD_NAME_RE.match(n) or ".." in n or "/" in n or "\\" in n:
        return None
    return n


def parse_workspace_dataset_basename_parts(basename: str | None) -> tuple[str, str] | None:
    """Retourne ``(stem, tf)`` pour un nom ``ETH_USDT__5m__15d.csv``, sinon ``None``."""
    raw = (basename or "").strip()
    if not raw:
        return None
    safe = safe_dataset_basename(raw)
    if not safe:
        return None
    m = _DATASET_FULL_RE.match(safe)
    if not m:
        return None
    return str(m.group("stem")), str(m.group("tf")).strip().lower()


def _stem_to_ccxt_symbol(stem: str) -> str | None:
    """Inverse partiel de ``suggest_workspace_dataset_basename`` : ``ETH_USDT`` → ``ETH/USDT``."""
    stem = stem.strip()
    if not stem:
        return None
    upper = stem.upper()
    for q in sorted(_WS_KNOWN_QUOTES, key=len, reverse=True):
        suf = f"_{q}"
        if upper.endswith(suf):
            base = stem[: -len(suf)]
            if not base:
                return None
            return f"{base}/{q}"
    if "_" not in stem:
        return None
    base, quote = stem.rsplit("_", 1)
    if not base or len(quote) < 2 or len(quote) > 10:
        return None
    return f"{base}/{quote.upper()}"


def infer_symbol_from_workspace_dataset_basename(basename: str | None) -> str | None:
    """Déduit la paire CCXT depuis le préfixe STEM d'un CSV workspace (convention dashboard)."""
    parts = parse_workspace_dataset_basename_parts(basename)
    if not parts:
        return None
    return _stem_to_ccxt_symbol(parts[0])


def infer_timeframe_from_workspace_dataset_basename(basename: str | None) -> str | None:
    """Lit le segment ``__{tf}__`` dans un nom ``…__5m__15d.csv`` (généré par le dashboard).

    Retourne le timeframe en minuscules (ex. ``5m``) ou ``None`` si le nom ne suit pas la convention.
    """
    parts = parse_workspace_dataset_basename_parts(basename)
    return parts[1] if parts else None


def suggest_workspace_dataset_basename(symbol: str, timeframe: str, calendar_days: int) -> str:
    stem = (symbol or "SYM").strip().replace("/", "_").replace(" ", "")
    stem = re.sub(r"[^\w.\-]+", "_", stem).strip("._") or "dataset"
    tf = (timeframe or "1h").strip()
    tf = re.sub(r"[^\w.\-]+", "_", tf).strip("._") or "1h"
    return f"{stem}__{tf}__{int(calendar_days)}d.csv"


def save_workspace_dataset_df(
    df: pd.DataFrame,
    basename: str,
    *,
    merge_with_existing: bool = False,
) -> dict[str, Any]:
    """Écrit un CSV OHLCV dans ``data/dashboard_datasets/`` (basename validé).

    Si ``merge_with_existing`` est vrai et que le fichier existe déjà, fusionne les lignes
    (dédoublonnage par ``timestamp``, conserve la dernière occurrence), trie, puis réécrit.
    """
    safe = safe_dataset_basename(basename)
    if safe is None:
        return {"ok": False, "error": "nom de fichier invalide", "basename": basename}
    if df.empty or "timestamp" not in df.columns:
        return {"ok": False, "error": "dataframe vide ou sans colonne timestamp"}
    path = workspace_datasets_dir() / safe
    merged_from_disk = bool(merge_with_existing and path.is_file())
    out = df.copy()
    if merged_from_disk:
        prev = pd.read_csv(path, parse_dates=["timestamp"])
        out = pd.concat([prev, out], ignore_index=True)
        out = out.drop_duplicates(subset=["timestamp"], keep="last")
    out = out.sort_values("timestamp")
    out.to_csv(path, index=False)
    return {
        "ok": True,
        "basename": safe,
        "path": str(path.resolve()),
        "rows": int(len(out)),
        "merged_with_existing": merged_from_disk,
    }


def method_json_path(basename: str) -> Path | None:
    safe = safe_method_basename(basename)
    if not safe:
        return None
    return workspace_methods_dir() / safe


def list_workspace_datasets() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    d = workspace_datasets_dir()
    for path in sorted(d.glob("*.csv"), key=lambda p: p.name.lower()):
        try:
            sz = path.stat().st_size
        except OSError:
            sz = 0
        out.append({"name": path.name, "bytes": sz, "path": str(path.resolve())})
    return out


def list_workspace_methods() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    d = workspace_methods_dir()
    for path in sorted(d.glob("*.json"), key=lambda p: p.name.lower()):
        try:
            sz = path.stat().st_size
        except OSError:
            sz = 0
        out.append({"name": path.name, "bytes": sz, "path": str(path.resolve())})
    return out


def is_workspace_csv_dataset_selected(state: dict[str, Any]) -> bool:
    """Vrai si un fichier ``*.csv`` du workspace est sélectionné (pas live / DB / CSV auto)."""
    ds = str(state.get("dashboard_dataset_file") or "").strip()
    return bool(ds) and ds not in ("__live__", "__database__", "__auto_csv__", "")


# Garde-fou mémoire : au-delà, on tronque à la fin de la série (cas rare).
WORKSPACE_CSV_SAFETY_MAX_ROWS = 350_000


def prepare_workspace_ohlcv_for_analysis(
    df: pd.DataFrame,
    base_meta: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Trie chronologiquement et utilise **toutes** les lignes du CSV, sauf dépassement du plafond technique."""
    meta = dict(base_meta)
    rows_on_disk = int(meta.get("rows_loaded", len(df)))
    meta["rows_on_disk"] = rows_on_disk
    n = len(df)
    if n == 0:
        meta["analysis_bars_load_limit"] = 0
        meta["workspace_load_policy"] = "empty"
        return df, meta
    if n <= WORKSPACE_CSV_SAFETY_MAX_ROWS:
        out = df.sort_values("timestamp").reset_index(drop=True)
        meta["analysis_bars_load_limit"] = len(out)
        meta["workspace_load_policy"] = "full_file"
        return out, meta
    out = df.sort_values("timestamp").tail(WORKSPACE_CSV_SAFETY_MAX_ROWS).reset_index(drop=True)
    meta["analysis_bars_load_limit"] = len(out)
    meta["workspace_load_policy"] = "truncated_safety_max"
    meta["workspace_safety_max_rows"] = WORKSPACE_CSV_SAFETY_MAX_ROWS
    meta["rows_after_safety_cap"] = len(out)
    return out, meta


def load_workspace_dataset_csv(basename: str, limit: int | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Lit un CSV du dossier datasets (basename validé)."""
    safe = safe_dataset_basename(basename)
    if safe is None:
        return pd.DataFrame(), {"ok": False, "error": "nom de fichier invalide", "basename": basename}
    path = workspace_datasets_dir() / safe
    if not path.is_file():
        return pd.DataFrame(), {"ok": False, "error": "fichier introuvable", "path": str(path.resolve())}
    df = pd.read_csv(path, parse_dates=["timestamp"])
    meta: dict[str, Any] = {
        "ok": True,
        "workspace_dataset_basename": safe,
        "path": str(path.resolve()),
        "rows_loaded": len(df),
    }
    if limit is not None and limit > 0 and len(df) > limit:
        df = df.sort_values("timestamp").tail(int(limit)).reset_index(drop=True)
        meta["rows_after_limit"] = len(df)
    return df, meta


def load_workspace_method_json(basename: str) -> dict[str, Any]:
    safe = safe_method_basename(basename)
    if safe is None:
        return {}
    path = workspace_methods_dir() / safe
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw if isinstance(raw, dict) else {}


def apply_optimized_params_to_workspace_method(
    basename: str,
    best_params: dict[str, Any],
    *,
    objective: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> dict[str, Any]:
    """Fusionne les meilleurs paramètres moteur dans ``data/dashboard_methods/<basename>``."""
    path = method_json_path(basename)
    if path is None:
        return {"ok": False, "error": "nom de fichier méthode invalide", "basename": basename}
    if not path.is_file():
        return {"ok": False, "error": "fichier méthode introuvable", "path": str(path.resolve())}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        return {"ok": False, "error": "JSON méthode : racine doit être un objet"}
    keys = ("rr_min", "fvg_proximity_pct", "ob_proximity_pct", "max_setups", "swing_left", "swing_right")
    cur = raw.get("best_engine_params")
    if not isinstance(cur, dict):
        cur = {}
    for k in keys:
        if k in best_params:
            cur[k] = best_params[k]
    raw["best_engine_params"] = cur
    raw["optimization_last_applied_utc"] = datetime.now(tz=UTC).isoformat()
    raw["optimization_last_run"] = {
        "objective": objective,
        "symbol": symbol,
        "timeframe": timeframe,
        "params": dict(cur),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return {"ok": True, "basename": path.name, "path": str(path.resolve())}


# Clés autorisées dans un JSON « méthode » (racine). Jamais : dashboard_*, enabled, clés registry, etc.
METHOD_OVERLAY_KEYS: frozenset[str] = frozenset(
    {
        "symbols",
        "timeframes",
        "scan_limit",
        "best_engine_params",
        "optimization_grid",
        "optimization_objective",
        "optimization_strategy",
        "optimization_max_trials",
        "training_bars",
        "max_holding_bars",
        "max_setups_per_bar",
        "unit_size",
        "entry_fee_rate",
        "exit_fee_rate",
        "funding_rate_8h",
        "replay_trail_after_r",
        "replay_trail_atr_mult",
        "replay_trail_atr_period",
        "replay_timeout_smart_extend",
        "replay_timeout_grace_bars",
        "replay_timeout_max_extensions",
        "replay_timeout_bb_period",
        "replay_timeout_sma_fast",
        "replay_timeout_sma_slow",
        "paper_live_send_telegram",
        "require_ifvg_confluence",
        "ifvg_confluence_pct",
        "require_rsi_divergence",
        "auto_parameters",
        "auto_fee_from_exchange",
        "fee_market_type",
        "auto_funding_from_exchange",
        "telegram_enabled",
        "analysis_period_preset",
        "period_max_bars",
        "backtest_limit",
        "optimize_limit",
        "top_optimization",
        "wf_splits",
        "paper_execution_backend",
        "paper_ohlcv_exchange_id",
        "chart_focus_last_bars",
        "chart_compact_overlays",
    }
)


def merge_method_overlay(base_state: dict[str, Any], method: dict[str, Any]) -> dict[str, Any]:
    """Copie shallow de ``base_state`` avec champs autorisés issus de ``method`` (fichier JSON)."""
    out = dict(base_state)
    for k in METHOD_OVERLAY_KEYS:
        if k not in method:
            continue
        v = method[k]
        if k == "best_engine_params" and isinstance(v, dict):
            cur = dict(out.get("best_engine_params") or {})
            cur.update(v)
            out["best_engine_params"] = cur
        elif k == "optimization_grid" and isinstance(v, dict):
            cur = dict(out.get("optimization_grid") or {})
            cur.update(v)
            out["optimization_grid"] = cur
        elif k in ("symbols", "timeframes") and isinstance(v, list) and len(v) > 0:
            cleaned = [str(x).strip() for x in v if str(x).strip()]
            if cleaned:
                out[k] = cleaned
        elif k == "scan_limit" and v is not None:
            try:
                out["scan_limit"] = max(100, min(5000, int(v)))
            except (TypeError, ValueError):
                pass
        elif k == "analysis_period_preset" and isinstance(v, str):
            p = v.strip().lower()
            if p in _PERIOD_PRESETS:
                out["analysis_period_preset"] = p
        elif k == "period_max_bars" and v is not None:
            try:
                out["period_max_bars"] = max(300, min(20000, int(v)))
            except (TypeError, ValueError):
                pass
        elif k == "backtest_limit" and v is not None:
            try:
                out["backtest_limit"] = max(300, min(20000, int(v)))
            except (TypeError, ValueError):
                pass
        elif k == "optimize_limit" and v is not None:
            try:
                out["optimize_limit"] = max(500, min(20000, int(v)))
            except (TypeError, ValueError):
                pass
        elif k == "top_optimization" and v is not None:
            try:
                out["top_optimization"] = max(1, min(20, int(v)))
            except (TypeError, ValueError):
                pass
        elif k == "wf_splits" and v is not None:
            try:
                out["wf_splits"] = max(1, min(12, int(v)))
            except (TypeError, ValueError):
                pass
        elif k == "paper_execution_backend":
            if isinstance(v, str) and v.strip():
                s = v.strip().lower()
                out["paper_execution_backend"] = (
                    s if s in ("sim_replay", "bitget_futures_sim") else "sim_replay"
                )
        elif k == "paper_ohlcv_exchange_id":
            s = str(v).strip().lower() if v is not None else ""
            out["paper_ohlcv_exchange_id"] = (s[:32] if s else None)
        elif k == "chart_focus_last_bars":
            if v is None:
                out["chart_focus_last_bars"] = None
            else:
                try:
                    vi = int(v)
                    out["chart_focus_last_bars"] = max(15, min(8000, vi))
                except (TypeError, ValueError):
                    pass
        elif k == "chart_compact_overlays" and v is not None:
            out["chart_compact_overlays"] = bool(v)
        else:
            out[k] = v
    return out


def method_trace_for_runs(state: dict[str, Any]) -> dict[str, Any]:
    """Métadonnées pour auditer un run : fichier méthode + hash du contenu sur disque."""
    mf = state.get("dashboard_method_file")
    if not mf:
        return {"method_file": None, "method_sha256": None, "method_overlay_applied": False}
    bare = safe_method_basename(str(mf))
    if not bare:
        return {"method_file": str(mf), "method_sha256": None, "method_overlay_applied": False}
    path = workspace_methods_dir() / bare
    if not path.is_file():
        return {"method_file": bare, "method_sha256": None, "method_overlay_applied": False}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"method_file": bare, "method_sha256": digest, "method_overlay_applied": True}


def effective_run_state(state: dict[str, Any]) -> dict[str, Any]:
    """État utilisé pour un backtest / optim / scan : état disque + méthode JSON sélectionnée (si fichier défini)."""
    mf = state.get("dashboard_method_file")
    if not mf:
        return dict(state)
    bare = safe_method_basename(str(mf))
    if not bare:
        return dict(state)
    method = load_workspace_method_json(bare)
    if not method:
        return dict(state)
    return merge_method_overlay(dict(state), method)


def effective_run_state_for_workspace_method(
    state: dict[str, Any],
    method_basename: str | None,
) -> dict[str, Any]:
    """Comme ``effective_run_state`` mais en forçant un fichier méthode du workspace (file d’optimisation)."""
    if not method_basename:
        return dict(state)
    bare = safe_method_basename(str(method_basename).strip())
    if not bare:
        return dict(state)
    method = load_workspace_method_json(bare)
    if not method:
        return dict(state)
    return merge_method_overlay(dict(state), method)


def workspace_snapshot() -> dict[str, Any]:
    """Pour GET API : chemins + listes de fichiers."""
    dd = workspace_datasets_dir()
    md = workspace_methods_dir()
    return {
        "datasets_dir": str(dd.resolve()),
        "methods_dir": str(md.resolve()),
        "dataset_files": list_workspace_datasets(),
        "method_files": list_workspace_methods(),
        "dataset_sentinels": [
            {"value": "__live__", "label": "Live — exchange (CCXT) à chaque run"},
            {"value": "__database__", "label": "PostgreSQL — table candles"},
            {"value": "__auto_csv__", "label": "CSV auto — data/ohlcv (SYMBOLE__timeframe.csv)"},
        ],
    }
