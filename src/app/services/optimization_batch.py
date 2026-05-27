"""File d’optimisations : plusieurs jobs séquentiels, exports JSON versionnés sur disque."""

from __future__ import annotations

import copy
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.services.dashboard_workspace import (
    effective_run_state_for_workspace_method,
    load_workspace_method_json,
    method_json_path,
    safe_method_basename,
    workspace_methods_dir,
)

_ALLOWED_OPT_STRAT = frozenset({"exhaustive", "random", "coordinate_descent"})
_ENGINE_PARAM_KEYS = (
    "rr_min",
    "fvg_proximity_pct",
    "ob_proximity_pct",
    "max_setups",
    "swing_left",
    "swing_right",
)


def slug_for_autoopt_filename(
    label: str | None,
    objective: str,
    strategy: str,
    job_index: int,
) -> str:
    raw = (label or "").strip()
    if not raw:
        raw = f"{objective}_{strategy}_j{job_index}"
    s = raw.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_") or f"job_{job_index}"
    return s[:48]


def allocate_autoopt_output_filename(source_basename: str, label_slug: str) -> str:
    """Nom fichier unique dans ``data/dashboard_methods/`` (suffixe ``__autoopt__``)."""
    safe = safe_method_basename(source_basename)
    if not safe:
        raise ValueError("invalid source_basename")
    stem = Path(safe).stem
    d = workspace_methods_dir()
    base = f"{stem}__autoopt__{label_slug}"
    name = f"{base}.json"
    if not (d / name).exists():
        return name
    for i in range(2, 10000):
        name = f"{base}__{i:03d}.json"
        if not (d / name).exists():
            return name
    return f"{base}__{datetime.now(tz=UTC).strftime('%H%M%S%f')}.json"


def build_effective_state_for_batch_job(
    base_state: dict[str, Any],
    *,
    source_method: str,
    optimization_objective: str | None,
    optimization_strategy: str | None,
    optimization_max_trials: int | None,
    optimization_grid: dict[str, Any] | None,
) -> dict[str, Any]:
    eff = effective_run_state_for_workspace_method(base_state, source_method)
    eff = dict(eff)
    if optimization_objective is not None:
        eff["optimization_objective"] = str(optimization_objective).strip().lower()
    if optimization_strategy is not None:
        s = str(optimization_strategy).strip().lower()
        if s in _ALLOWED_OPT_STRAT:
            eff["optimization_strategy"] = s
    if optimization_max_trials is not None:
        eff["optimization_max_trials"] = int(optimization_max_trials)
    if optimization_grid and isinstance(optimization_grid, dict):
        cur = dict(eff.get("optimization_grid") or {})
        for k, v in optimization_grid.items():
            if v is not None:
                cur[k] = v
        eff["optimization_grid"] = cur
    return eff


def save_autoopt_method_variant(
    source_basename: str,
    best_params: dict[str, Any],
    *,
    output_filename: str,
    batch_id: str,
    job_index: int,
    label_slug: str,
    objective: str,
    strategy: str,
    max_trials: int | None,
    symbol: str,
    timeframe: str,
    metrics: dict[str, Any],
    combinations_tested: int,
    ranking_explanation_fr: str | None,
) -> dict[str, Any]:
    """Écrit une **nouvelle** copie méthode (ne modifie pas le fichier source)."""
    safe_src = safe_method_basename(source_basename)
    if not safe_src:
        return {"ok": False, "error": "nom de fichier méthode source invalide"}
    src_path = method_json_path(safe_src)
    if src_path is None or not src_path.is_file():
        return {"ok": False, "error": "fichier méthode source introuvable", "basename": safe_src}

    safe_out = safe_method_basename(output_filename)
    if not safe_out:
        return {"ok": False, "error": "nom de fichier de sortie invalide"}

    raw = load_workspace_method_json(safe_src)
    if not raw:
        return {"ok": False, "error": "JSON méthode source vide ou illisible"}
    doc = copy.deepcopy(raw)
    if not isinstance(doc, dict):
        return {"ok": False, "error": "JSON méthode source : racine doit être un objet"}

    cur = doc.get("best_engine_params")
    if not isinstance(cur, dict):
        cur = {}
    for k in _ENGINE_PARAM_KEYS:
        if k in best_params:
            cur[k] = best_params[k]
    doc["best_engine_params"] = cur
    doc["optimization_objective"] = objective
    doc["optimization_strategy"] = strategy
    if max_trials is not None:
        doc["optimization_max_trials"] = max_trials

    now = datetime.now(tz=UTC).isoformat()
    doc["optimization_last_applied_utc"] = now
    doc["optimization_last_run"] = {
        "objective": objective,
        "symbol": symbol,
        "timeframe": timeframe,
        "params": dict(cur),
        "via": "optimization_batch",
        "batch_id": batch_id,
        "job_index": job_index,
        "label": label_slug,
    }
    doc["auto_optimization_batch_export"] = {
        "batch_id": batch_id,
        "job_index": job_index,
        "label": label_slug,
        "source_method_file": safe_src,
        "exported_utc": now,
        "optimization": {
            "objective": objective,
            "strategy": strategy,
            "max_trials": max_trials,
            "symbol": symbol,
            "timeframe": timeframe,
            "ranking_explanation_fr": ranking_explanation_fr,
            "combinations_tested": combinations_tested,
        },
        "top1_backtest_metrics": metrics,
    }

    out_path = workspace_methods_dir() / safe_out
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return {"ok": True, "basename": safe_out, "path": str(out_path.resolve())}
