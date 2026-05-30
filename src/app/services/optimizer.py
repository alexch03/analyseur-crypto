"""Parameter optimization utilities for deterministic trading rules."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import product
from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd

from app.paper.engine_replay import BacktestReport, replay_engine_from_bt_cfg


@dataclass(slots=True, frozen=True)
class OptimizationResult:
    params: dict[str, Any]
    report: BacktestReport


def optimize_setup_parameters(
    ohlcv_df: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    objective: str = "net_pnl_quote",
    backtest_config: dict[str, Any] | None = None,
    max_setups_values: list[int] | None = None,
    rr_min_values: list[float] | None = None,
    fvg_proximity_values: list[float] | None = None,
    ob_proximity_values: list[float] | None = None,
    swing_left_values: list[int] | None = None,
    swing_right_values: list[int] | None = None,
    strategy: str = "exhaustive",
    max_trials: int | None = None,
    random_seed: int | None = 42,
) -> list[OptimizationResult]:
    """Optimise les paramètres moteur sur une grille discrète.

    ``strategy`` :

    - ``exhaustive`` : produit cartésien complet (comportement historique).
    - ``random`` : échantillon aléatoire sans remplacement parmi les combinaisons
      (plafonné par ``max_trials``, défaut 150 si absent).
    - ``coordinate_descent`` : recherche locale axe par axe depuis le centre de la grille ;
      beaucoup moins de backtests qu’une grille pleine, au prix d’un optimum local.

    ``objective`` :

    - ``composite`` : score borné (tanh) mélangeant PnL, PF, espérance R, somme R, DD quote
      et volume de trades.
    - ``penalized_pnl_quote`` : **PnL net quote moins pénalités explicites** (DD quote,
      sous-minimum de trades, profit factor inférieur à 1, sur-trading) — même échelle
      que le PnL, interprétable comme « PnL ajusté risque » pour le tri.
    """
    rr_min_values = rr_min_values or [1.8, 2.0, 2.5]
    fvg_proximity_values = fvg_proximity_values or [0.003, 0.005]
    ob_proximity_values = ob_proximity_values or [0.003, 0.005]
    swing_left_values = swing_left_values or [2, 3]
    swing_right_values = swing_right_values or [2, 3]
    max_setups_values = max_setups_values or [3, 5]

    strat = (strategy or "exhaustive").strip().lower()
    if strat not in ("exhaustive", "random", "coordinate_descent"):
        strat = "exhaustive"

    backtest_config = backtest_config or {}
    engine = replay_engine_from_bt_cfg(backtest_config)
    filter_keys = ("require_ifvg_confluence", "ifvg_confluence_pct", "require_rsi_divergence")
    filter_extra = {k: backtest_config[k] for k in filter_keys if k in backtest_config}

    dims: list[list[Any]] = [
        list(rr_min_values),
        list(fvg_proximity_values),
        list(ob_proximity_values),
        list(swing_left_values),
        list(swing_right_values),
        list(max_setups_values),
    ]

    def run_combo(
        rr_min: float,
        fvg_prox: float,
        ob_prox: float,
        s_left: int,
        s_right: int,
        max_setups: int,
    ) -> OptimizationResult:
        params = {
            "rr_min": rr_min,
            "fvg_proximity_pct": fvg_prox,
            "ob_proximity_pct": ob_prox,
            "max_setups": max_setups,
        }
        report = engine.run_walkforward(
            ohlcv_df,
            symbol=symbol,
            timeframe=timeframe,
            swing_left=s_left,
            swing_right=s_right,
            engine_params={**params, **filter_extra},
        )
        return OptimizationResult(
            params={**params, "swing_left": s_left, "swing_right": s_right},
            report=report,
        )

    if strat == "coordinate_descent":
        results = _coordinate_descent_search(
            dims, run_combo, objective=objective
        )
    else:
        combos: Iterable[tuple[Any, ...]] = product(*dims)
        combo_list = list(combos)
        if strat == "random":
            cap = int(max_trials) if max_trials is not None else 80
            cap = max(1, min(cap, len(combo_list)))
            if len(combo_list) > cap:
                # Seed fixe => optimisation reproductible run-to-run (cf README "deterministic").
                rng = random.Random(random_seed)
                combo_list = rng.sample(combo_list, cap)
        results = [run_combo(*c) for c in combo_list]

    results.sort(
        key=lambda r: _objective_score(r.report, objective),
        reverse=True,
    )
    return results


def neighbor_stability(
    results: list[OptimizationResult],
    top_params: dict[str, Any],
    objective: str,
) -> dict[str, Any]:
    """Measure how stable the top-1 result is compared to its grid neighbors.

    Returns the average and min objective score of results whose params
    differ from *top_params* on exactly one dimension.  A high ratio
    (neighbor_avg / top_score) close to 1.0 means the optimum is on a
    plateau (robust).  A low ratio means isolated spike (likely overfitting).
    """
    top_keys = sorted(top_params.keys())

    def _is_neighbor(p: dict[str, Any]) -> bool:
        diffs = sum(1 for k in top_keys if p.get(k) != top_params.get(k))
        return diffs == 1

    top_score = _objective_score_scalar(
        next((r for r in results if r.params == top_params), results[0]).report,
        objective,
    )
    neighbor_scores = [
        _objective_score_scalar(r.report, objective)
        for r in results
        if _is_neighbor(r.params)
    ]

    if not neighbor_scores:
        return {"neighbor_count": 0, "stability_ratio": None, "interpretation_fr": "Pas de voisins trouvés."}

    avg = sum(neighbor_scores) / len(neighbor_scores)
    worst = min(neighbor_scores)
    ratio = avg / top_score if top_score != 0 else 0.0

    if ratio >= 0.80:
        interp = "Plateau stable — les paramètres voisins donnent des résultats similaires (bon signe)."
    elif ratio >= 0.50:
        interp = "Stabilité moyenne — certains voisins divergent, prudence."
    else:
        interp = "Pic isolé — le résultat top-1 est probablement du sur-ajustement."

    return {
        "neighbor_count": len(neighbor_scores),
        "top_score": round(top_score, 4),
        "neighbor_avg_score": round(avg, 4),
        "neighbor_worst_score": round(worst, 4),
        "stability_ratio": round(ratio, 4),
        "interpretation_fr": interp,
    }


def _objective_score_scalar(report: BacktestReport, objective: str) -> float:
    """Single float for objective comparison (first element of the tuple)."""
    return _objective_score(report, objective)[0]


def _coordinate_descent_search(
    dims: list[list[Any]],
    run_combo: Callable[..., OptimizationResult],
    *,
    objective: str,
) -> list[OptimizationResult]:
    """Hill-climbing : pour chaque dimension, garde la meilleure valeur en fixant les autres."""
    lengths = [max(1, len(d)) for d in dims]
    idx = [min(lengths[i] // 2, lengths[i] - 1) for i in range(6)]
    seen: dict[tuple[Any, ...], OptimizationResult] = {}

    def params_tuple(ix: list[int]) -> tuple[Any, ...]:
        return tuple(dims[i][ix[i]] for i in range(6))

    def get_res(ix: list[int]) -> OptimizationResult:
        key = tuple(ix)
        if key in seen:
            return seen[key]
        t = params_tuple(ix)
        r = run_combo(*t)
        seen[key] = r
        return r

    get_res(idx)
    while True:
        improved = False
        base_rep = get_res(idx).report
        base_key = _objective_score(base_rep, objective)
        for dim in range(6):
            best_i = idx[dim]
            best_key = base_key
            for vi in range(lengths[dim]):
                if vi == idx[dim]:
                    continue
                trial = idx.copy()
                trial[dim] = vi
                rep = get_res(trial).report
                k = _objective_score(rep, objective)
                if k > best_key:
                    best_key = k
                    best_i = vi
            if best_i != idx[dim]:
                idx[dim] = best_i
                improved = True
                base_rep = get_res(idx).report
                base_key = _objective_score(base_rep, objective)
        if not improved:
            break
    return list(seen.values())


# penalized_pnl_quote: fixed penalties, no batch normalization.
# Low-trade penalty is mild: selective strategies on short datasets are fine.
_PEN_DD_WEIGHT = 0.50
_PEN_MIN_TRADES = 3
_PEN_LOW_TRADE_WEIGHT = 40.0
_PEN_LOW_TRADE_EXP = 1.2
_PEN_PF_BELOW_ONE_WEIGHT = 200.0
_PEN_PF_BELOW_ONE_EXP = 1.0
_PEN_OVERTRADE_START = 80
_PEN_OVERTRADE_WEIGHT = 2.0
_PEN_OVERTRADE_EXP = 1.1


def _safe_profit_factor(pf: float) -> float:
    if pf == float("inf") or (isinstance(pf, float) and math.isnan(pf)):
        return 25.0
    return min(max(float(pf), 0.001), 50.0)


def _penalized_pnl_scalar(report: BacktestReport) -> float:
    """PnL net (quote) moins pénalités : DD, échantillon trop petit, PF < 1, sur-trading."""
    pnl = float(report.net_pnl_quote)
    dd = max(float(report.max_drawdown_quote), 0.0)
    nt = int(report.total_trades)
    pf = _safe_profit_factor(float(report.profit_factor))

    dd_pen = _PEN_DD_WEIGHT * dd

    deficit = max(0, _PEN_MIN_TRADES - nt)
    low_trade_pen = _PEN_LOW_TRADE_WEIGHT * (deficit**_PEN_LOW_TRADE_EXP) if deficit else 0.0

    pf_pen = 0.0
    if pf < 1.0:
        pf_pen = _PEN_PF_BELOW_ONE_WEIGHT * ((1.0 - pf) ** _PEN_PF_BELOW_ONE_EXP)

    over_pen = 0.0
    if nt > _PEN_OVERTRADE_START:
        over_pen = _PEN_OVERTRADE_WEIGHT * ((nt - _PEN_OVERTRADE_START) ** _PEN_OVERTRADE_EXP)

    return pnl - dd_pen - low_trade_pen - pf_pen - over_pen


def _composite_scalar(report: BacktestReport) -> float:
    """Combine plusieurs métriques (échelles fixes, pas de normalisation sur tout le batch)."""
    pf = report.profit_factor
    if pf == float("inf") or (isinstance(pf, float) and math.isnan(pf)):
        pf = 10.0
    pf = min(max(float(pf), 0.0), 10.0)
    trades = float(report.total_trades)
    return (
        0.28 * math.tanh(report.net_pnl_quote / 400.0)
        + 0.20 * math.tanh((pf - 1.0) / 2.5)
        + 0.18 * math.tanh(report.expectancy_r * 2.0)
        + 0.16 * math.tanh(report.net_r / 8.0)
        + 0.18 * math.tanh(-report.max_drawdown_quote / 250.0)
        - 0.10 * math.tanh(trades / 100.0)
    )


def _objective_score(report: BacktestReport, objective: str) -> tuple[float, ...]:
    obj = (objective or "net_pnl_quote").strip().lower()
    if obj == "net_r":
        return (report.net_r, report.profit_factor, -report.max_drawdown_r)
    if obj == "profit_factor":
        return (report.profit_factor, report.net_pnl_quote, -report.max_drawdown_quote)
    if obj == "expectancy_r":
        return (report.expectancy_r, report.net_pnl_quote, -report.max_drawdown_quote)
    if obj == "sharpe_like":
        ratio = report.expectancy_r / report.max_drawdown_r if report.max_drawdown_r > 0 else report.expectancy_r
        return (ratio, report.net_pnl_quote, -report.max_drawdown_quote)
    if obj == "composite":
        s = _composite_scalar(report)
        return (s, report.net_pnl_quote, report.profit_factor, -report.max_drawdown_quote)
    if obj in ("penalized_pnl_quote", "penalized_net_pnl"):
        s = _penalized_pnl_scalar(report)
        pf = _safe_profit_factor(float(report.profit_factor))
        return (s, report.net_pnl_quote, pf, -report.max_drawdown_quote)
    # Default objective: quote-denominated net PnL
    return (report.net_pnl_quote, report.profit_factor, -report.max_drawdown_quote)


def report_optimization_export_metrics(report: BacktestReport, *, objective: str) -> dict[str, Any]:
    """Métriques sérialisables pour export JSON (file d’optimisation, rapports)."""
    pf = float(report.profit_factor)
    pf_out: float | None
    if pf == float("inf"):
        pf_out = None
    elif isinstance(pf, float) and math.isnan(pf):
        pf_out = None
    else:
        pf_out = round(pf, 4)
    obj = (objective or "net_pnl_quote").strip().lower()
    out: dict[str, Any] = {
        "total_trades": report.total_trades,
        "wins": report.wins,
        "losses": report.losses,
        "win_rate": round(report.win_rate, 4),
        "profit_factor": pf_out,
        "expectancy_r": round(report.expectancy_r, 4),
        "net_r": round(report.net_r, 4),
        "net_pnl_quote": round(report.net_pnl_quote, 4),
        "max_drawdown_quote": round(report.max_drawdown_quote, 4),
        "max_drawdown_r": round(report.max_drawdown_r, 4),
        "realized_gains_quote": round(report.realized_gains_quote, 4),
        "realized_losses_quote": round(report.realized_losses_quote, 4),
        "optimization_objective_used": obj,
    }
    if obj in ("penalized_pnl_quote", "penalized_net_pnl"):
        out["penalized_pnl_adjusted_score"] = round(_penalized_pnl_scalar(report), 4)
    return out
