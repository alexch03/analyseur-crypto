"""Optimisation in-sample puis évaluation hors-échantillon sur la fenêtre chronologique suivante.

Découpe la série en (n_splits + 1) segments consécutifs de même taille. Pour k = 0 .. n_splits-1 :
  - train = concaténation des segments [0 .. k]
  - test  = segment k+1 (jamais vu à l'optimisation)

Sur ``train``, on lance la grille ``optimize_setup_parameters`` ; le rang 1 est ré-appliqué sur ``test``
avec le même moteur de replay (même warmup / frais). Les métriques OOS mesurent la robustesse hors
échantillon par rapport au seul backtest global in-sample.
"""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Any

import pandas as pd

from app.paper.engine_replay import replay_engine_from_bt_cfg
from app.services.optimizer import optimize_setup_parameters


def _report_summary(r: Any) -> dict[str, Any]:
    pf = r.profit_factor
    return {
        "total_trades": r.total_trades,
        "wins": r.wins,
        "losses": r.losses,
        "win_rate": round(r.win_rate, 4),
        "profit_factor": round(pf, 4) if pf != float("inf") else None,
        "net_r": round(r.net_r, 4),
        "net_pnl_quote": round(r.net_pnl_quote, 4),
        "realized_gains_quote": round(r.realized_gains_quote, 4),
        "realized_losses_quote": round(r.realized_losses_quote, 4),
        "max_drawdown_quote": round(r.max_drawdown_quote, 4),
    }


def run_walk_forward_oos(
    ohlcv_df: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    objective: str,
    backtest_config: dict[str, Any],
    rr_min_values: list[float],
    fvg_proximity_values: list[float],
    ob_proximity_values: list[float],
    swing_left_values: list[int],
    swing_right_values: list[int],
    max_setups_values: list[int],
    n_splits: int = 3,
    optimization_strategy: str = "exhaustive",
    optimization_max_trials: int | None = None,
) -> dict[str, Any]:
    """Retourne une ligne par split : métriques IS (meilleur sur train) vs OOS (même params sur test)."""
    methodology_fr = (
        "Découpe chronologique : optimisation uniquement sur le passé agrégé (train), "
        "puis une seule évaluation sur la tranche suivante (test), sans ré-optimiser sur le test. "
        "Si le PnL OOS s'effondre alors que l'IS est bon, la grille est probablement sur-ajustée."
    )

    if "timestamp" in ohlcv_df.columns:
        df = ohlcv_df.sort_values("timestamp").reset_index(drop=True)
    else:
        df = ohlcv_df.copy().reset_index(drop=True)

    n = len(df)
    warmup = int(backtest_config.get("warmup_bars", 120))
    min_seg = max(warmup * 2, 250)
    max_splits = max(1, n // min_seg - 1)
    eff_splits = max(1, min(int(n_splits), max_splits))
    seg_len = n // (eff_splits + 1)
    if seg_len < min_seg:
        return {
            "ok": False,
            "error": f"Série trop courte pour {n_splits} splits (n={n}, segment min ~{min_seg}).",
            "methodology_fr": methodology_fr,
            "splits": [],
            "oos_summary": {},
        }

    splits_out: list[dict[str, Any]] = []
    oos_pnls: list[float] = []

    for k in range(eff_splits):
        train_end = (k + 1) * seg_len
        test_start = train_end
        test_end = min(test_start + seg_len, n)
        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()
        if len(train_df) < warmup + 80 or len(test_df) < warmup + 40:
            continue

        ranked = optimize_setup_parameters(
            train_df,
            symbol=symbol,
            timeframe=timeframe,
            objective=objective,
            backtest_config=backtest_config,
            rr_min_values=rr_min_values,
            fvg_proximity_values=fvg_proximity_values,
            ob_proximity_values=ob_proximity_values,
            swing_left_values=swing_left_values,
            swing_right_values=swing_right_values,
            max_setups_values=max_setups_values,
            strategy=optimization_strategy,
            max_trials=optimization_max_trials,
        )
        if not ranked:
            continue
        best = ranked[0]
        p = best.params
        engine = replay_engine_from_bt_cfg(backtest_config)
        engine_params = {
            "rr_min": p["rr_min"],
            "fvg_proximity_pct": p["fvg_proximity_pct"],
            "ob_proximity_pct": p["ob_proximity_pct"],
            "max_setups": p["max_setups"],
        }
        for fk in ("require_ifvg_confluence", "ifvg_confluence_pct", "require_rsi_divergence"):
            if fk in backtest_config:
                engine_params[fk] = backtest_config[fk]
        oos_report = engine.run_walkforward(
            test_df,
            symbol=symbol,
            timeframe=timeframe,
            swing_left=int(p["swing_left"]),
            swing_right=int(p["swing_right"]),
            engine_params=engine_params,
        )
        oos_pnls.append(float(oos_report.net_pnl_quote))
        splits_out.append(
            {
                "split_index": k + 1,
                "train_bars": len(train_df),
                "test_bars": len(test_df),
                "train_range": [0, train_end - 1],
                "test_range": [test_start, test_end - 1],
                "best_params": p,
                "in_sample": _report_summary(best.report),
                "out_of_sample": _report_summary(oos_report),
                "oos_minus_is_net_pnl": round(oos_report.net_pnl_quote - best.report.net_pnl_quote, 4),
            }
        )

    if not splits_out:
        return {
            "ok": False,
            "error": "Aucun split valide (données ou warmup).",
            "methodology_fr": methodology_fr,
            "splits": [],
            "oos_summary": {},
        }

    pos_oos = sum(1 for x in oos_pnls if x > 0)
    oos_std = round(pstdev(oos_pnls), 4) if len(oos_pnls) > 1 else 0.0
    summary = {
        "splits_count": len(splits_out),
        "oos_net_pnl_mean": round(mean(oos_pnls), 4),
        "oos_net_pnl_stdev": oos_std,
        "oos_positive_splits": pos_oos,
        "interpretation_fr": (
            f"{pos_oos}/{len(oos_pnls)} fenêtres OOS avec PnL net > 0. "
            "Comparer la moyenne OOS au PnL d'une optimisation classique sur toute la série."
        ),
    }
    return {
        "ok": True,
        "methodology_fr": methodology_fr,
        "n_splits_requested": n_splits,
        "n_splits_effective": eff_splits,
        "segment_bars": seg_len,
        "splits": splits_out,
        "oos_summary": summary,
    }
