"""Évaluation OUT-OF-SAMPLE (anti-sur-apprentissage) du modèle de sélection.

Méthodologie — la seule honnête pour « est-ce que ça aurait marché en live » :

  Walk-forward expanding (TimeSeriesSplit) : on trie par date, on entraîne sur le
  passé, on prédit le futur. Chaque probabilité OOS provient d'un modèle qui n'a
  JAMAIS vu ce trade ni aucun trade postérieur. Aucune fuite temporelle.

On juge ensuite le modèle non pas sur l'AUC mais sur l'ÉCONOMIE : le PnL/PF
réalisé du sous-ensemble de trades que la politique ACCEPTE, comparé aux
références (tout prendre ; filtre statique confluence>0.4).

Critère « modèle suffisant » : sur les trades acceptés OOS,
    PF >= 1.2  ET  PnL_total > 0  ET  n_acceptes >= 30.

Lancer :  PYTHONPATH=src python -m app.ml.evaluate
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from app.ml.dataset import load_dataset
from app.ml.model import build_pipeline, logreg_coefficients, split_xy, train_model

SUCCESS_PF = 1.2
SUCCESS_MIN_TRADES = 30


def profit_factor(pct_gains: np.ndarray) -> float:
    g = pct_gains[pct_gains > 0].sum()
    l = pct_gains[pct_gains < 0].sum()
    if l == 0:
        return float("inf") if g > 0 else 0.0
    return abs(g / l)


def economic(df: pd.DataFrame, mask: np.ndarray) -> dict:
    """Stats économiques du sous-ensemble accepté par ``mask``."""
    sub = df.loc[mask]
    n = len(sub)
    if n == 0:
        return {"n": 0, "wr": 0.0, "pnl": 0.0, "pf": 0.0, "expectancy": 0.0}
    pg = sub["pct_gain"].to_numpy()
    return {
        "n": n,
        "wr": 100.0 * (pg > 0).mean(),
        "pnl": float(pg.sum()),
        "pf": profit_factor(pg),
        "expectancy": float(pg.mean()),
    }


def oos_predictions(df: pd.DataFrame, model_type: str, n_splits: int = 5) -> pd.DataFrame:
    """Probabilités OOS par walk-forward expanding. Retourne les lignes testées."""
    df = df.sort_values("entry_timestamp").reset_index(drop=True)
    X, y = split_xy(df)
    rr = pd.to_numeric(df["rr"], errors="coerce")
    rr = rr.fillna(rr.median()).to_numpy()
    pg = pd.to_numeric(df["pct_gain"], errors="coerce").fillna(0.0).to_numpy()
    conf = pd.to_numeric(df["confluence_score"], errors="coerce").fillna(0.0).to_numpy()

    tss = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for fold, (tr, te) in enumerate(tss.split(X)):
        if y.iloc[tr].nunique() < 2:
            continue  # fold initial sans classe positive : insuffisant pour entraîner
        pipe = build_pipeline(model_type)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        prob = pipe.predict_proba(X.iloc[te])[:, 1]
        for j, i in enumerate(te):
            rows.append({
                "fold": fold, "y": int(y.iloc[i]), "prob": float(prob[j]),
                "pct_gain": float(pg[i]), "rr": float(rr[i]),
                "confluence_score": float(conf[i]),
                "pattern_kind": df["pattern_kind"].iloc[i],
            })
    return pd.DataFrame(rows)


def threshold_sweep(oos: pd.DataFrame) -> pd.DataFrame:
    out = []
    for thr in np.arange(0.10, 0.66, 0.05):
        m = (oos["prob"] >= thr).to_numpy()
        e = economic(oos, m)
        out.append({"threshold": round(float(thr), 2), **e})
    return pd.DataFrame(out)


def ev_mask(oos: pd.DataFrame) -> np.ndarray:
    """Règle d'espérance : accepter si P(gagnant) > 1/(1+RR) (espérance R positive)."""
    p_star = 1.0 / (1.0 + oos["rr"].to_numpy())
    return (oos["prob"].to_numpy() > p_star)


def _print_block(title: str, e: dict) -> None:
    print(f"  {title:34} n={e['n']:4}  wr={e['wr']:5.1f}%  "
          f"PnL={e['pnl']:+8.1f}%  PF={e['pf']:.2f}  E[R]={e['expectancy']:+.3f}%")


def evaluate_model(df: pd.DataFrame, model_type: str, n_splits: int = 5) -> dict:
    print("=" * 78)
    print(f"MODÈLE: {model_type}   (walk-forward expanding, {n_splits} folds)")
    print("=" * 78)
    oos = oos_predictions(df, model_type, n_splits=n_splits)
    if oos.empty:
        print("  pas assez de données pour un OOS")
        return {}

    y, prob, pg = oos["y"].to_numpy(), oos["prob"].to_numpy(), oos["pct_gain"].to_numpy()
    auc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan")
    brier = brier_score_loss(y, prob)
    base_rate = 100.0 * y.mean()
    print(f"\n  OOS pool: {len(oos)} trades | base rate gagnants={base_rate:.1f}% | "
          f"AUC={auc:.3f} | Brier={brier:.3f}")
    print(f"  (AUC 0.5 = aléatoire ; >0.55 = signal exploitable)\n")

    # Références et politique
    _print_block("RÉFÉRENCE tout prendre", economic(oos, np.ones(len(oos), bool)))
    _print_block("RÉFÉRENCE confluence>0.4 (statique)",
                 economic(oos, (oos["confluence_score"] > 0.4).to_numpy()))
    ev = ev_mask(oos)
    _print_block("POLITIQUE EV  P>1/(1+RR)", economic(oos, ev))

    print("\n  --- Balayage de seuil sur P(gagnant) ---")
    sweep = threshold_sweep(oos)
    print(f"  {'seuil':>6} {'n':>5} {'wr%':>6} {'PnL%':>9} {'PF':>6} {'E[R]%':>7}")
    for _, r in sweep.iterrows():
        print(f"  {r['threshold']:>6.2f} {int(r['n']):>5} {r['wr']:>6.1f} "
              f"{r['pnl']:>9.1f} {r['pf']:>6.2f} {r['expectancy']:>7.3f}")

    # Verdict
    best = sweep.loc[sweep["pf"].replace([np.inf], 0).idxmax()] if not sweep.empty else None
    ev_e = economic(oos, ev)
    suff = (ev_e["pf"] >= SUCCESS_PF and ev_e["pnl"] > 0 and ev_e["n"] >= SUCCESS_MIN_TRADES)
    print(f"\n  >>> Politique EV: PF={ev_e['pf']:.2f}, PnL={ev_e['pnl']:+.1f}%, n={ev_e['n']}")
    print(f"  >>> MODÈLE SUFFISANT (PF>={SUCCESS_PF}, PnL>0, n>={SUCCESS_MIN_TRADES}) ? "
          f"{'OUI' if suff else 'NON'}")
    return {"auc": auc, "brier": brier, "oos": oos, "sweep": sweep,
            "ev": ev_e, "sufficient": bool(suff)}


def main() -> None:
    df = load_dataset()
    print(f"### Évaluation OOS sur {len(df)} trades clos\n")
    results = {}
    for mt in ("logreg", "gbm"):
        results[mt] = evaluate_model(df, mt)
        print()

    # Interprétabilité : coefficients logreg (entraînés sur TOUT le dataset).
    # In-sample, à lire comme « directions » et non comme performance.
    print("=" * 78)
    print("COEFFICIENTS LOGREG (in-sample, interprétation — quelles variables comptent)")
    print("=" * 78)
    pipe = train_model(df, "logreg")
    coefs = logreg_coefficients(pipe)
    print(coefs.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
