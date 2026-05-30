"""Teste si COMBINER les variables predictives en un score ameliore vraiment la
decision -- avec la seule mesure qui compte : la VALIDATION CROISEE (out-of-sample).

Le piege : en in-sample, un modele entraine sur les memes trades qu'on evalue
parait TOUJOURS bon (overfitting). C'est l'illusion qui faisait croire a un
"85% winrate". La verite = performance sur des trades JAMAIS vus a l'entrainement.

Methode :
  1. Reutilise les trades + features deja collectes (DB tune_weights).
  2. Composite = regression logistique sur les variables fortes (signe appris).
  3. Compare :
       - IN-SAMPLE  : entraine et evalue sur tout (optimiste, trompeur)
       - 5-FOLD CV  : entraine sur 4/5, evalue sur le 1/5 restant (honnete)
  4. Verdict : le score bat-il le base rate (48.4%) OUT-OF-SAMPLE ?

Usage : python scripts/eval_composite.py
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_TMP_DB = Path(tempfile.gettempdir()) / "analyseur_tune_weights.db"  # peuplee par tune_adaptive_weights
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TMP_DB.as_posix()}"

import asyncio  # noqa: E402

import numpy as np  # noqa: E402

# Variables a combiner. On teste 2 ensembles : les 4 fortes, et les 11 toutes.
STRONG = ["trend_strength", "bb_zscore_revert", "entry_body_ratio", "volume_spike"]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logreg(X, y, l2=1.0, iters=1500, lr=0.3):
    n, m = X.shape
    w = np.zeros(m)
    b = 0.0
    for _ in range(iters):
        p = _sigmoid(X @ w + b)
        gw = X.T @ (p - y) / n + l2 * w / n
        gb = float(np.mean(p - y))
        w -= lr * gw
        b -= lr * gb
    return w, b


def auc(y, scores):
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Mann-Whitney U
    alls = np.concatenate([pos, neg])
    order = alls.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(alls) + 1)
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return u / (len(pos) * len(neg))


def metrics(y, p):
    take = p > 0.5
    skip = ~take
    wr_take = 100 * y[take].mean() if take.any() else float("nan")
    wr_skip = 100 * y[skip].mean() if skip.any() else float("nan")
    acc = 100 * ((p > 0.5) == (y == 1)).mean()
    return acc, wr_take, wr_skip, len(y[take]), auc(y, p)


def walk_forward_purged(X, y, entry_ns, exit_ns, *, n_test_blocks=5, init_frac=0.4,
                        embargo_frac=0.01):
    """Validation TEMPORELLE honnete (Lopez de Prado) : entraine sur le passe,
    teste sur le futur, avec PURGE (retire du train les trades dont la fenetre
    [entry,exit] chevauche le bloc de test) + EMBARGO (gap apres le train).

    Renvoie les predictions out-of-sample (NaN pour les trades jamais testes).
    """
    import numpy as np
    n = len(y)
    order = np.argsort(entry_ns)  # ordre chronologique
    Xo, yo = X[order], y[order]
    en, ex = entry_ns[order], exit_ns[order]
    span = max(1, en[-1] - en[0])
    embargo = embargo_frac * span

    p_oof = np.full(n, np.nan)
    start = int(n * init_frac)
    if start < 10 or n - start < n_test_blocks:
        return p_oof  # pas assez de donnees
    blocks = np.array_split(np.arange(start, n), n_test_blocks)
    for blk in blocks:
        if len(blk) == 0:
            continue
        test_lo = blk[0]
        test_start_entry = en[test_lo]
        # Train = tout trade AVANT le bloc dont la fenetre de label ne chevauche pas
        # le test (purge) et termine avant l'embargo.
        train_mask = np.zeros(n, dtype=bool)
        for j in range(test_lo):
            if ex[j] < test_start_entry - embargo:
                train_mask[j] = True
        if train_mask.sum() < 10:
            continue
        mu = Xo[train_mask].mean(0)
        sd = Xo[train_mask].std(0) + 1e-9
        w, b = fit_logreg((Xo[train_mask] - mu) / sd, yo[train_mask])
        p_oof[blk] = _sigmoid(((Xo[blk] - mu) / sd) @ w + b)
    # remet dans l'ordre original
    out = np.full(n, np.nan)
    out[order] = p_oof
    return out


async def main() -> int:
    from app.config import settings
    from app.ingestion.ccxt_fetcher import CCXTFetcher
    from app.services.continuous_scanner import _rows_to_df
    from app.strategy.adaptive import compute_evaluation_features
    import pandas as pd

    if not _TMP_DB.exists():
        print("DB tune_weights absente. Lance d'abord scripts/tune_adaptive_weights.py")
        return 1

    con = sqlite3.connect(str(_TMP_DB))
    cur = con.cursor()
    cur.execute("""
        SELECT s.base||'/'||s.quote, ut.side, ut.entry_timestamp, ut.pct_gain, tf.code,
               ut.exit_timestamp
        FROM unit_trades ut JOIN symbols s ON s.id = ut.symbol_id
        JOIN timeframes tf ON tf.id = ut.timeframe_id
    """)
    rows = cur.fetchall()
    con.close()

    symbols = sorted({r[0] for r in rows})
    tf = rows[0][4] if rows else "15m"
    fetcher = CCXTFetcher(settings.exchange_id)
    cache = {}
    for sym in symbols:
        cache[sym] = _rows_to_df(await fetcher.fetch_ohlcv(sym, tf, limit=1500))
    await fetcher.close()

    feat_rows, ys, entry_ns, exit_ns = [], [], [], []
    for sym, side, entry_ts, pct, _tf, exit_ts in rows:
        df = cache.get(sym)
        if df is None or df.empty:
            continue
        ts = pd.Timestamp(entry_ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        tss = pd.to_datetime(df["timestamp"], utc=True)
        m = tss[tss <= ts]
        if m.empty:
            continue
        idx = m.index[-1]
        if idx < 25:
            continue
        sl = df.iloc[: idx + 1]
        d = 1 if side == "LONG" else -1
        feats = compute_evaluation_features(
            sl["high"].to_numpy(float), sl["low"].to_numpy(float),
            sl["close"].to_numpy(float), sl["volume"].to_numpy(float),
            sl["open"].to_numpy(float), direction=d, entry=float(sl["close"].iloc[-1]),
        )
        if not all(k in feats for k in STRONG):
            continue
        ex = pd.Timestamp(exit_ts) if exit_ts else ts
        if ex.tzinfo is None:
            ex = ex.tz_localize("UTC")
        entry_ns.append(ts.value)
        exit_ns.append(ex.value)
        feat_rows.append(feats)
        ys.append(1 if (pct or 0) > 0 else 0)

    y = np.array(ys, dtype=float)
    en_arr = np.array(entry_ns, dtype=np.int64)
    ex_arr = np.array(exit_ns, dtype=np.int64)
    n = len(y)
    base_wr = 100 * y.mean()
    print(f"Dataset : {n} trades, base rate (prendre TOUT) = {base_wr:.1f}% winrate\n")
    if n < 30:
        print("Trop peu de trades pour une CV fiable.")
        return 0

    for fset_name, fset in (("4 variables FORTES", STRONG),
                            ("TOUTES (11)", sorted(feat_rows[0].keys()))):
        X = np.array([[fr[k] for k in fset] for fr in feat_rows], dtype=float)

        # ── IN-SAMPLE (optimiste) : standardise + fit + eval sur TOUT ──
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - mu) / sd
        w, b = fit_logreg(Xs, y)
        p_in = _sigmoid(Xs @ w + b)
        acc_i, wt_i, ws_i, ntake_i, auc_i = metrics(y, p_in)

        # ── 5-FOLD CV moyennee sur 20 seeds (honnete + robuste au hasard du split) ──
        wt_list, auc_list = [], []
        for seed in range(20):
            rng = np.random.RandomState(seed)
            idx = rng.permutation(n)
            folds = np.array_split(idx, 5)
            p_oof = np.full(n, np.nan)
            for fi in range(5):
                test_idx = folds[fi]
                train_idx = np.concatenate([folds[j] for j in range(5) if j != fi])
                mu_t, sd_t = X[train_idx].mean(0), X[train_idx].std(0) + 1e-9
                Xtr = (X[train_idx] - mu_t) / sd_t
                Xte = (X[test_idx] - mu_t) / sd_t
                wt, bt = fit_logreg(Xtr, y[train_idx])
                p_oof[test_idx] = _sigmoid(Xte @ wt + bt)
            _, wt_o, _, _, auc_o = metrics(y, p_oof)
            wt_list.append(wt_o)
            auc_list.append(auc_o)
        wt_mean, wt_std = float(np.mean(wt_list)), float(np.std(wt_list))
        auc_mean, auc_std = float(np.mean(auc_list)), float(np.std(auc_list))

        # ── WALK-FORWARD PURGE (la VRAIE validation, Lopez de Prado) ──
        p_wf = walk_forward_purged(X, y, en_arr, ex_arr)
        mask = ~np.isnan(p_wf)
        if mask.sum() >= 8:
            acc_w, wt_w, ws_w, ntake_w, auc_w = metrics(y[mask], p_wf[mask])
            n_wf = int(mask.sum())
        else:
            wt_w = auc_w = float("nan")
            n_wf = int(mask.sum())

        print("=" * 78)
        print(f"COMPOSITE sur {fset_name}")
        print("=" * 78)
        print(f"{'':<28}{'IN-SAMPLE':>11}{'CV aleatoire':>15}{'WALK-FWD purge':>16}")
        print(f"{'AUC (0.5=hasard)':<28}{auc_i:>11.3f}{auc_mean:>10.3f}±{auc_std:.2f}{auc_w:>16.3f}")
        print(f"{'WR si dit PRENDRE':<28}{wt_i:>10.1f}%{wt_mean:>9.1f}%±{wt_std:.1f}{wt_w:>15.1f}%")
        print(f"  base rate={base_wr:.1f}%  |  walk-fwd teste sur {n_wf} trades OOS")
        gain_cv = wt_mean - base_wr
        gain_wf = wt_w - base_wr if mask.sum() >= 8 else float("nan")
        if mask.sum() >= 8 and gain_wf > 5 and auc_w > 0.55:
            verdict = "EDGE CONFIRME (survit a la validation temporelle)"
        elif mask.sum() >= 8 and gain_wf > 0:
            verdict = "MARGINAL en walk-forward"
        else:
            verdict = "NON CONFIRME en walk-forward (l'edge CV etait optimiste)"
        print(f"  >>> Gain CV={gain_cv:+.1f}pts | Walk-fwd={gain_wf:+.1f}pts  ->  {verdict}\n")

    print("=" * 78)
    print("LECTURE (Lopez de Prado) : IN-SAMPLE ment, CV aleatoire FUIT (labels chevauchants),")
    print("WALK-FWD purge = la seule honnete. Si l'edge s'effondre de CV a walk-fwd, c'etait")
    print("de l'optimisme. Sur ~60 trades 1-regime c'est indicatif : confirmer sur 200+ trades.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
