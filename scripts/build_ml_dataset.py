"""Construit le dataset ML enrichi avec les indicateurs techniques OHLCV.

PROBLEME : build_dataset() (dataset.py) laisse tous les indicateurs (adx,
vol_ratio, bb_pos, entry_body_ratio...) a NaN car la requete SQL ne calcule
pas de series de prix. Ce script les recalcule a partir des OHLCV reels.

Ce que fait ce script :
  1. Charge les trades clos (TARGET_HIT/STOPPED) depuis la DB.
     Mode --backfill : backfill frais sur une DB temporaire (propre, pas
     de contamination par runs pre-fix regime).
  2. Pour chaque (symbole, timeframe), recupere les OHLCV via CCXT.
  3. Aligne chaque trade sur sa bougie d'entree -> appelle compute_features().
  4. Injecte les indicateurs dans le dataset, sauvegarde data/ml/dataset.csv.
  5. Lance l'evaluation walk-forward (logreg + gbm) sauf si --no-eval.
  6. Lance aussi une evaluation parcimoneuse (4 features validees WF seules).

Usage :
    # Dataset depuis DB live (peut etre contaminee -- voir handoff)
    python scripts/build_ml_dataset.py [--db analyseur.db]

    # Dataset propre via backfill frais (recommande)
    python scripts/build_ml_dataset.py --backfill [--days 21] [--symbols ...]

    # Reutilise la DB temp existante (skip le backfill)
    python scripts/build_ml_dataset.py --backfill --reuse

    # Juste reconstruire le CSV sans relancer l'eval (debug)
    python scripts/build_ml_dataset.py --backfill --reuse --no-eval
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# --- DATABASE_URL doit etre defini AVANT tout import d'app ---------------------
# On check sys.argv directement (argparse n'est pas encore instancie).
_TMP_DB = Path(tempfile.gettempdir()) / "analyseur_ml_dataset.db"
if "--backfill" in sys.argv:
    if "--reuse" not in sys.argv and _TMP_DB.exists():
        _TMP_DB.unlink()
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TMP_DB.as_posix()}"
# ------------------------------------------------------------------------------

import argparse
import asyncio
import sqlite3

import numpy as np
import pandas as pd

# Features parcimoneuses validees en walk-forward OOS (ne pas elargir sans re-validation).
# adx ~ trend_strength, vol_ratio ~ volume_spike, bb_pos ~ bb_zscore.
PARSIMONIOUS = ["adx", "vol_ratio", "bb_pos", "entry_body_ratio"]

_SQL_WITH_SYM = """
SELECT
  u.id, u.hypothesis_id, u.symbol_id,
  s.base || '/' || s.quote AS symbol,
  u.timeframe_id, u.side, u.pattern_kind,
  u.entry_price, u.entry_timestamp, u.pct_gain, u.outcome,
  u.confluence_score, u.confluence_tags,
  h.entry_price        AS h_entry,
  h.target_price       AS target_price,
  h.invalidation_price AS invalidation_price,
  h.pattern_snapshot   AS pattern_snapshot,
  (SELECT trend    FROM market_regime_snapshots r
     WHERE r.snapshot_ts <= u.entry_timestamp
     ORDER BY r.snapshot_ts DESC LIMIT 1) AS regime_trend,
  (SELECT strength FROM market_regime_snapshots r
     WHERE r.snapshot_ts <= u.entry_timestamp
     ORDER BY r.snapshot_ts DESC LIMIT 1) AS regime_strength
FROM unit_trades u
JOIN hypotheses h  ON h.id = u.hypothesis_id
JOIN symbols    s  ON s.id = u.symbol_id
WHERE u.outcome IN ('TARGET_HIT', 'STOPPED')
ORDER BY u.entry_timestamp
"""


def _bars_for(days: int, tf: str) -> int:
    per_day = {"1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}.get(tf, 96)
    return days * per_day


def _inject_indicators(df: pd.DataFrame, ohlcv_cache: dict) -> pd.DataFrame:
    """Calcule compute_features() a la bougie d'entree de chaque trade.

    Modifie df en place, retourne df.
    Requiert la colonne 'symbol' (nom complet style 'BTC/USDT') dans df.
    """
    from app.ml.indicators import compute_features, INDICATOR_OHLCV

    injected = 0
    missing_cache = set()
    for i, row in df.iterrows():
        sym = str(row.get("symbol", ""))
        tf = str(row.get("timeframe_id", ""))
        key = (sym, tf)
        ohlcv = ohlcv_cache.get(key)
        if ohlcv is None or ohlcv.empty:
            missing_cache.add(key)
            continue

        ts = pd.Timestamp(row["entry_timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        tss = pd.to_datetime(ohlcv["timestamp"], utc=True)
        matches = tss[tss <= ts]
        if matches.empty:
            continue
        # entry_idx = position dans le DataFrame (RangeIndex => label == position)
        entry_idx = int(matches.index[-1])
        if entry_idx < 30:
            continue

        feats = compute_features(ohlcv, idx=entry_idx)
        for col, val in feats.items():
            df.at[i, col] = val
        injected += 1

    pct = 100 * injected // max(1, len(df))
    print(f"  Indicateurs injectes : {injected}/{len(df)} trades ({pct}%)")
    if missing_cache:
        print(f"  OHLCV manquant pour {len(missing_cache)} paires : {sorted(missing_cache)[:5]}")
    return df


def build_enriched_dataset(db_path: Path, ohlcv_cache: dict) -> pd.DataFrame:
    """Construit le dataset ML avec indicateurs OHLCV peuples.

    1. Charge les trades + noms de symboles depuis db_path.
    2. Injecte les indicateurs techniques via ohlcv_cache.
    3. Appelle engineer_features() pour calculer toutes les features.
    4. Retourne un DataFrame prêt pour evaluate_model().
    """
    from app.ml.dataset import engineer_features
    from app.ml.features import ALL_FEATURES, LABEL, META_COLUMNS
    from app.ml.indicators import INDICATOR_OHLCV

    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(_SQL_WITH_SYM, conn)
    finally:
        conn.close()

    if df.empty:
        raise RuntimeError(f"Aucun trade clos (TARGET_HIT/STOPPED) dans {db_path}")

    print(f"  Trades charges : {len(df)} (TARGET_HIT={int((df['outcome']=='TARGET_HIT').sum())}, "
          f"STOPPED={int((df['outcome']=='STOPPED').sum())})")

    # Initialise les colonnes indicateurs a NaN avant injection
    for col in INDICATOR_OHLCV:
        df[col] = np.nan

    df = _inject_indicators(df, ohlcv_cache)

    # Label + engineer_features
    df[LABEL] = (df["outcome"] == "TARGET_HIT").astype(int)
    df = engineer_features(df)

    keep = META_COLUMNS + ALL_FEATURES + [LABEL]
    # 'symbol' n'est pas dans keep -> dropped automatiquement
    return df[[c for c in keep if c in df.columns]].copy()


def _eval_parsimonious(df: pd.DataFrame, n_splits: int = 5) -> None:
    """Eval walk-forward parcimoneuse : 4 features validees OOS uniquement.

    Evite le sur-apprentissage du kitchen-sink (one-hot pattern_kind ~ 16 modalites).
    Utilise une logistic regression L2 simple.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from app.ml.dataset import load_dataset
    from app.ml.evaluate import economic, ev_mask, profit_factor

    print("=" * 78)
    print("EVAL PARCIMONEUSE (4 features validees walk-forward)")
    print(f"  Features : {PARSIMONIOUS}")
    print("=" * 78)

    available = [f for f in PARSIMONIOUS if f in df.columns]
    if len(available) < 2:
        print(f"  Seulement {len(available)} feature(s) disponible(s) -- skip.")
        return

    missing = [f for f in PARSIMONIOUS if f not in available]
    if missing:
        print(f"  Attention : {missing} absentes (NaN 100%).")

    df2 = df.sort_values("entry_timestamp").reset_index(drop=True)
    X_raw = df2[available].copy().astype(float)
    y = df2["label"].astype(int)
    rr = pd.to_numeric(df2["rr"], errors="coerce").fillna(1.5).to_numpy()
    pg = pd.to_numeric(df2["pct_gain"], errors="coerce").fillna(0.0).to_numpy()

    nan_pct = X_raw.isna().mean()
    for feat in available:
        pct = int(100 * nan_pct[feat])
        if pct > 50:
            print(f"  WARN: {feat} a {pct}% de NaN -> signal faible/absent")

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(C=0.5, max_iter=500, random_state=42)),
    ])

    tss = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for tr, te in tss.split(X_raw):
        if y.iloc[tr].nunique() < 2:
            continue
        pipe.fit(X_raw.iloc[tr], y.iloc[tr])
        prob = pipe.predict_proba(X_raw.iloc[te])[:, 1]
        for j, i in enumerate(te):
            rows.append({"y": int(y.iloc[i]), "prob": float(prob[j]),
                         "pct_gain": float(pg[i]), "rr": float(rr[i])})

    if not rows:
        print("  Pas assez de donnees pour OOS.")
        return

    oos = pd.DataFrame(rows)
    y_arr = oos["y"].to_numpy()
    prob_arr = oos["prob"].to_numpy()
    pg_arr = oos["pct_gain"].to_numpy()

    auc = roc_auc_score(y_arr, prob_arr) if len(np.unique(y_arr)) > 1 else float("nan")
    base_wr = 100.0 * y_arr.mean()
    print(f"\n  OOS pool: {len(oos)} trades | base WR={base_wr:.1f}% | AUC={auc:.3f}")

    # Politique EV : P > 1/(1+RR)
    p_star = 1.0 / (1.0 + oos["rr"].to_numpy())
    ev = (prob_arr > p_star)
    n_ev = int(ev.sum())
    if n_ev > 0:
        sub_pg = pg_arr[ev]
        wins = (sub_pg > 0).sum()
        wr_ev = 100.0 * wins / n_ev
        pf_ev = profit_factor(sub_pg)
        pnl_ev = float(sub_pg.sum())
        print(f"  Politique EV : n={n_ev}  WR={wr_ev:.1f}%  PF={pf_ev:.2f}  PnL={pnl_ev:+.1f}%")
        suff = pf_ev >= 1.2 and pnl_ev > 0 and n_ev >= 30
        print(f"  SUFFISANT (PF>=1.2, PnL>0, n>=30) ? {'OUI' if suff else 'NON'}")
    else:
        print("  Politique EV : 0 trades acceptes.")

    # Balayage seuil rapide
    print(f"\n  {'seuil':>6} {'n':>5} {'WR%':>6} {'PnL%':>9} {'PF':>6}")
    for thr in np.arange(0.35, 0.66, 0.05):
        m = prob_arr >= thr
        n_m = int(m.sum())
        if n_m == 0:
            continue
        sub = pg_arr[m]
        pf = profit_factor(sub)
        pnl = float(sub.sum())
        wr_t = 100.0 * (sub > 0).mean()
        print(f"  {thr:>6.2f} {n_m:>5} {wr_t:>6.1f} {pnl:>9.1f} {pf:>6.2f}")


async def _run_backfill(args) -> None:
    """Lance le backfill fidele (meme config que le live) sur la DB temp."""
    from app.config import settings
    from app.db.models import Base
    from app.db.session import engine as app_engine
    from app.ingestion.ccxt_fetcher import CCXTFetcher
    from app.services.continuous_scanner import (
        ContinuousScanner, ScanPlan, _rows_to_df, build_regime_timeline,
    )

    print(f"DB backtest : {settings.database_url}")
    assert "ml_dataset" in settings.database_url, "DATABASE_URL non redirige !"

    async with app_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    tfs = [t.strip() for t in args.tfs.split(",") if t.strip()]
    # Calcule l'historique en bars pour le TF le plus fin (plus grande valeur)
    history_bars = max(_bars_for(args.days, tf) for tf in tfs)

    print(f"Backfill fidele : {len(symbols)} symboles x {tfs}, {args.days}j (~{history_bars} bougies max)")

    fetcher = CCXTFetcher(settings.exchange_id)
    print("Fetch BTC/USDT 1h pour timeline de regime...")
    btc_rows = await fetcher.fetch_ohlcv("BTC/USDT", "1h", limit=500)
    btc_df = _rows_to_df(btc_rows)
    await fetcher.close()
    timeline = build_regime_timeline(btc_df)
    print(f"Timeline regime : {len(timeline)} points")

    plan = ScanPlan(
        symbols=symbols,
        timeframes=tfs,
        candles_per_fetch=history_bars + 60,
    )
    scanner = ContinuousScanner(plan=plan)
    result = await scanner.backfill(
        history_bars=history_bars,
        bars_per_step=1,
        symbols=symbols,
        timeframes=tfs,
        regime_timeline=timeline,
    )
    await scanner.stop()
    print(f"Backfill termine : {result['total_steps']} steps, "
          f"{result['total_patterns_detected']} patterns, "
          f"{result['elapsed_seconds']}s")


async def _fetch_ohlcv_cache(symbols: list[str], tfs: list[str],
                              days: int) -> dict:
    """Retourne {(symbol, tf): DataFrame} pour le calcul des indicateurs."""
    from app.config import settings
    from app.ingestion.ccxt_fetcher import CCXTFetcher
    from app.services.continuous_scanner import _rows_to_df

    fetcher = CCXTFetcher(settings.exchange_id)
    cache = {}
    total = len(symbols) * len(tfs)
    done = 0
    for sym in symbols:
        for tf in tfs:
            bars = _bars_for(days, tf) + 60
            try:
                rows = await fetcher.fetch_ohlcv(sym, tf, limit=bars)
                df = _rows_to_df(rows)
                if not df.empty:
                    df = df.reset_index(drop=True)
                    cache[(sym, tf)] = df
            except Exception as exc:
                print(f"  WARN fetch {sym}/{tf}: {exc}")
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  OHLCV : {done}/{total} paires fetchees...")
    await fetcher.close()
    return cache


async def main() -> int:
    ap = argparse.ArgumentParser(
        description="Construit le dataset ML enrichi avec indicateurs OHLCV.")
    ap.add_argument("--db", default=None,
                    help="Chemin DB existante (defaut: analyseur.db)")
    ap.add_argument("--backfill", action="store_true",
                    help="Genere un backfill frais sur DB temporaire")
    ap.add_argument("--days", type=int, default=21,
                    help="Historique en jours (mode --backfill, defaut 21)")
    ap.add_argument("--tfs", default="15m,1h,4h",
                    help="Timeframes a backfiller (virgule, defaut 15m,1h,4h)")
    ap.add_argument("--symbols",
                    default="BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,ADA/USDT,"
                            "AVAX/USDT,DOGE/USDT,LINK/USDT,DOT/USDT,LTC/USDT,ATOM/USDT",
                    help="Symboles a backfiller (virgule)")
    ap.add_argument("--reuse", action="store_true",
                    help="Reutilise la DB temp existante (skip le backfill)")
    ap.add_argument("--no-eval", action="store_true",
                    help="Sauvegarde le CSV sans lancer l'evaluation walk-forward")
    args = ap.parse_args()

    # Determine le chemin de la DB source
    if args.backfill:
        db_path = _TMP_DB
        if not args.reuse:
            await _run_backfill(args)
        else:
            print(f"--reuse : utilise {_TMP_DB.name} existante")
        if not db_path.exists():
            print("Erreur : la DB temp n'existe pas. Lancez sans --reuse d'abord.")
            return 1
    else:
        if args.db:
            db_path = Path(args.db)
        else:
            db_path = ROOT / "analyseur.db"
        if not db_path.exists():
            print(f"Erreur : DB non trouvee : {db_path}")
            print("  Utilisez --db <chemin> ou --backfill pour un dataset propre.")
            return 1
        print(f"Mode DB existante : {db_path}")
        print("  ATTENTION : la DB live peut etre contaminee (runs pre-fix regime).")
        print("  Recommande : utiliser --backfill pour un dataset propre.")

    # Determine les symboles et TFs a fetcher pour l'OHLCV
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    tfs = [t.strip() for t in args.tfs.split(",") if t.strip()]

    # En mode DB existante, lire les (symbol, tf) reels depuis la DB
    if not args.backfill:
        try:
            conn = sqlite3.connect(str(db_path))
            rows = conn.execute("""
                SELECT DISTINCT s.base||'/'||s.quote, u.timeframe_id
                FROM unit_trades u
                JOIN symbols s ON s.id = u.symbol_id
                WHERE u.outcome IN ('TARGET_HIT','STOPPED')
            """).fetchall()
            conn.close()
            if rows:
                symbols = sorted(set(r[0] for r in rows))
                tfs = sorted(set(str(r[1]) for r in rows))
                print(f"  Symboles en DB : {len(symbols)} | TFs : {tfs}")
        except Exception as exc:
            print(f"  WARN lecture DB pour symboles : {exc}")

    print(f"\nFetch OHLCV : {len(symbols)} symboles x {len(tfs)} TFs...")
    ohlcv_cache = await _fetch_ohlcv_cache(symbols, tfs, args.days)
    print(f"Cache OHLCV : {len(ohlcv_cache)} paires chargees")

    print("\nConstruction du dataset enrichi...")
    df = build_enriched_dataset(db_path, ohlcv_cache)

    # Diagnostic NaN des indicateurs
    from app.ml.indicators import INDICATOR_OHLCV
    n = len(df)
    print(f"\nDataset : {n} trades | WR global={100*df['label'].mean():.1f}%")
    print("NaN par indicateur apres injection :")
    for col in INDICATOR_OHLCV:
        if col in df.columns:
            miss = int(df[col].isna().sum())
            pct = 100 * miss // max(1, n)
            flag = " <-- PROBLEME" if pct > 80 else ""
            print(f"  {col:<20} {miss:>4}/{n} NaN ({pct}%){flag}")

    # Sauvegarde
    out_path = ROOT / "data" / "ml" / "dataset.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSauvegarde -> {out_path}")

    if args.no_eval:
        print("--no-eval : evaluation skip.")
        return 0

    # Evaluation walk-forward complete
    from app.ml.evaluate import evaluate_model
    n_splits = min(5, max(2, n // 30))
    print(f"\nEvaluation walk-forward ({n_splits} folds) sur {n} trades...")
    for mt in ("logreg", "gbm"):
        evaluate_model(df, mt, n_splits=n_splits)
        print()

    # Evaluation parcimoneuse (4 features validees WF)
    _eval_parsimonious(df, n_splits=n_splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
