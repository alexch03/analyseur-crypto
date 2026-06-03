"""Phase 1 DOE : le DETECTEUR bat-il des entrees ALEATOIRES appariees ?

Pour chaque trade du detecteur (dataset.csv 4h/150j), on genere K controles aleatoires
qui partagent TOUT sauf la bougie d'entree :
  - meme symbole, meme side, meme stop%/cible%, meme REGIME (bar tire au hasard
    parmi les bougies du symbole dans le meme regime).
On resout detecteur ET aleatoire avec EXACTEMENT la meme logique (forward, stop-first
conservateur) sur les OHLCV reels -> comparaison apples-to-apples.

Sortie : par regime, apres couts, esperance + PF, et IC bootstrap sur la difference
(detecteur - aleatoire). Jackknife des jours de krach.

Usage : PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/detector_vs_random.py
"""
from __future__ import annotations
import sys, sqlite3, tempfile, asyncio
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TMP = Path(tempfile.gettempdir()) / "analyseur_ml_dataset.db"
COST = 0.2          # cout round-trip en %
K = 5               # controles aleatoires par trade detecteur
SEED = 42
CRASH_DAYS = {"2026-02-03", "2026-02-04", "2026-02-05"}  # jackknife


def pf(x: np.ndarray) -> float:
    g = x[x > 0].sum(); l = x[x < 0].sum()
    return float("inf") if l == 0 else abs(g / l)


def resolve(hi, lo, cl, idx, side, stop_pct, tgt_pct):
    """Rendement realise (%) en partant de la bougie idx, resolution forward,
    stop-first conservateur. None si non resolu avant la fin des donnees."""
    entry = cl[idx]
    n = len(cl)
    if side == "LONG":
        stop = entry * (1 - stop_pct); tgt = entry * (1 + tgt_pct)
        for j in range(idx + 1, n):
            hit_stop = lo[j] <= stop; hit_tgt = hi[j] >= tgt
            if hit_stop:  # stop d'abord si les deux dans la meme bougie
                return -stop_pct * 100.0
            if hit_tgt:
                return tgt_pct * 100.0
    else:
        stop = entry * (1 + stop_pct); tgt = entry * (1 - tgt_pct)
        for j in range(idx + 1, n):
            hit_stop = hi[j] >= stop; hit_tgt = lo[j] <= tgt
            if hit_stop:
                return -stop_pct * 100.0
            if hit_tgt:
                return tgt_pct * 100.0
    return None


async def fetch_ohlcv(symbols, tf="4h", limit=1000):
    from app.config import settings
    from app.ingestion.ccxt_fetcher import CCXTFetcher
    from app.services.continuous_scanner import _rows_to_df, build_regime_timeline
    f = CCXTFetcher(settings.exchange_id)
    cache = {}
    for s in symbols:
        try:
            rows = await f.fetch_ohlcv(s, tf, limit=limit)
            df = _rows_to_df(rows).reset_index(drop=True)
            if not df.empty:
                cache[s] = df
        except Exception as e:
            print(f"  WARN {s}: {e}")
    btc = _rows_to_df(await f.fetch_ohlcv("BTC/USDT", "4h", limit=1000))
    await f.close()
    tl = build_regime_timeline(btc)
    return cache, tl


def regime_array(timeline):
    """(ts_ns trie, regimes) pour lookup vectorise via searchsorted."""
    if not timeline:
        return np.array([], dtype="int64"), np.array([], dtype=object)
    ts = np.array([pd.Timestamp(t).value for t, _ in timeline], dtype="int64")
    rg = np.array([r.trend for _, r in timeline], dtype=object)
    order = np.argsort(ts)
    return ts[order], rg[order]


def regime_at(ts_ns, rg_arr, query_ns):
    if len(ts_ns) == 0:
        return "NONE"
    i = np.searchsorted(ts_ns, query_ns, side="right") - 1
    return rg_arr[i] if i >= 0 else "NONE"


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(ROOT / "data" / "ml" / "dataset.csv")
    # symbol_id -> nom
    conn = sqlite3.connect(str(TMP))
    symmap = dict(conn.execute("SELECT id, base||'/'||quote FROM symbols").fetchall())
    conn.close()
    df["symbol"] = df["symbol_id"].map(symmap)
    df = df.dropna(subset=["symbol"]).copy()
    df["ts"] = pd.to_datetime(df["entry_timestamp"], utc=True, errors="coerce")
    print(f"Trades detecteur : {len(df)} | symboles : {df['symbol'].nunique()}")

    symbols = sorted(df["symbol"].unique())
    cache, timeline = asyncio.run(fetch_ohlcv(symbols))
    ts_arr, rg_arr = regime_array(timeline)
    print(f"OHLCV : {len(cache)} symboles | timeline regime : {len(timeline)} pts")

    # Pre-calc par symbole : arrays + regime par bougie + pools d'indices par regime
    pools = {}   # (symbol, regime) -> liste d'indices eligibles (>=30, < n-1)
    arrs = {}    # symbol -> (hi, lo, cl, ts_ns)
    for s, o in cache.items():
        hi = o["high"].to_numpy(float); lo = o["low"].to_numpy(float)
        cl = o["close"].to_numpy(float)
        # tz-aware -> ns UTC robuste (astype int64 sur tz-aware est traitre)
        tns = o["timestamp"].to_numpy().astype("datetime64[ns]").astype("int64")
        arrs[s] = (hi, lo, cl, tns)
        bar_rg = np.array([regime_at(ts_arr, rg_arr, t) for t in tns], dtype=object)
        for r in ("BULL", "BEAR", "RANGE"):
            idxs = np.where((bar_rg == r) & (np.arange(len(o)) >= 30)
                            & (np.arange(len(o)) < len(o) - 1))[0]
            pools[(s, r)] = idxs

    det_rows = []   # (regime, ret, is_crash)
    rnd_rows = []
    n_unres_det = n_unres_rnd = 0
    sk_sym = sk_geom = sk_idx = 0
    for _, t in df.iterrows():
        s = t["symbol"]
        if s not in arrs:
            sk_sym += 1; continue
        hi, lo, cl, tns = arrs[s]
        side = str(t["side"]); sp = float(t["stop_dist_pct"]); tp = float(t["tgt_dist_pct"])
        if not (sp > 0 and tp > 0):
            sk_geom += 1; continue
        ent_ns = pd.Timestamp(t["ts"]).value
        eidx = int(np.searchsorted(tns, ent_ns, side="right") - 1)
        if eidx < 30 or eidx >= len(cl) - 1:
            sk_idx += 1; continue
        reg = regime_at(ts_arr, rg_arr, ent_ns)
        crash = str(pd.Timestamp(t["ts"]).date()) in CRASH_DAYS
        # detecteur (resolu avec la meme logique)
        rd = resolve(hi, lo, cl, eidx, side, sp, tp)
        if rd is None:
            n_unres_det += 1
        else:
            det_rows.append((reg, rd - COST, crash))
        # K controles aleatoires meme (symbole, side, stop/tgt, REGIME)
        pool = pools.get((s, reg))
        if pool is None or len(pool) == 0:
            continue
        for ridx in rng.choice(pool, size=min(K, len(pool)), replace=len(pool) < K):
            rr = resolve(hi, lo, cl, int(ridx), side, sp, tp)
            if rr is None:
                n_unres_rnd += 1
            else:
                rnd_rows.append((reg, rr - COST, crash))

    det = pd.DataFrame(det_rows, columns=["rg", "ret", "crash"])
    rnd = pd.DataFrame(rnd_rows, columns=["rg", "ret", "crash"])
    print(f"Skips: symbole={sk_sym} geom={sk_geom} index={sk_idx}")
    _s0 = next(iter(arrs)); _t = arrs[_s0][3]
    print(f"Ex {_s0}: {len(_t)} bougies, "
          f"{pd.Timestamp(_t[0]).date()}->{pd.Timestamp(_t[-1]).date()}")
    print(f"Resolus -> detecteur {len(det)} (non-resolus {n_unres_det}), "
          f"aleatoire {len(rnd)} (non-resolus {n_unres_rnd})\n")
    if len(det) == 0:
        print("0 resolu -> abandon"); return

    def boot_mean_ci(x, B=3000):
        if len(x) == 0:
            return (float("nan"), float("nan"))
        idx = rng.integers(0, len(x), size=(B, len(x)))
        means = x[idx].mean(axis=1)
        return (np.percentile(means, 2.5), np.percentile(means, 97.5))

    def block(label, dsub, rsub):
        de = dsub["ret"].to_numpy(); re = rsub["ret"].to_numpy()
        dm = de.mean() if len(de) else float("nan")
        rm = re.mean() if len(re) else float("nan")
        dlo, dhi = boot_mean_ci(de); rlo, rhi = boot_mean_ci(re)
        # IC bootstrap sur la difference des moyennes
        diff_lo = diff_hi = float("nan")
        if len(de) and len(re):
            B = 3000
            di = rng.integers(0, len(de), size=(B, len(de)))
            ri = rng.integers(0, len(re), size=(B, len(re)))
            diffs = de[di].mean(axis=1) - re[ri].mean(axis=1)
            diff_lo, diff_hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
        verdict = "EDGE" if diff_lo > 0 else ("anti" if diff_hi < 0 else "= hasard")
        print(f"  {label:18} | DET n={len(de):4} E[R]={dm:+.3f}% PF={pf(de):.2f} "
              f"| RND n={len(re):5} E[R]={rm:+.3f}% PF={pf(re):.2f} "
              f"| diff IC95=[{diff_lo:+.3f},{diff_hi:+.3f}] -> {verdict}")

    print("=== DETECTEUR vs ALEATOIRE (apparie), apres cout %.1f%% ===" % COST)
    block("GLOBAL", det, rnd)
    for r in ("BULL", "BEAR", "RANGE"):
        block(r, det[det.rg == r], rnd[rnd.rg == r])
    print("\n=== JACKKNIFE : hors jours de krach (%s) ===" % ",".join(sorted(CRASH_DAYS)))
    block("GLOBAL hors-krach", det[~det.crash], rnd[~rnd.crash])
    for r in ("BULL", "BEAR", "RANGE"):
        block(r + " hors-krach", det[(det.rg == r) & (~det.crash)], rnd[(rnd.rg == r) & (~rnd.crash)])


if __name__ == "__main__":
    main()
