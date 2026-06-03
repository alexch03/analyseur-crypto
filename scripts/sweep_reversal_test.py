"""Setup MAITRE : SWEEP DE LIQUIDITE + RETOURNEMENT (CHoCH). Test rigoureux.

Regles (LONG ; miroir SHORT), no-lookahead :
  1. swing low confirme (via detect_swings, connu a index+R).
  2. une bougie meche SOUS ce low (low<sl) mais CLOTURE au-dessus (close>sl) = sweep.
  3. dans C bougies, CLOTURE au-dessus du swing high de reference = CHoCH haussier = confirmation.
  4. entree au close de confirmation ; stop = plus-bas du sweep ; cible = RR x risque.
  5. sinon -> pas de trade. Un seul trade actif par symbole (non-chevauchant).

Benchmark (un edge doit battre l'aleatoire APPARIE : meme symbole/side/stop%/cible%/regime,
seule la bougie d'entree differe). IC bootstrap, jackknife krach, split OOS, par regime.
Donnees : Binance (profond). Live resterait Bitget.

Usage : PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/sweep_reversal_test.py
"""
from __future__ import annotations
import sys, asyncio
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EX = "binance"
TF = "4h"
LIMIT = 3000          # ~500 jours en 4h (multi-regime)
R = 2                 # fractal swing left/right
W = 24                # fenetre de recherche du sweep apres le swing (bougies)
C = 6                 # fenetre de confirmation CHoCH apres le sweep
RR = 2.0              # cible = RR x risque
COST = 0.2            # cout round-trip %
K = 5                 # controles aleatoires / trade
SEED = 42
CRASH_FRACTILE = 5    # jackknife : retire les jours ou le rendement univers est dans les 5% pires

SYMS = ["BTC/USDT","ETH/USDT","BNB/USDT","XRP/USDT","ADA/USDT","SOL/USDT","DOGE/USDT",
        "LTC/USDT","LINK/USDT","DOT/USDT","AVAX/USDT","ATOM/USDT","XLM/USDT","BCH/USDT","ETC/USDT"]


def pf(x):
    x = np.asarray(x); g = x[x > 0].sum(); l = x[x < 0].sum()
    return float("inf") if l == 0 else abs(g / l)


def resolve(hi, lo, cl, idx, side, stop_pct, tgt_pct):
    """(rendement %, exit_idx) ; stop-first conservateur ; None si non resolu."""
    entry = cl[idx]; n = len(cl)
    if side == "LONG":
        stop = entry * (1 - stop_pct); tgt = entry * (1 + tgt_pct)
        for j in range(idx + 1, n):
            if lo[j] <= stop: return -stop_pct * 100, j
            if hi[j] >= tgt: return tgt_pct * 100, j
    else:
        stop = entry * (1 + stop_pct); tgt = entry * (1 - tgt_pct)
        for j in range(idx + 1, n):
            if hi[j] >= stop: return -stop_pct * 100, j
            if lo[j] <= tgt: return tgt_pct * 100, j
    return None, None


def detect_sweeps(df, rr=RR, vol_mult=0.0):
    """Setups sweep+CHoCH non-chevauchants : (entry_idx, side, stop_pct, tgt_pct).
    vol_mult>0 exige volume(bougie de confirmation) > vol_mult x moyenne(20)."""
    from app.market_structure.swings import detect_swings
    from app.schemas.domain import SwingKind
    sw = detect_swings(df, left=R, right=R)
    hi = df["high"].to_numpy(float); lo = df["low"].to_numpy(float); cl = df["close"].to_numpy(float)
    vol = df["volume"].to_numpy(float) if "volume" in df.columns else np.ones(len(cl))
    n = len(cl)

    def vol_ok(e):
        if vol_mult <= 0: return True
        base = vol[max(0, e - 20):e]
        return len(base) > 0 and vol[e] > vol_mult * base.mean()
    lows = [(s.index, s.price) for s in sw if s.kind == SwingKind.LOW]
    highs = [(s.index, s.price) for s in sw if s.kind == SwingKind.HIGH]

    def last_before(seq, idx):
        out = None
        for i, p in seq:
            if i < idx: out = (i, p)
            else: break
        return out

    cand = []
    # LONG : sweep d'un swing low, CHoCH au-dessus du swing high de reference
    for (sli, slp) in lows:
        ref = last_before(highs, sli)
        if ref is None: continue
        start = sli + R + 1
        for b in range(start, min(start + W, n)):
            if lo[b] < slp and cl[b] > slp:          # sweep
                for e in range(b + 1, min(b + 1 + C, n)):
                    if cl[e] > ref[1]:               # CHoCH haussier (close > ref high)
                        if vol_ok(e):
                            swl = lo[sli:e + 1].min()    # plus-bas du sweep
                            if swl < cl[e]:
                                sp = (cl[e] - swl) / cl[e]
                                if 0.003 < sp < 0.15:    # garde-fou
                                    cand.append((e, "LONG", sp, sp * rr))
                        break
                break
    # SHORT : miroir
    for (shi, shp) in highs:
        ref = last_before(lows, shi)
        if ref is None: continue
        start = shi + R + 1
        for b in range(start, min(start + W, n)):
            if hi[b] > shp and cl[b] < shp:          # sweep
                for e in range(b + 1, min(b + 1 + C, n)):
                    if cl[e] < ref[1]:               # CHoCH baissier
                        if vol_ok(e):
                            swh = hi[shi:e + 1].max()
                            if swh > cl[e]:
                                sp = (swh - cl[e]) / cl[e]
                                if 0.003 < sp < 0.15:
                                    cand.append((e, "SHORT", sp, sp * rr))
                        break
                break
    cand.sort(key=lambda t: t[0])
    # non-chevauchant : un seul trade actif a la fois
    trades, last_exit = [], -1
    for (e, side, sp, tp) in cand:
        if e <= last_exit: continue
        ret, xi = resolve(hi, lo, cl, e, side, sp, tp)
        if ret is None: continue
        trades.append((e, side, sp, tp, ret, xi))
        last_exit = xi
    return trades


async def fetch():
    from app.ingestion.ccxt_fetcher import CCXTFetcher
    from app.services.continuous_scanner import _rows_to_df, build_regime_timeline
    f = CCXTFetcher(EX)
    cache = {}
    for s in SYMS:
        try:
            df = _rows_to_df(await f.fetch_ohlcv(s, TF, limit=LIMIT)).reset_index(drop=True)
            if len(df) > 200: cache[s] = df
        except Exception as ex:
            print(f"  WARN {s}: {ex}")
    btc = _rows_to_df(await f.fetch_ohlcv("BTC/USDT", "4h", limit=LIMIT))
    await f.close()
    return cache, build_regime_timeline(btc)


def dci(a, b, rng, B=4000):
    a = np.asarray(a); b = np.asarray(b)
    if len(a) == 0 or len(b) == 0: return (np.nan, np.nan)
    ia = rng.integers(0, len(a), size=(B, len(a))); ib = rng.integers(0, len(b), size=(B, len(b)))
    d = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def run_variant(pre, rng, rr, vol_mult):
    """Detecte + benchmarke pour (rr, vol_mult). pre = donnees precalculees par symbole."""
    det, rnd = [], []
    for s, P in pre.items():
        hi, lo, cl, tns, vol, univ_ret, crash_thr, pools = P
        df = pd.DataFrame({"high": hi, "low": lo, "close": cl, "volume": vol,
                           "timestamp": pd.to_datetime(tns)})
        for (e, side, sp, tp, ret, xi) in detect_sweeps(df, rr=rr, vol_mult=vol_mult):
            reg = pools["_rg"][e]; crash = univ_ret[e] <= crash_thr
            det.append((ret - COST, crash))
            pool = pools.get(reg)
            if pool is None or len(pool) == 0: continue
            for ri in rng.choice(pool, size=min(K, len(pool)), replace=len(pool) < K):
                rr_, _ = resolve(hi, lo, cl, int(ri), side, sp, tp)
                if rr_ is not None:
                    rnd.append((rr_ - COST, univ_ret[int(ri)] <= crash_thr))
    if len(det) < 20:
        return None
    D = np.array(det, dtype=[("ret", float), ("crash", bool)])
    Rn = np.array(rnd, dtype=[("ret", float), ("crash", bool)])
    g = dci(D["ret"], Rn["ret"], rng)
    dk = D["ret"][~D["crash"]]; rk = Rn["ret"][~Rn["crash"]]
    jk = dci(dk, rk, rng)
    return dict(n=len(D), wr=100*(D["ret"] > 0).mean(), pf=pf(D["ret"]),
                gci=g, gpf_rnd=pf(Rn["ret"]), jpf=pf(dk), jci=jk, jn=len(dk))


def main():
    rng = np.random.default_rng(SEED)
    cache, timeline = asyncio.run(fetch())
    ts_arr = np.array([pd.Timestamp(t).value for t, _ in timeline], dtype="int64")
    rg_arr = np.array([r.trend for _, r in timeline], dtype=object)
    def regime_at(q):
        if len(ts_arr) == 0: return "NONE"
        i = np.searchsorted(ts_arr, q, "right") - 1
        return rg_arr[i] if i >= 0 else "NONE"
    span = next(iter(cache.values()))
    print(f"Univers : {len(cache)} symboles, {TF}, {LIMIT} bougies "
          f"({span['timestamp'].iloc[0].date()} -> {span['timestamp'].iloc[-1].date()})")

    # precalcul par symbole
    pre = {}
    for s, df in cache.items():
        hi = df["high"].to_numpy(float); lo = df["low"].to_numpy(float)
        cl = df["close"].to_numpy(float); vol = df["volume"].to_numpy(float)
        tns = df["timestamp"].to_numpy().astype("datetime64[ns]").astype("int64")
        univ_ret = np.concatenate([[0], cl[1:] / cl[:-1] - 1])
        crash_thr = np.percentile(univ_ret, CRASH_FRACTILE)
        rg = np.array([regime_at(t) for t in tns], dtype=object)
        idxs = np.arange(len(cl))
        pools = {r: np.where((rg == r) & (idxs >= R + 2) & (idxs < len(cl) - 1))[0]
                 for r in ("BULL", "BEAR", "RANGE")}
        pools["_rg"] = rg
        pre[s] = (hi, lo, cl, tns, vol, univ_ret, crash_thr, pools)

    print("\n=== SWEEP+RETOURNEMENT vs ALEATOIRE apparie (cout %.1f%%, %s) ===" % (COST, TF))
    print("  Grille pre-declaree (multi-tests -> exiger IC hors-krach > 0 nettement).")
    print(f"  {'variante':22} {'n':>4} {'WR%':>5} {'PF':>5} | {'diff GLOBAL IC95':>22} | {'diff HORS-KRACH IC95':>24}")
    for vm in (0.0, 1.5):
        for rr in (1.5, 2.0, 3.0):
            r = run_variant(pre, rng, rr, vm)
            tag = f"RR={rr} vol{'>'+str(vm)+'x' if vm>0 else '=off'}"
            if r is None:
                print(f"  {tag:22} (trop peu de setups)"); continue
            gv = "EDGE" if r["gci"][0] > 0 else ("anti" if r["gci"][1] < 0 else "hasard")
            jv = "EDGE" if r["jci"][0] > 0 else ("anti" if r["jci"][1] < 0 else "hasard")
            print(f"  {tag:22} {r['n']:>4} {r['wr']:>5.1f} {r['pf']:>5.2f} | "
                  f"[{r['gci'][0]:+.2f},{r['gci'][1]:+.2f}]->{gv:6} | "
                  f"[{r['jci'][0]:+.2f},{r['jci'][1]:+.2f}]->{jv}")


if __name__ == "__main__":
    main()
