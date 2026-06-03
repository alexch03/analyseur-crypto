"""Phase 3bis DOE : MOMENTUM CROSS-SECTIONNEL (force relative) — test rigoureux.

Strategie : a chaque rebalancement, classer l'univers par rendement passe (lookback L),
longer le top-k / shorter le bottom-k, tenir H jours. Rebalancer.

Benchmarks OBLIGATOIRES (un edge doit battre les DEUX) :
  - BUY&HOLD equal-weight (le beta du marche)
  - portefeuilles ALEATOIRES (k symboles tires au hasard a chaque rebal)

Rigueur : IC bootstrap sur le rendement moyen par periode, jackknife des semaines de
krach, split OOS (1ere moitie / 2e moitie). Couts appliques par rebalancement.

Usage : PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/momentum_test.py
"""
from __future__ import annotations
import sys, asyncio
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SYMS = ["BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT","ADA/USDT","AVAX/USDT",
        "DOGE/USDT","LINK/USDT","DOT/USDT","LTC/USDT","ATOM/USDT","TRX/USDT","UNI/USDT",
        "NEAR/USDT","ICP/USDT","ETC/USDT","XLM/USDT","BCH/USDT","FIL/USDT","APT/USDT",
        "ARB/USDT","OP/USDT","INJ/USDT","AAVE/USDT"]
COST = 0.2     # cout en % par rebalancement par jambe (long-only=1 jambe, L/S=2)
SEED = 42
N_RAND = 400   # tirages aleatoires pour le benchmark random


RESEARCH_EXCHANGE = "binance"  # Binance sert des ANNEES d'historique (Bitget plafonne ~200 barres en profond)


async def fetch_close(symbols, tf="1d", limit=1500):
    from app.ingestion.ccxt_fetcher import CCXTFetcher
    from app.services.continuous_scanner import _rows_to_df
    f = CCXTFetcher(RESEARCH_EXCHANGE)
    series = {}
    for s in symbols:
        try:
            df = _rows_to_df(await f.fetch_ohlcv(s, tf, limit=limit))
            if len(df) > 60:
                series[s] = df.set_index("timestamp")["close"]
        except Exception as e:
            print(f"  WARN {s}: {e}")
    await f.close()
    px = pd.DataFrame(series).sort_index()
    px = px.dropna(axis=1, thresh=int(0.9 * len(px)))   # garde symboles ~complets
    px = px.ffill().dropna()
    return px


def momentum_series(px, lookback, hold, k, long_short, rng):
    """Retourne (dates, port_rets, bh_rets, rand_rets) alignes sur les rebalancements."""
    P = px.to_numpy(float)
    cols = px.columns.to_list()
    n, m = P.shape
    dates, port, bh, rnd = [], [], [], []
    legs = 2 if long_short else 1
    for t in range(lookback, n - hold, hold):
        past = P[t] / P[t - lookback] - 1.0
        fwd = P[t + hold] / P[t] - 1.0
        order = np.argsort(past)
        longs = order[-k:]; shorts = order[:k]
        if long_short:
            r = fwd[longs].mean() - fwd[shorts].mean()
        else:
            r = fwd[longs].mean()
        r -= COST / 100.0 * legs
        port.append(r); bh.append(fwd.mean())
        # random : k longs (et k shorts si L/S) au hasard
        rr = []
        for _ in range(N_RAND):
            idx = rng.permutation(m)
            rl = fwd[idx[:k]].mean()
            rv = rl - fwd[idx[k:2*k]].mean() if long_short else rl
            rr.append(rv - COST/100.0*legs)
        rnd.append(np.mean(rr))
        dates.append(px.index[t])
    return np.array(dates), np.array(port), np.array(bh), np.array(rnd)


def boot_ci(x, rng, B=4000):
    if len(x) == 0: return (np.nan, np.nan)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    mu = x[idx].mean(axis=1)
    return np.percentile(mu, 2.5), np.percentile(mu, 97.5)


def diff_ci(a, b, rng, B=4000):
    idx = rng.integers(0, len(a), size=(B, len(a)))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def sharpe(x, per_year):
    return float(x.mean() / x.std() * np.sqrt(per_year)) if x.std() > 0 else 0.0


def main():
    rng = np.random.default_rng(SEED)
    print("Fetch 1d close...")
    px = asyncio.run(fetch_close(SYMS))
    print(f"Univers : {px.shape[1]} symboles x {px.shape[0]} jours "
          f"({px.index[0].date()} -> {px.index[-1].date()})")
    per_year = 365 / 7  # rebal hebdo
    crash = px.pct_change().mean(axis=1)  # rendement moyen univers / jour

    print("\n=== MOMENTUM CROSS-SECTIONNEL (rebal hebdo, k=5, cout %.1f%%/jambe) ===" % COST)
    print("  Un edge doit battre buy&hold ET random (IC95 de la diff > 0).\n")
    for L in (14, 30, 60):
        for ls in (False, True):
            d, port, bh, rndm = momentum_series(px, L, 7, 5, ls, rng)
            if len(port) < 10:
                continue
            tag = f"L={L:>2} {'L/S' if ls else 'long-only'}"
            dvb = diff_ci(port, bh, rng); dvr = diff_ci(port, rndm, rng)
            vb = "OUI" if dvb[0] > 0 else "non"
            vr = "OUI" if dvr[0] > 0 else "non"
            print(f"  {tag:18} n={len(port):3} | mean/sem={port.mean()*100:+.2f}% "
                  f"Sharpe={sharpe(port,per_year):+.2f} totPnL={ (np.prod(1+port)-1)*100:+.0f}% "
                  f"| vs B&H {dvb[0]*100:+.2f}..{dvb[1]*100:+.2f}%->{vb} "
                  f"| vs RND {dvr[0]*100:+.2f}..{dvr[1]*100:+.2f}%->{vr}")

    # Focus config canonique L=30 L/S : robustesse
    print("\n=== ROBUSTESSE : L=30, L/S, k=5 ===")
    d, port, bh, rndm = momentum_series(px, 30, 7, 5, True, rng)
    print(f"  B&H (beta)  : mean/sem={bh.mean()*100:+.2f}% Sharpe={sharpe(bh,per_year):+.2f} "
          f"totPnL={(np.prod(1+bh)-1)*100:+.0f}%")
    print(f"  Momentum    : mean/sem={port.mean()*100:+.2f}% Sharpe={sharpe(port,per_year):+.2f} "
          f"totPnL={(np.prod(1+port)-1)*100:+.0f}%")
    print(f"  Random      : mean/sem={rndm.mean()*100:+.2f}%")
    # jackknife : retirer les 10% pires semaines de marche (krachs)
    thr = np.percentile(bh, 10)
    keep = bh > thr
    dvb = diff_ci(port[keep], bh[keep], rng)
    print(f"  Hors-krach (retire {int((~keep).sum())} pires semaines) : "
          f"momentum vs B&H IC95=[{dvb[0]*100:+.2f},{dvb[1]*100:+.2f}]% -> "
          f"{'EDGE' if dvb[0]>0 else 'pas d edge'}")
    # OOS : 2 moities
    h = len(port)//2
    for name, sl in [("1ere moitie", slice(0,h)), ("2e moitie (OOS)", slice(h,None))]:
        p, b = port[sl], bh[sl]
        dc = diff_ci(p, b, rng)
        print(f"  {name:16}: mom mean={p.mean()*100:+.2f}% B&H={b.mean()*100:+.2f}% "
              f"diff IC95=[{dc[0]*100:+.2f},{dc[1]*100:+.2f}]% -> {'EDGE' if dc[0]>0 else 'pas d edge'}")


if __name__ == "__main__":
    main()
