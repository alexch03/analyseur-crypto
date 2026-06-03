"""Phase 3ter : MOMENTUM TEMPOREL (trend-following / TSMOM) — test rigoureux.

Pour chaque actif : etre LONG quand il est en tendance haussiere (rendement sur L jours > 0,
connu a t-1), sinon CASH (0). Equal-weight sur l'univers. La valeur du trend-following
est dans la REDUCTION du drawdown (on sort des bears), pas le rendement brut.

Benchmark : buy&hold equal-weight. Metriques : rendement total, Sharpe annualise,
max drawdown, + IC bootstrap sur la diff de rendement moyen, + split OOS.
Donnees : Binance daily (4 ans, multi-cycles).

Usage : PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/trend_test.py
"""
from __future__ import annotations
import sys, asyncio
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
EX = "binance"
COST = 0.1  # cout en % par changement de position (entree ou sortie)
SEED = 42
SYMS = ["BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT","ADA/USDT","AVAX/USDT",
        "DOGE/USDT","LINK/USDT","DOT/USDT","LTC/USDT","ATOM/USDT","TRX/USDT","UNI/USDT",
        "ICP/USDT","ETC/USDT","XLM/USDT","BCH/USDT","FIL/USDT","AAVE/USDT","NEAR/USDT",
        "XMR/USDT","ALGO/USDT"]


async def fetch_close(symbols, tf="1d", limit=1500):
    from app.ingestion.ccxt_fetcher import CCXTFetcher
    from app.services.continuous_scanner import _rows_to_df
    f = CCXTFetcher(EX)
    ser = {}
    for s in symbols:
        try:
            df = _rows_to_df(await f.fetch_ohlcv(s, tf, limit=limit))
            if len(df) > 200:
                ser[s] = df.set_index("timestamp")["close"]
        except Exception as e:
            print(f"  WARN {s}: {e}")
    await f.close()
    px = pd.DataFrame(ser).sort_index()
    px = px.dropna(axis=1, thresh=int(0.9 * len(px))).ffill().dropna()
    return px


def maxdd(equity):
    peak = np.maximum.accumulate(equity)
    return float(((equity - peak) / peak).min() * 100)


def metrics(daily_ret):
    daily_ret = daily_ret[~np.isnan(daily_ret)]
    eq = np.cumprod(1 + daily_ret)
    tot = (eq[-1] - 1) * 100
    sh = daily_ret.mean() / daily_ret.std() * np.sqrt(365) if daily_ret.std() > 0 else 0.0
    return tot, sh, maxdd(eq)


def tsmom_portfolio(px, L):
    """Rendement journalier equal-weight d'une strategie long/cash par actif (signal=tendance L jours)."""
    P = px.to_numpy(float)
    n, m = P.shape
    rets = P[1:] / P[:-1] - 1.0                      # (n-1, m)
    strat = np.full_like(rets, np.nan)
    for j in range(m):
        for t in range(1, len(rets)):
            # signal connu a la cloture t (utilise pour le rendement t->t+1, donc decale)
            pass
    # vectorise : signal_t = 1 si P[t-1] > P[t-1-L] (tendance haussiere connue a t-1)
    sig = np.zeros((n, m))
    for t in range(L + 1, n):
        sig[t] = (P[t - 1] > P[t - 1 - L]).astype(float)
    pos = sig[1:]                                    # position appliquee au rendement du jour
    strat = pos * rets
    # cout sur changement de position
    chg = np.abs(np.diff(np.vstack([np.zeros((1, m)), pos]), axis=0))
    strat = strat - chg * (COST / 100.0)
    port = np.nanmean(strat, axis=1)
    bh = np.nanmean(rets, axis=1)
    return port, bh


def boot_diff_ci(a, b, rng, B=4000):
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    nmin = min(len(a), len(b))
    ia = rng.integers(0, len(a), size=(B, nmin)); ib = rng.integers(0, len(b), size=(B, nmin))
    d = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return np.percentile(d, 2.5) * 100, np.percentile(d, 97.5) * 100


def main():
    rng = np.random.default_rng(SEED)
    print("Fetch 1d close (Binance)...")
    px = asyncio.run(fetch_close(SYMS))
    print(f"Univers : {px.shape[1]} symboles x {px.shape[0]} jours "
          f"({px.index[0].date()} -> {px.index[-1].date()})\n")

    _, bh = tsmom_portfolio(px, 50)
    tb, sb, db = metrics(bh)
    print(f"BUY&HOLD equal-weight : totPnL={tb:+.0f}%  Sharpe={sb:+.2f}  maxDD={db:.0f}%\n")

    print("TREND-FOLLOWING (long/cash, signal=tendance L jours) :")
    for L in (50, 100, 150, 200):
        port, _ = tsmom_portfolio(px, L)
        tp, sp, dp = metrics(port)
        lo, hi = boot_diff_ci(port, bh, rng)
        verdict = ("MEILLEUR Sharpe & DD" if (sp > sb and dp > db) else
                   "DD reduit" if dp > db else "pas mieux")
        print(f"  L={L:>3} : totPnL={tp:+5.0f}%  Sharpe={sp:+.2f}  maxDD={dp:5.0f}%  "
              f"| diff rdt/j vs B&H IC95=[{lo:+.3f},{hi:+.3f}]%  -> {verdict}")

    # OOS sur L=100
    print("\nOOS (L=100, 2 moities) :")
    port, bh = tsmom_portfolio(px, 100)
    h = len(port) // 2
    for name, sl in [("1ere moitie", slice(0, h)), ("2e moitie (OOS)", slice(h, None))]:
        tp, sp, dp = metrics(port[sl]); tb2, sb2, db2 = metrics(bh[sl])
        print(f"  {name:16}: TF Sharpe={sp:+.2f} DD={dp:5.0f}%  |  B&H Sharpe={sb2:+.2f} DD={db2:5.0f}%")


if __name__ == "__main__":
    main()
