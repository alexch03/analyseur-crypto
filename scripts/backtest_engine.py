"""Moteur de backtest VALIDE et fiable (recherche).

Objectif : un backtest auquel on peut SE FIER. Trois piliers :
  1. BONNES DONNEES  : Binance (profond, gratuit), avec rapport qualite (trous, NaN, OHLC).
  2. MECANIQUE CORRECTE : fill au close de la bougie de signal, resolution forward
     stop-first conservateur (si stop ET cible touches dans la meme bougie -> stop),
     couts round-trip, AUCUN look-ahead (on ne lit que des bougies > entree).
  3. CONFIANCE : auto-tests (controles positif/negatif qui DOIVENT passer) + sur chaque
     strategie : benchmark vs ALEATOIRE apparie, split OOS, IC bootstrap.

Brancher une strategie = fournir signal(df) -> list[(entry_idx, side)]. Le moteur
dimensionne stop/cible en ATR, resout, et produit un RAPPORT DE CONFIANCE.

Usage : PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/backtest_engine.py
"""
from __future__ import annotations
import sys, asyncio
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EXCHANGE = "binance"           # source profonde et propre pour la recherche
COST = 0.2                     # cout round-trip en %
SEED = 42
_rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------------------- data
async def fetch_clean(symbols, tf, limit):
    """Telecharge + VALIDE l'OHLCV. Retourne (cache, rapport_qualite)."""
    from app.ingestion.ccxt_fetcher import CCXTFetcher
    from app.services.continuous_scanner import _rows_to_df
    try:
        from app.ingestion.data_quality import validate_ohlcv_df
    except Exception:
        validate_ohlcv_df = None
    f = CCXTFetcher(EXCHANGE)
    cache, report = {}, []
    bar_sec = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}.get(tf, 3600)
    for s in symbols:
        try:
            df = _rows_to_df(await f.fetch_ohlcv(s, tf, limit=limit)).reset_index(drop=True)
        except Exception as e:
            report.append((s, "FETCH_FAIL", str(e)[:40])); continue
        if len(df) < 200:
            report.append((s, "TROP_COURT", len(df))); continue
        ts = pd.to_datetime(df["timestamp"])
        gaps = int(((ts.diff().dt.total_seconds().round() != bar_sec).sum()) - 1)  # -1 pour le 1er NaN
        nan = int(df[["open", "high", "low", "close"]].isna().sum().sum())
        incoh = int(((df.high < df.low) | (df.high < df.close) | (df.low > df.close)
                     | (df.high < df.open) | (df.low > df.open)).sum())
        report.append((s, len(df), f"{ts.iloc[0].date()}->{ts.iloc[-1].date()}", gaps, nan, incoh))
        cache[s] = df
    await f.close()
    return cache, report


# ----------------------------------------------------------------------- mecanique
def _atr(h, l, c, n=14):
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    a = pd.Series(tr).rolling(n).mean().to_numpy()
    return np.concatenate([[np.nan], a])


def resolve(hi, lo, cl, idx, side, stop_pct, tgt_pct):
    """(rendement %, exit_idx). Stop-first conservateur. None si non resolu.
    N'utilise QUE des bougies > idx -> aucun look-ahead."""
    e = cl[idx]; n = len(cl)
    if side == "LONG":
        st, tg = e * (1 - stop_pct), e * (1 + tgt_pct)
        for j in range(idx + 1, n):
            if lo[j] <= st: return -stop_pct * 100, j
            if hi[j] >= tg: return tgt_pct * 100, j
    else:
        st, tg = e * (1 + stop_pct), e * (1 - tgt_pct)
        for j in range(idx + 1, n):
            if hi[j] >= st: return -stop_pct * 100, j
            if lo[j] <= tg: return tgt_pct * 100, j
    return None, None


def self_test(cache):
    """Controles : un setup qui DOIT gagner (TP1%/SL50%) -> WR~tres haut ;
    un qui DOIT perdre (TP50%/SL1%) -> WR~tres bas. Sinon le moteur n'est pas fiable."""
    def rnd_wr(sp, tp):
        out = []
        for s, df in cache.items():
            cl = df.close.to_numpy(float); hi = df.high.to_numpy(float); lo = df.low.to_numpy(float)
            for ri in _rng.integers(20, len(cl) - 1, size=200):
                r, _ = resolve(hi, lo, cl, int(ri), "LONG", sp, tp)
                if r is not None: out.append(r)
        return 100 * (np.array(out) > 0).mean()
    wr_easy = rnd_wr(0.50, 0.01)   # TP tres facile
    wr_hard = rnd_wr(0.01, 0.50)   # SL tres facile
    ok = wr_easy > 90 and wr_hard < 10
    return ok, wr_easy, wr_hard


# ----------------------------------------------------------------------- strategies
def sig_bollinger(df):
    cl = df.close.to_numpy(float)
    m = pd.Series(cl).rolling(20).mean().to_numpy(); sd = pd.Series(cl).rolling(20).std().to_numpy()
    lo, up = m - 2 * sd, m + 2 * sd; out = []
    for i in range(21, len(cl) - 1):
        if cl[i] < lo[i] and cl[i - 1] >= lo[i - 1]: out.append((i, "LONG"))
        elif cl[i] > up[i] and cl[i - 1] <= up[i - 1]: out.append((i, "SHORT"))
    return out


def sig_trend_breakout(df):
    cl = df.close.to_numpy(float); hi = df.high.to_numpy(float)
    ema = pd.Series(cl).ewm(span=50, adjust=False).mean().to_numpy()
    hh = pd.Series(hi).rolling(20).max().shift(1).to_numpy(); out = []
    for i in range(51, len(cl) - 1):
        if cl[i] > ema[i] and cl[i] > hh[i]: out.append((i, "LONG"))
    return out


STRATEGIES = {"bollinger_mr": sig_bollinger, "trend_breakout": sig_trend_breakout}


# ----------------------------------------------------------------------- backtest
def backtest(cache, signal_fn, atr_stop=1.5, rr=2.0):
    """Trades non-chevauchants + controles aleatoires apparies (meme side/stop%/cible%)."""
    det, rnd = [], []
    for s, df in cache.items():
        cl = df.close.to_numpy(float); hi = df.high.to_numpy(float); lo = df.low.to_numpy(float)
        a = _atr(hi, lo, cl); sigs = dict(signal_fn(df)); last = -1
        for i in sorted(sigs):
            if i <= last or i >= len(cl) - 1 or np.isnan(a[i]) or a[i] <= 0: continue
            sp = atr_stop * a[i] / cl[i]; tp = sp * rr
            if not (0.003 < sp < 0.15): continue
            side = sigs[i]
            r, xi = resolve(hi, lo, cl, i, side, sp, tp)
            if r is None: continue
            det.append(r - COST); last = xi
            for ri in _rng.integers(20, len(cl) - 1, size=5):
                rr_, _ = resolve(hi, lo, cl, int(ri), side, sp, tp)
                if rr_ is not None: rnd.append(rr_ - COST)
    return np.array(det), np.array(rnd)


def _pf(x):
    x = np.asarray(x); g = x[x > 0].sum(); l = x[x < 0].sum()
    return float("inf") if l == 0 else abs(g / l)


def _ci(x, B=4000):
    x = np.asarray(x)
    if len(x) < 10: return (np.nan, np.nan)
    m = x[_rng.integers(0, len(x), (B, len(x)))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def trust_report(name, det, rnd):
    if len(det) < 20:
        print(f"  [{name}] n={len(det)} insuffisant pour conclure"); return
    lo, hi = _ci(det)
    # diff vs aleatoire
    ia = _rng.integers(0, len(det), (4000, len(det))); ib = _rng.integers(0, len(rnd), (4000, len(rnd)))
    dlo, dhi = np.percentile(det[ia].mean(1) - rnd[ib].mean(1), [2.5, 97.5])
    h = len(det) // 2
    verdict = "EDGE (bat l'aleatoire)" if dlo > 0 else "PAS d'edge (= aleatoire)"
    print(f"  [{name}] n={len(det)} WR={100*(det>0).mean():.1f}% E[R]={det.mean():+.3f}% PF={_pf(det):.2f}")
    print(f"     IC95 E[R]=[{lo:+.3f},{hi:+.3f}] | vs aleatoire diff IC95=[{dlo:+.3f},{dhi:+.3f}] -> {verdict}")
    print(f"     OOS : 1ere moitie PF={_pf(det[:h]):.2f} | 2e moitie PF={_pf(det[h:]):.2f}")


SYMS = ["BTC/USDT","ETH/USDT","BNB/USDT","XRP/USDT","ADA/USDT","SOL/USDT","DOGE/USDT",
        "LTC/USDT","LINK/USDT","DOT/USDT","ATOM/USDT","XLM/USDT"]


def main():
    TF, LIMIT = "4h", 3000
    print("=" * 70)
    print(f"MOTEUR DE BACKTEST VALIDE | source={EXCHANGE} TF={TF} cout={COST}%")
    print("=" * 70)
    cache, report = asyncio.run(fetch_clean(SYMS, TF, LIMIT))

    print("\n--- 1) QUALITE DES DONNEES (trous / NaN / incoherences OHLC) ---")
    tot_bad = 0
    for row in report:
        if len(row) == 6:
            s, n, span, gaps, nan, incoh = row
            tot_bad += gaps + nan + incoh
            flag = "OK" if (gaps + nan + incoh) == 0 else "<-- ANOMALIE"
            print(f"   {s:10} {n:5} bougies {span} | trous={gaps} NaN={nan} incoh={incoh} {flag}")
        else:
            print(f"   {row[0]:10} {row[1]} {row[2]}")
    print(f"   => Donnees {'PROPRES (fiables)' if tot_bad == 0 else 'AVEC ANOMALIES'} sur {len(cache)} symboles")

    print("\n--- 2) AUTO-VALIDATION DU MOTEUR (controles) ---")
    ok, we, wh = self_test(cache)
    print(f"   Setup gagnant force (SL50/TP1) : WR={we:.1f}% (attendu >90%)")
    print(f"   Setup perdant force (SL1/TP50) : WR={wh:.1f}% (attendu <10%)")
    print(f"   Look-ahead : impossible par construction (resolution sur bougies > entree uniquement)")
    print(f"   => MOTEUR {'FIABLE (controles OK)' if ok else 'SUSPECT (controles ECHOUES)'}")

    print("\n--- 3) RAPPORT DE CONFIANCE PAR STRATEGIE ---")
    for name, fn in STRATEGIES.items():
        det, rnd = backtest(cache, fn)
        trust_report(name, det, rnd)

    print("\n" + "=" * 70)
    print("VERDICT : on peut se fier au moteur ssi (1) donnees propres ET (2) controles OK.")
    print("Une strategie n'a un edge que si 'vs aleatoire' = EDGE ET OOS coherent (2 moities >1).")
    print("=" * 70)


if __name__ == "__main__":
    main()
