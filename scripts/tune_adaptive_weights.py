"""Tune les poids des features adaptatives DEPUIS LES DONNEES (pas a la main).

Methode :
  1. Rejoue le backtest fidele (adaptatif OFF) pour produire un jeu de trades reels.
  2. Pour chaque trade, reconstruit les features A L'ENTREE (slice OHLCV jusqu'a la
     bougie d'entree -> aucun look-ahead) + l'issue (gagnant / perdant).
  3. Mesure, par feature : moyenne chez les gagnants vs perdants, correlation
     point-bisériale avec l'issue, et winrate top-tercile vs bottom-tercile.
  4. Propose des poids proportionnels au pouvoir predictif (|correlation|).

Sortie = poids data-driven a reporter dans app.strategy.adaptive (remplacant les priors).

Usage :
    python scripts/tune_adaptive_weights.py [--days 14] [--tf 15m] [--symbols ...]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_TMP_DB = Path(tempfile.gettempdir()) / "analyseur_tune_weights.db"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TMP_DB.as_posix()}"

_FEATURE_NAMES = [
    "momentum", "volume_conviction", "htf_alignment",
    "structure_health", "volatility_state", "mfe_progress",
]


def _bars_for(days: int, tf: str) -> int:
    per_day = {"1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}.get(tf, 96)
    return days * per_day


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--tf", default="15m")
    ap.add_argument("--symbols",
                    default="BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,ADA/USDT,"
                            "AVAX/USDT,DOGE/USDT,LINK/USDT,DOT/USDT,LTC/USDT,ATOM/USDT")
    ap.add_argument("--reuse", action="store_true",
                    help="Reutilise la DB temp deja peuplee (skip le backfill).")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    from app.config import settings
    from app.db.models import Base
    from app.db.session import engine as app_engine
    from app.ingestion.ccxt_fetcher import CCXTFetcher
    from app.schemas.domain import Side
    from app.services.continuous_scanner import (
        ContinuousScanner, ScanPlan, build_regime_timeline, _rows_to_df,
    )
    from app.strategy.adaptive import compute_evaluation_features

    assert "tune_weights" in settings.database_url
    if not args.reuse and _TMP_DB.exists():
        _TMP_DB.unlink()
    async with app_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    history_bars = _bars_for(args.days, args.tf)

    fetcher = CCXTFetcher(settings.exchange_id)
    # 1. Backtest fidele (adaptatif OFF) pour produire des trades (skip si --reuse)
    if not args.reuse:
        btc_df = _rows_to_df(await fetcher.fetch_ohlcv("BTC/USDT", "1h", limit=600))
        timeline = build_regime_timeline(btc_df)
        print(f"Collecte : {len(symbols)} symboles x {args.tf}, {args.days}j ...")
        plan = ScanPlan(symbols=symbols, timeframes=[args.tf], candles_per_fetch=history_bars + 60)
        scanner = ContinuousScanner(plan=plan)  # engine live par defaut (adaptatif OFF)
        await scanner.backfill(history_bars=history_bars, bars_per_step=1,
                               symbols=symbols, timeframes=[args.tf], regime_timeline=timeline)
        await scanner.stop()
    else:
        print(f"--reuse : reutilise {_TMP_DB.name} (skip backfill)")

    # 2. Cache OHLCV par symbole pour reconstruire les features a l'entree
    ohlcv_cache: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        rows = await fetcher.fetch_ohlcv(sym, args.tf, limit=history_bars + 60)
        ohlcv_cache[sym] = _rows_to_df(rows)
    await fetcher.close()

    # 3. Pour chaque trade : features a l'entree + issue
    con = sqlite3.connect(str(_TMP_DB))
    cur = con.cursor()
    cur.execute("""
        SELECT s.base||'/'||s.quote, ut.side, ut.entry_timestamp, ut.pct_gain
        FROM unit_trades ut JOIN symbols s ON s.id = ut.symbol_id
    """)
    rows = cur.fetchall()
    con.close()
    print(f"Trades collectes : {len(rows)}")
    if len(rows) < 15:
        print("Trop peu de trades pour tuner des poids fiables. Augmente --days/--symbols.")
        return 0

    from collections import defaultdict
    per_trade: list[tuple[dict, int]] = []
    for sym, side, entry_ts, pct in rows:
        df = ohlcv_cache.get(sym)
        if df is None or df.empty:
            continue
        ts = pd.Timestamp(entry_ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        tss = pd.to_datetime(df["timestamp"], utc=True)
        matches = tss[tss <= ts]
        if matches.empty:
            continue
        entry_idx = matches.index[-1]
        if entry_idx < 25:
            continue
        sl = df.iloc[: entry_idx + 1]
        direction = 1 if side == "LONG" else -1
        entry_price = float(sl["close"].iloc[-1])
        feats = compute_evaluation_features(
            sl["high"].to_numpy(float), sl["low"].to_numpy(float),
            sl["close"].to_numpy(float), sl["volume"].to_numpy(float),
            sl["open"].to_numpy(float),
            direction=direction, entry=entry_price,
        )
        if not feats:
            continue
        per_trade.append((feats, 1 if (pct or 0) > 0 else 0))

    if len(per_trade) < 15:
        print(f"Trop peu de trades exploitables ({len(per_trade)}).")
        return 0

    # Garde uniquement les features presentes pour TOUS les trades (alignement)
    common = set(per_trade[0][0].keys())
    for feats, _ in per_trade:
        common &= set(feats.keys())
    common = sorted(common)

    y = np.array([o for _, o in per_trade], dtype=float)
    n = len(y)
    wr = 100 * y.mean()
    print(f"\nDataset : {n} trades, winrate global {wr:.1f}%")
    print(f"Variables candidates mesurees : {len(common)}")
    print("=" * 78)
    print(f"{'variable':<24} {'|corr|':>7} {'corr':>7} {'WR top1/3':>10} {'WR bot1/3':>10} {'spread':>7}")
    print("-" * 78)

    ranked = []
    for k in common:
        x = np.array([feats[k] for feats, _ in per_trade], dtype=float)
        corr = 0.0 if x.std() < 1e-9 else float(np.corrcoef(x, y)[0, 1])
        order = np.argsort(x)
        t = max(1, n // 3)
        wr_top = 100 * y[order[-t:]].mean()
        wr_bot = 100 * y[order[:t]].mean()
        ranked.append((abs(corr), corr, k, wr_top, wr_bot, wr_top - wr_bot))

    ranked.sort(reverse=True)
    for acorr, corr, k, wr_top, wr_bot, spread in ranked:
        flag = " <<<" if acorr >= 0.20 else ""
        print(f"{k:<24} {acorr:>7.3f} {corr:>+7.3f} {wr_top:>9.1f}% {wr_bot:>9.1f}% {spread:>+6.1f}{flag}")

    print("\n" + "=" * 78)
    print("LECTURE :")
    print("  - |corr| eleve = variable qui DEPLACE le 50/50 (signal exploitable).")
    print("  - spread = WR(top tercile) - WR(bot tercile) : >+15 ou <-15 = discriminant.")
    print("  - corr NEGATIVE = la variable predit l'INVERSE (utile aussi, signe a inverser).")
    print(f"  - {n} trades : indicatif. Confirmer sur 200+ trades multi-regime avant de promouvoir.")
    print("  - Variables marquees <<< = candidates a integrer dans le modele de conviction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
