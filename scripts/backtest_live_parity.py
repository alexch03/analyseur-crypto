"""Backtest FIDELE AU LIVE : rejoue exactement le scanner live (memes 10 detecteurs,
meme HypothesisEngine avec reject_if_regime_unknown=True, meme regime rejoue bar-par-bar)
sur N jours d'historique.

Contrairement a run_loop.py (6 detecteurs, pas de regime) et engine_replay.py (setups SMC
que le live ne trade pas), CE backtest produit exactement les memes trades que le live aurait
produits sur la periode -> mesure honnete de l'effet des corrections.

Usage :
    python scripts/backtest_live_parity.py [--days 7] [--tf 15m] [--symbols BTC/USDT,ETH/USDT,...]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ── DB TEMPORAIRE : doit etre defini AVANT tout import d'app (engine bind au load) ──
_TMP_DB = Path(tempfile.gettempdir()) / "analyseur_parity_backtest.db"
if _TMP_DB.exists():
    _TMP_DB.unlink()
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TMP_DB.as_posix()}"


def _bars_for(days: int, tf: str) -> int:
    per_day = {"1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}.get(tf, 96)
    return days * per_day


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--tf", default="15m")
    ap.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,ADA/USDT")
    args = ap.parse_args()

    from app.config import settings
    print(f"DB backtest : {settings.database_url}")
    assert "parity_backtest" in settings.database_url, "DATABASE_URL non redirige !"

    from sqlalchemy.ext.asyncio import create_async_engine
    from app.db.models import Base
    from app.db.session import async_session_factory
    from app.services.continuous_scanner import (
        ContinuousScanner, ScanPlan, build_regime_timeline, _rows_to_df, default_detectors,
    )
    from app.ingestion.ccxt_fetcher import CCXTFetcher

    # 1. Cree le schema
    from app.db.session import engine as app_engine
    async with app_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    history_bars = _bars_for(args.days, args.tf)
    print(f"Backtest FIDELE LIVE : {len(symbols)} symboles x {args.tf}, {args.days}j "
          f"({history_bars} bougies/paire)")
    print(f"Detecteurs : {len(default_detectors())} (identique au live)")

    # 2. Timeline de regime (BTC 1h) — rejoue le regime comme en live
    fetcher = CCXTFetcher(settings.exchange_id)
    print("Fetch BTC 1h pour timeline de regime ...")
    btc_rows = await fetcher.fetch_ohlcv("BTC/USDT", "1h", limit=500)
    btc_df = _rows_to_df(btc_rows)
    await fetcher.close()
    timeline = build_regime_timeline(btc_df)
    print(f"Timeline regime : {len(timeline)} points")
    if timeline:
        from collections import Counter
        regimes = Counter(r.trend for _, r in timeline)
        print(f"  Distribution : {dict(regimes)}")

    # 3. Scanner FIDELE : plan par defaut (10 detecteurs) + engine live (reject_if_regime_unknown)
    plan = ScanPlan(symbols=symbols, timeframes=[args.tf], candles_per_fetch=history_bars + 60)
    scanner = ContinuousScanner(plan=plan)  # engine = config live exacte (cf __init__)

    print("Replay bar-par-bar (peut prendre quelques minutes) ...")
    result = await scanner.backfill(
        history_bars=history_bars, bars_per_step=1,
        symbols=symbols, timeframes=[args.tf],
        regime_timeline=timeline,
    )
    await scanner.stop()
    print(f"Backfill termine : {result['total_steps']} steps, "
          f"{result['total_patterns_detected']} patterns detectes, "
          f"{result['elapsed_seconds']}s")

    # 4. Stats sur les unit_trades produits
    import sqlite3
    con = sqlite3.connect(str(_TMP_DB))
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM unit_trades")
    total = cur.fetchone()[0]
    print()
    print("=" * 70)
    print(f"RESULTAT BACKTEST FIDELE LIVE ({args.days}j, {args.tf})")
    print("=" * 70)
    if total == 0:
        print("  0 trade clos. (Filtres regime + criteres durcis tres selectifs,")
        print("   ou periode sans setup valide.)")
        con.close()
        return 0

    cur.execute("""
        SELECT pattern_kind, COUNT(*),
               SUM(CASE WHEN pct_gain>0 THEN 1 ELSE 0 END),
               AVG(pct_gain), SUM(pct_gain)
        FROM unit_trades GROUP BY pattern_kind ORDER BY 2 DESC
    """)
    print(f"  {'pattern':<26} {'N':>4} {'win%':>6} {'avg':>7} {'sum':>9}")
    for pat, n, w, avg, s in cur.fetchall():
        wr = 100 * w / n if n else 0
        print(f"  {pat:<26} {n:>4} {wr:>5.1f}% {avg:>+6.2f}% {s:>+8.2f}%")

    cur.execute("SELECT pct_gain FROM unit_trades ORDER BY entry_timestamp")
    gains = [r[0] or 0.0 for r in cur.fetchall()]
    wins = sum(1 for g in gains if g > 0)
    compound = 1.0
    for g in gains:
        compound *= 1.0 + g / 100.0
    compound_pct = (compound - 1.0) * 100.0
    print()
    print(f"  TOTAL : {total} trades")
    print(f"  Winrate global : {100*wins/total:.1f}%")
    print(f"  PnL moyen/trade : {sum(gains)/total:+.3f}%")
    print(f"  Somme PnL (additive) : {sum(gains):+.2f}%")
    print(f"  Compound (1u/trade)  : {compound_pct:+.2f}%")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
