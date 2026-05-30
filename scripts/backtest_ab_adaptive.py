"""A/B backtest : couche adaptative OFF vs ON, sur EXACTEMENT les memes donnees.

Rejoue le moteur live fidele (10 detecteurs, regime rejoue) deux fois sur le meme
historique : une fois sans la couche adaptative, une fois avec. Compare trades,
winrate, compound, et compte les actions adaptatives (cancel / extend / tighten / exit).

Usage :
    python scripts/backtest_ab_adaptive.py [--days 7] [--tf 15m] [--symbols ...]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_TMP_DB = Path(tempfile.gettempdir()) / "analyseur_ab_adaptive.db"
if _TMP_DB.exists():
    _TMP_DB.unlink()
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TMP_DB.as_posix()}"


def _bars_for(days: int, tf: str) -> int:
    per_day = {"1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}.get(tf, 96)
    return days * per_day


def _clear_tables() -> None:
    con = sqlite3.connect(str(_TMP_DB))
    cur = con.cursor()
    for t in ("unit_trades", "hypotheses", "scan_runs"):
        try:
            cur.execute(f"DELETE FROM {t}")
        except sqlite3.OperationalError:
            pass
    con.commit()
    con.close()


def _collect_stats() -> dict:
    con = sqlite3.connect(str(_TMP_DB))
    cur = con.cursor()
    cur.execute("SELECT COUNT(*), SUM(CASE WHEN pct_gain>0 THEN 1 ELSE 0 END) FROM unit_trades")
    total, wins = cur.fetchone()
    total = total or 0
    wins = wins or 0
    cur.execute("SELECT pct_gain FROM unit_trades ORDER BY entry_timestamp")
    gains = [r[0] or 0.0 for r in cur.fetchall()]
    compound = 1.0
    for g in gains:
        compound *= 1.0 + g / 100.0
    # Compte les actions adaptatives via les tags des hypotheses
    cur.execute("SELECT confluence_tags FROM hypotheses")
    counts = {"adaptive_exit": 0, "adaptive_extend": 0, "adaptive_tighten": 0}
    cancels = 0
    for (tags_str,) in cur.fetchall():
        if not tags_str:
            continue
        try:
            tags = json.loads(tags_str)
        except Exception:
            continue
        for k in counts:
            if k in tags:
                counts[k] += 1
    # Annulations adaptatives = hypotheses INVALIDATED avec transition "adaptive"
    cur.execute("SELECT COUNT(*) FROM hypotheses WHERE state='INVALIDATED' AND transitions LIKE '%adaptive%'")
    cancels = cur.fetchone()[0] or 0
    con.close()
    return {
        "trades": total,
        "winrate": (100 * wins / total) if total else 0.0,
        "sum_pct": sum(gains),
        "compound_pct": (compound - 1.0) * 100.0,
        "actions": counts,
        "cancels": cancels,
    }


async def _run(adaptive: bool, symbols, tf, history_bars, timeline) -> dict:
    from app.config import settings
    from app.services.continuous_scanner import ContinuousScanner, ScanPlan
    from app.services.hypothesis_engine import ConfluenceScorer, HypothesisEngine

    plan = ScanPlan(symbols=symbols, timeframes=[tf], candles_per_fetch=history_bars + 60)
    scanner = ContinuousScanner(plan=plan)
    # Rebuild l'engine avec la MEME config live + flag adaptatif voulu.
    scanner._engine = HypothesisEngine(
        confluence_scorer=ConfluenceScorer(),
        min_confluence_score=float(settings.min_confluence_score),
        min_rr_ratio=float(settings.min_rr_ratio),
        reject_trend_counter=bool(settings.reject_trend_counter),
        require_volume_expansion=bool(settings.require_volume_expansion),
        breakeven_trigger_pct=float(settings.breakeven_trigger_pct),
        reject_if_regime_unknown=True,
        adaptive_enabled=adaptive,
    )
    await scanner.backfill(
        history_bars=history_bars, bars_per_step=1,
        symbols=symbols, timeframes=[tf], regime_timeline=timeline,
    )
    await scanner.stop()
    return _collect_stats()


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--tf", default="15m")
    ap.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT")
    args = ap.parse_args()

    from app.config import settings
    assert "ab_adaptive" in settings.database_url
    from app.db.models import Base
    from app.db.session import engine as app_engine
    from app.services.continuous_scanner import build_regime_timeline, _rows_to_df
    from app.ingestion.ccxt_fetcher import CCXTFetcher

    async with app_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    history_bars = _bars_for(args.days, args.tf)

    fetcher = CCXTFetcher(settings.exchange_id)
    btc_df = _rows_to_df(await fetcher.fetch_ohlcv("BTC/USDT", "1h", limit=500))
    await fetcher.close()
    timeline = build_regime_timeline(btc_df)

    print(f"A/B adaptatif : {len(symbols)} symboles x {args.tf}, {args.days}j, "
          f"timeline regime {len(timeline)} pts")

    print("\n[A] Run BASELINE (adaptatif OFF) ...")
    _clear_tables()
    base = await _run(False, symbols, args.tf, history_bars, timeline)

    print("[B] Run ADAPTATIF (adaptatif ON) ...")
    _clear_tables()
    adpt = await _run(True, symbols, args.tf, history_bars, timeline)

    print("\n" + "=" * 64)
    print(f"COMPARAISON A/B ({args.days}j, {args.tf})")
    print("=" * 64)
    print(f"{'metrique':<22} {'BASELINE':>14} {'ADAPTATIF':>14}")
    print(f"{'trades':<22} {base['trades']:>14} {adpt['trades']:>14}")
    print(f"{'winrate %':<22} {base['winrate']:>13.1f}% {adpt['winrate']:>13.1f}%")
    print(f"{'somme PnL %':<22} {base['sum_pct']:>+13.2f}% {adpt['sum_pct']:>+13.2f}%")
    print(f"{'compound %':<22} {base['compound_pct']:>+13.2f}% {adpt['compound_pct']:>+13.2f}%")
    print()
    print("Actions adaptatives (run B) :")
    print(f"  ordres annules (cancel) : {adpt['cancels']}")
    print(f"  sorties anticipees (exit): {adpt['actions']['adaptive_exit']}")
    print(f"  cibles etendues (extend) : {adpt['actions']['adaptive_extend']}")
    print(f"  stops resserres (tighten): {adpt['actions']['adaptive_tighten']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
