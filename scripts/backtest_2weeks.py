"""Backtest profitabilite 2 semaines + grid search filtres.

1. Reset DB (debug_v2)
2. Backfill 14 jours x 15m sur top 10 cryptos liquides
3. Compute analytics baseline
4. Grid search filtres
5. Re-run avec config gagnante + breakeven SL active
6. Affiche comparaison avant/apres
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# DB file changes between phases to avoid lock conflicts.
_DB_PHASE = {"current": "analyseur_backtest_p1.db"}


def _set_db_phase(name: str) -> None:
    _DB_PHASE["current"] = name
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///./{name}"


async def reset_db() -> None:
    """Supprime + recrée la DB, recharge les modules session pour éviter
    que le session_factory pointe toujours sur l'ancienne DB."""
    import importlib
    import sys
    from sqlalchemy.ext.asyncio import create_async_engine

    # 1. Reload app.config → re-instancie Settings() avec le nouvel env var
    if "app.config" in sys.modules:
        try:
            importlib.reload(sys.modules["app.config"])
        except Exception:
            pass

    # 2. Dispose + reload session_mod → nouveau engine avec nouveau database_url
    if "app.db.session" in sys.modules:
        sm = sys.modules["app.db.session"]
        try:
            await sm.engine.dispose()
        except Exception:
            pass
        try:
            importlib.reload(sm)
        except Exception:
            pass

    # 3. Reload continuous_scanner → relie async_session_factory
    if "app.services.continuous_scanner" in sys.modules:
        try:
            importlib.reload(sys.modules["app.services.continuous_scanner"])
        except Exception:
            pass

    db_file = Path(_DB_PHASE["current"])
    if db_file.exists():
        try:
            db_file.unlink()
        except PermissionError:
            print(f"[WARN] cannot unlink {db_file}, leaving in place")
    eng = create_async_engine(os.environ["DATABASE_URL"])
    from app.db.models import Base
    async with eng.begin() as c:
        await c.run_sync(Base.metadata.create_all)
    await eng.dispose()
    print(f"[OK] DB reset -> {db_file}")


TOP_10_LIQUID = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT",
    "AVAX/USDT", "DOT/USDT", "LINK/USDT", "DOGE/USDT", "BNB/USDT",
]

# 2 semaines x 15m = 4*24*14 = 1344 bars + buffer warmup
HISTORY_BARS = 1400
TIMEFRAME = "15m"


async def run_backfill(min_conf: float = 0.0,
                       reject_counter: bool = False,
                       require_volume: bool = False,
                       breakeven_pct: float = 0.0,
                       min_rr: float = 0.0) -> dict:
    from app.services.continuous_scanner import ContinuousScanner, ScanPlan
    from app.services.hypothesis_engine import HypothesisEngine, ConfluenceScorer
    plan = ScanPlan(
        symbols=TOP_10_LIQUID,
        timeframes=[TIMEFRAME],
        candles_per_fetch=HISTORY_BARS + 50,
    )
    scanner = ContinuousScanner(plan=plan)
    scanner._engine = HypothesisEngine(
        confluence_scorer=ConfluenceScorer(),
        min_confluence_score=min_conf,
        reject_trend_counter=reject_counter,
        require_volume_expansion=require_volume,
        breakeven_trigger_pct=breakeven_pct,
        min_rr_ratio=min_rr,
    )
    try:
        result = await scanner.backfill(
            history_bars=HISTORY_BARS, bars_per_step=1
        )
    finally:
        await scanner.stop()
    return result


async def compute_stats() -> dict:
    """Stats depuis la DB courante (engine direct, évite le cache module)."""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from app.services.analytics import compute_breakdowns, optimize_filters

    engine = create_async_engine(os.environ["DATABASE_URL"])
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with Session() as session:
            return {
                "breakdowns": await compute_breakdowns(session),
                "optimize": await optimize_filters(session, top_n=10),
            }
    finally:
        await engine.dispose()


def print_segment(label: str, s: dict) -> None:
    cls = "+" if s["cumul_compound_pct"] >= 0 else ""
    print(f"  {label:40} N={s['count']:5}  win={s['win_rate_pct']:5.1f}%  "
          f"exp={s['expectancy_pct']:+6.3f}%  cumul_compound={cls}{s['cumul_compound_pct']:+8.2f}%")


async def main() -> None:
    from app.logging_setup import setup_logging
    setup_logging(level_console=logging.WARNING, disable_file=True)

    print("=" * 72)
    print(f"BACKTEST 2 SEMAINES x {len(TOP_10_LIQUID)} symbols x {TIMEFRAME}")
    print("=" * 72)

    # === Phase 1: baseline ===
    print("\n[1/3] Reset DB + backfill baseline (zero filtre)")
    _set_db_phase("analyseur_backtest_p1.db")
    await reset_db()
    bf1 = await run_backfill()
    print(f"   {bf1}")

    stats1 = await compute_stats()
    base = stats1["breakdowns"]["overall"][0]
    print("\nBASELINE")
    print_segment("ALL (no filter)", base)

    print("\nTOP 5 segments par expectancy :")
    for s in stats1["breakdowns"]["by_pattern_x_tag"][:5]:
        print_segment(s["label"][:40], s)
    print("\nBOTTOM 5 segments par expectancy :")
    for s in stats1["breakdowns"]["by_pattern_x_tag"][-5:]:
        print_segment(s["label"][:40], s)

    # === Phase 2: grid search ===
    print("\n[2/3] Grid search virtuel sur les filtres...")
    opt = stats1["optimize"]
    if not opt["top_configs"]:
        print("   Pas assez de trades pour optimiser.")
        return
    print(f"\nTOP 10 CONFIGS sur {opt['trades_available']} trades :")
    for i, c in enumerate(opt["top_configs"][:10], 1):
        print(f"   #{i:2}  N={c['count']:5}  win={c['win_rate_pct']:5.1f}%  "
              f"exp={c['expectancy_pct']:+6.3f}%  compound={c['cumul_compound_pct']:+7.2f}%  | "
              f"score>={c['config']['min_confluence_score']}  "
              f"reject={c['config']['reject_tags']}  "
              f"require={c['config']['required_tags']}")

    best = opt["top_configs"][0]
    cfg = best["config"]
    print(f"\nMEILLEURE CONFIG (virtuelle) : {cfg}")

    # === Phase 3: re-backfill avec best filters + breakeven 0.5 ===
    print("\n[3/3] Reset + backfill avec best config + BE=0.5...")
    _set_db_phase("analyseur_backtest_p3.db")
    await reset_db()
    require_vol = "volume_expansion" in cfg["required_tags"]
    reject_ctr = "trend_counter" in cfg["reject_tags"]
    bf3 = await run_backfill(
        min_conf=cfg["min_confluence_score"],
        reject_counter=reject_ctr,
        require_volume=require_vol,
        breakeven_pct=0.5,
    )
    print(f"   {bf3}")
    stats3 = await compute_stats()
    final = stats3["breakdowns"]["overall"][0]

    print("\n" + "=" * 72)
    print("RESULTAT FINAL")
    print("=" * 72)
    print_segment("Baseline (no filter)         ", base)
    print_segment("Best filters virtuels        ", best)
    print_segment("Best filters + BE 50% (REEL) ", final)

    print(f"\n>>> Variables .env recommandées :")
    print(f"MIN_CONFLUENCE_SCORE={cfg['min_confluence_score']}")
    if reject_ctr:
        print("REJECT_TREND_COUNTER=true")
    if require_vol:
        print("REQUIRE_VOLUME_EXPANSION=true")
    print("BREAKEVEN_TRIGGER_PCT=0.5")


if __name__ == "__main__":
    asyncio.run(main())
