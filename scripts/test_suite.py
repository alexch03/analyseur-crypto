"""Suite de tests rapides (sans pytest) pour valider l'integrite du systeme.

Lance :
    python scripts/test_suite.py

Teste :
    1. Schema DB (INTEGER PK autoincrement OK)
    2. Detectoreurs de patterns (donnees synthetiques)
    3. HypothesisEngine lifecycle
    4. Analytics computation
    5. Mini backfill (BTC/USDT 1h, 100 bars)
"""

from __future__ import annotations

import asyncio
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# DB de test temporaire
_TEST_DB = "test_suite_tmp.db"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///./{_TEST_DB}"

PASS = "✓"
FAIL = "✗"
results: list[tuple[str, bool, str]] = []


def test(name: str):
    """Decorateur de test."""
    def decorator(fn):
        async def wrapper():
            try:
                await fn() if asyncio.iscoroutinefunction(fn) else fn()
                results.append((name, True, ""))
                print(f"  {PASS}  {name}")
            except Exception as exc:
                tb = traceback.format_exc()
                results.append((name, False, str(exc)))
                print(f"  {FAIL}  {name}")
                print(f"      {exc}")
                if "--verbose" in sys.argv:
                    print(tb)
        return wrapper
    return decorator


# ============================================================
# 1. Schema DB
# ============================================================
@test("DB: Schema INTEGER PK autoincrement")
async def test_db_schema():
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine
    from app.db.models import Base

    engine = create_async_engine(os.environ["DATABASE_URL"])
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Verifie que scan_runs.id est INTEGER (pas BIGINT)
    async with engine.connect() as conn:
        result = await conn.execute(text(
            "SELECT lower(sql) FROM sqlite_master "
            "WHERE type='table' AND name='scan_runs'"
        ))
        row = result.fetchone()
        assert row and row[0], "Table scan_runs manquante"
        sql = row[0]
        assert "bigint" not in sql or "autoincrement" in sql, (
            f"Schema stale detecte dans scan_runs: {sql[:200]}"
        )

    await engine.dispose()


@test("DB: Insert scan_run sans id explicite (autoincrement)")
async def test_db_autoincrement():
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine
    from app.db.models import Base, ScanRun
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker
    from datetime import datetime, UTC

    engine = create_async_engine(os.environ["DATABASE_URL"])
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        # Ajoute un ScanRun sans id : doit autoincrementer
        run = ScanRun(
            symbol_id=1,
            timeframe_id=1,
            ts_started=datetime.now(UTC),
            ts_finished=datetime.now(UTC),
            candles_fetched=0,
            patterns_detected=0,
            hypotheses_active=0,
        )
        session.add(run)
        await session.flush()
        assert run.id is not None and run.id > 0, f"id non assigné: {run.id}"
        await session.rollback()

    await engine.dispose()


# ============================================================
# 2. Detecteurs de patterns (donnees synthetiques)
# ============================================================

def _make_df(n: int = 100, close_fn=None):
    import pandas as pd
    import numpy as np
    rng = np.random.default_rng(42)
    closes = [100.0]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1.0 + rng.normal(0, 0.005)))
    if close_fn:
        closes = [close_fn(i, n) for i in range(n)]
    lows = [c * 0.997 for c in closes]
    highs = [c * 1.003 for c in closes]
    opens = [c * 0.999 for c in closes]
    volumes = [1_000_000 + rng.integers(-200000, 200000) for _ in range(n)]
    ts = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


@test("Detectors: TriangleDetector (import + detect)")
def test_triangle():
    from app.patterns.triangles import TriangleDetector
    from app.market_structure.swings import detect_swings
    df = _make_df(120)
    swings = detect_swings(df, left=3, right=3)
    det = TriangleDetector()
    patterns = det.detect(df, swings, symbol="BTC/USDT", timeframe="15m")
    assert isinstance(patterns, list)


@test("Detectors: RectangleDetector (import + detect)")
def test_rectangle():
    from app.patterns.rectangles import RectangleDetector
    from app.market_structure.swings import detect_swings
    df = _make_df(120)
    swings = detect_swings(df, left=3, right=3)
    det = RectangleDetector()
    patterns = det.detect(df, swings, symbol="BTC/USDT", timeframe="15m")
    assert isinstance(patterns, list)


@test("Detectors: WedgeDetector (import + detect)")
def test_wedge():
    from app.patterns.wedges import WedgeDetector
    from app.market_structure.swings import detect_swings
    df = _make_df(120)
    swings = detect_swings(df, left=3, right=3)
    det = WedgeDetector()
    patterns = det.detect(df, swings, symbol="ETH/USDT", timeframe="1h")
    assert isinstance(patterns, list)


@test("Detectors: FlagDetector (import + detect)")
def test_flag():
    from app.patterns.flags import FlagDetector
    from app.market_structure.swings import detect_swings
    df = _make_df(120)
    swings = detect_swings(df, left=3, right=3)
    det = FlagDetector()
    patterns = det.detect(df, swings, symbol="SOL/USDT", timeframe="15m")
    assert isinstance(patterns, list)


@test("Detectors: ReversalDetector (import + detect)")
def test_reversal():
    from app.patterns.reversal import ReversalDetector
    from app.market_structure.swings import detect_swings
    df = _make_df(120)
    swings = detect_swings(df, left=3, right=3)
    det = ReversalDetector()
    patterns = det.detect(df, swings, symbol="BTC/USDT", timeframe="4h")
    assert isinstance(patterns, list)


# ============================================================
# 3. HypothesisEngine lifecycle
# ============================================================
@test("Engine: lifecycle FORMING→TRIGGERED→TARGET_HIT")
def test_engine_lifecycle():
    import pandas as pd
    from datetime import datetime, UTC
    from app.services.hypothesis_engine import HypothesisEngine, ConfluenceScorer
    from app.schemas.patterns import ChartPatternDTO, PatternKind, BreakoutDirection, TrendLine
    from app.schemas.hypothesis import HypothesisState

    engine = HypothesisEngine(confluence_scorer=ConfluenceScorer())

    # Cree un DataFrame minimal
    ts = pd.date_range("2024-01-01", periods=80, freq="15min", tz="UTC")
    closes = [100.0] * 80
    df = pd.DataFrame({
        "timestamp": ts,
        "open": closes,
        "high": [c * 1.002 for c in closes],
        "low": [c * 0.998 for c in closes],
        "close": closes,
        "volume": [1_000_000.0] * 80,
    })

    # Pattern avec breakout_level=101 (juste au-dessus)
    from datetime import timedelta
    p = ChartPatternDTO(
        kind=PatternKind.TRIANGLE_ASC,
        symbol="BTC/USDT",
        timeframe="15m",
        start_index=0,
        end_index=60,
        start_timestamp=ts[0].to_pydatetime(),
        end_timestamp=ts[60].to_pydatetime(),
        breakout_level=101.0,
        invalidation_level=97.0,
        breakout_direction=BreakoutDirection.UP,
        height=4.0,
        target=105.0,
        confidence=0.8,
    )

    # Step 1 : creation
    result = engine.step(df, [p], [])
    assert len(result.created) == 1
    h = result.created[0]
    assert h.state == HypothesisState.FORMING

    # Step 2 : bougie qui casse au-dessus du breakout
    ts2 = pd.date_range("2024-01-01 20:00", periods=5, freq="15min", tz="UTC")
    df2 = pd.DataFrame({
        "timestamp": ts2,
        "open": [101.5] * 5,
        "high": [106.0] * 5,   # depasse target (105)
        "low": [101.0] * 5,
        "close": [105.5] * 5,
        "volume": [2_000_000.0] * 5,
    })
    result2 = engine.step(df2, [], [h])
    assert len(result2.updated) == 1
    h2 = result2.updated[0]
    # Doit etre TARGET_HIT ou au moins TRIGGERED
    assert h2.state in (HypothesisState.TRIGGERED, HypothesisState.TARGET_HIT), (
        f"State inattendu: {h2.state}"
    )


# ============================================================
# 4. Analytics computation
# ============================================================
@test("Analytics: compute_breakdowns (DB vide ok)")
async def test_analytics_empty():
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from app.db.models import Base
    from app.services.analytics import compute_breakdowns, optimize_filters

    engine = create_async_engine(os.environ["DATABASE_URL"])
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        bd = await compute_breakdowns(session)
        assert "overall" in bd
        assert bd["overall"][0]["count"] == 0

        opt = await optimize_filters(session, top_n=5)
        assert "trades_available" in opt
        assert opt["trades_available"] == 0

    await engine.dispose()


# ============================================================
# 5. Mini backfill (optionnel, desactive si --no-network)
# ============================================================
@test("Backfill: mini BTC/USDT 15m 80 bars (requiert connexion)")
async def test_mini_backfill():
    if "--no-network" in sys.argv:
        print("     [skip] --no-network")
        return

    from app.services.continuous_scanner import ContinuousScanner, ScanPlan
    from app.db.models import Base
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(os.environ["DATABASE_URL"])
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()

    plan = ScanPlan(symbols=["BTC/USDT"], timeframes=["15m"])
    scanner = ContinuousScanner(plan=plan)
    try:
        result = await scanner.backfill(history_bars=80, bars_per_step=5)
        assert "total_steps" in result
        assert result["total_steps"] > 0, "Aucun step effectue"
        print(f"     {result['total_steps']} steps, {result['total_patterns_detected']} patterns")
    finally:
        await scanner.stop()


# ============================================================
# Main
# ============================================================
async def run_all():
    print("\n" + "=" * 60)
    print("TEST SUITE — Analyseur Crypto")
    print("=" * 60)

    tests = [
        test_db_schema,
        test_db_autoincrement,
        test_triangle,
        test_rectangle,
        test_wedge,
        test_flag,
        test_reversal,
        test_engine_lifecycle,
        test_analytics_empty,
        test_mini_backfill,
    ]

    for t in tests:
        await t()

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"Résultat : {passed} OK  {failed} ECHEC  / {len(results)} tests")
    if failed:
        print("\nECHECS :")
        for name, ok, msg in results:
            if not ok:
                print(f"  {FAIL} {name}: {msg}")

    # Nettoyage DB de test
    db = Path(_TEST_DB)
    if db.exists():
        db.unlink()
        print(f"\nDB de test supprimée : {_TEST_DB}")

    return failed == 0


if __name__ == "__main__":
    ok = asyncio.run(run_all())
    sys.exit(0 if ok else 1)
