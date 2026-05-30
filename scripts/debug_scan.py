"""Diagnostic: lance une passe scan_once sur un mini-univers et logge tout."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Force une DB de test separee pour ne pas verrouiller analyseur.db
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./analyseur_debug.db"

from app.logging_setup import setup_logging  # noqa: E402


async def main() -> None:
    setup_logging(level_console=logging.DEBUG)
    from app.services.continuous_scanner import ContinuousScanner, ScanPlan, _CODE_VERSION

    print(f"=== Scanner code version: {_CODE_VERSION} ===")

    plan = ScanPlan(
        symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "PEPE/USDT"],
        timeframes=["1h"],
        interval_seconds=60,
        candles_per_fetch=200,
    )
    scanner = ContinuousScanner(plan=plan)
    try:
        print("\n--- SCAN ONCE ---")
        result = await scanner.scan_once()
        print("scan_once:", result)

        print("\n--- BACKFILL (BTC/USDT 1h, 200 bars, step=2) ---")
        result_bf = await scanner.backfill(
            symbols=["BTC/USDT"],
            timeframes=["1h"],
            history_bars=200,
            bars_per_step=2,
        )
        print("backfill:", result_bf)
    finally:
        await scanner.stop()


if __name__ == "__main__":
    asyncio.run(main())
