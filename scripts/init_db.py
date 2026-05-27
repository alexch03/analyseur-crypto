"""Cree ou repare toutes les tables (SQLite) via SQLAlchemy create_all.

- Detecte automatiquement les schemas stales (BIGINT sans AUTOINCREMENT)
  et recrée la DB apres backup.
- Lance-le a chaque demarrage : idempotent si le schema est deja correct.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from app.config import settings  # noqa: E402
from app.db.models import Base  # noqa: E402  # importe tous les modeles


# Tables qui utilisaient autrefois BIGINT PK (schema stale)
_BIGINT_TABLES = (
    "scan_runs",
    "candles",
    "swing_points",
    "sr_levels",
    "market_structure_events",
    "signals",
    "paper_orders",
    "telegram_deliveries",
    "feature_snapshots",
)


async def detect_stale_schema(engine) -> list[str]:
    """Retourne la liste des tables avec schema stale (BIGINT PK sans AUTOINCREMENT)."""
    stale: list[str] = []
    try:
        async with engine.connect() as conn:
            for tname in _BIGINT_TABLES:
                result = await conn.execute(text(
                    "SELECT lower(sql) FROM sqlite_master "
                    "WHERE type='table' AND name=:tname"
                ), {"tname": tname})
                row = result.fetchone()
                if row and row[0]:
                    sql = row[0]
                    # Schema stale : "id bigint" sans "autoincrement"
                    # Schema correct : "id integer" (SQLite rowid alias)
                    if "bigint" in sql and "autoincrement" not in sql:
                        stale.append(tname)
    except Exception as exc:
        print(f"  [WARN] Impossible de verifier le schema: {exc}")
    return stale


async def repair_schema(engine, stale_tables: list[str]) -> None:
    """Backup de la DB, puis drop + recreate toutes les tables."""
    db_url = str(engine.url)
    # Backup du fichier SQLite
    if "sqlite" in db_url.lower():
        path_part = db_url.split("///", 1)[-1]
        if path_part.startswith("./"):
            path_part = path_part[2:]
        db_path = Path(path_part).resolve()
        if db_path.exists():
            bak = db_path.with_suffix(".db.bak")
            shutil.copy2(str(db_path), str(bak))
            print(f"  [OK] Backup sauvegardé : {bak}")

    print(f"  [!] Tables stales ({stale_tables}) — drop + recreate ...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    print("  [OK] Schema recree : INTEGER PK (SQLite autoincrement natif)")


async def main() -> None:
    url = settings.database_url
    short = url if "@" not in url else url.split("@")[-1]
    print(f"DATABASE_URL: {short}")

    engine = create_async_engine(url, pool_pre_ping=True)
    try:
        # 1. Detecte schema stale
        stale = await detect_stale_schema(engine)
        if stale:
            print(f"  [!] Schema incompatible detecte : {stale}")
            await engine.dispose()
            # Recrée le engine pour avoir un handle propre
            engine = create_async_engine(url, pool_pre_ping=True)
            await repair_schema(engine, stale)
        else:
            # 2. Cree les tables manquantes (idempotent)
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            print("OK : tables creees (ou deja presentes avec schema correct).")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
