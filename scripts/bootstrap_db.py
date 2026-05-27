"""Crée les tables PostgreSQL à partir des modèles SQLAlchemy (développement local).

Prérequis :
  1. PostgreSQL installé et démarré (service Windows).
  2. Base créée une fois (compte superuser adapté) :
       psql -U postgres -h 127.0.0.1 -c "CREATE DATABASE analyseur_crypto;"
  3. Fichier .env à la racine du projet avec par exemple :
       DATABASE_URL=postgresql+asyncpg://postgres:TON_MDP@127.0.0.1:5432/analyseur_crypto

Lancer depuis la racine du dépôt :
  .venv\\Scripts\\python scripts/bootstrap_db.py
  ou : python scripts/bootstrap_db.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from app.config import settings  # noqa: E402
from app.db.models.candle import Base  # noqa: E402


async def main() -> None:
    print("DATABASE_URL =", settings.database_url.split("@")[-1] if "@" in settings.database_url else settings.database_url)
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    finally:
        await engine.dispose()
    print("OK : tables créées (ou déjà présentes).")
    print(
        "Alembic : si la table alembic_version est vide, marque la révision actuelle avec :\n"
        "  .venv\\\\Scripts\\\\alembic stamp e8c2aea87d0c\n"
        "(la migration initiale est vide car le schéma vient des modèles ORM.)"
    )


if __name__ == "__main__":
    asyncio.run(main())
