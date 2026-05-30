"""Inspecteur de schéma DB : tables, colonnes, comptes, échantillons.

Usage:
    python scripts/inspect_db.py [db_path]

Sert de base à la construction du dataset d'apprentissage : on veut savoir
quelles variables indépendantes existent et comment relier features -> label.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "analyseur.db"

# Tables dont on veut un aperçu détaillé (schéma + samples + stats)
KEY_TABLES = ["feature_snapshots", "unit_trades", "hypotheses", "signals",
              "market_regime_snapshots"]


def main() -> None:
    if not DB_PATH.exists():
        print(f"DB introuvable: {DB_PATH}")
        return
    print(f"### DB: {DB_PATH}  ({DB_PATH.stat().st_size/1e6:.1f} MB)")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [r["name"] for r in cur.fetchall()]

    print("\n=== ROW COUNTS ===")
    for t in tables:
        try:
            n = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t:32} {n:>10}")
        except Exception as e:
            print(f"  {t:32} ERR {e}")

    for t in KEY_TABLES:
        if t not in tables:
            print(f"\n=== {t} : ABSENTE ===")
            continue
        print(f"\n{'='*70}\n=== SCHEMA: {t} ===\n{'='*70}")
        cols = cur.execute(f"PRAGMA table_info({t})").fetchall()
        for c in cols:
            print(f"  {c['name']:30} {c['type']:12} "
                  f"{'NOTNULL' if c['notnull'] else ''}"
                  f"{' PK' if c['pk'] else ''}")
        # Sample
        try:
            rows = cur.execute(f"SELECT * FROM {t} LIMIT 2").fetchall()
            print(f"  --- {min(2, len(rows))} sample row(s) ---")
            for r in rows:
                d = dict(r)
                # tronque les valeurs longues
                for k, v in d.items():
                    s = str(v)
                    if len(s) > 60:
                        d[k] = s[:57] + "..."
                print(f"    {d}")
        except Exception as e:
            print(f"  sample ERR: {e}")

    conn.close()


if __name__ == "__main__":
    main()
