"""Cleanup : INVALIDATE les hypotheses FORMING/ARMED qui ne passent plus le filtre regime.

Apres un changement d'affinite ou de seuil, les hypotheses creees AVANT le fix
restent en DB et risquent de trigger contre-regime. Ce script les marque INVALIDATED
proprement (avec transition log) pour eviter le carnage.

USAGE:
  python scripts/cleanup_counter_regime.py --dry-run   # voir ce qui serait invalide
  python scripts/cleanup_counter_regime.py             # apply

Le script lit le regime courant depuis market_regime_snapshots (dernier snapshot)
et applique PATTERN_REGIME_AFFINITY + min_score=0.65 pour decider.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "analyseur.db"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.services.market_regime import PATTERN_REGIME_AFFINITY  # noqa: E402

MIN_SCORE = 0.65  # cf. _DEFAULT_REGIME_MIN_SCORE dans hypothesis_engine.py


def regime_score(pattern_kind: str, trend: str, strength: float) -> float:
    affin = PATTERN_REGIME_AFFINITY.get(pattern_kind)
    if not affin:
        return 1.0
    base = affin.get(trend, 1.0)
    delta = base - 1.0
    return 1.0 + delta * strength


def main() -> None:
    apply = "--apply" in sys.argv or "-y" in sys.argv
    if not apply and "--dry-run" not in sys.argv:
        print("USAGE: python cleanup_counter_regime.py [--dry-run | --apply]")
        print("Default: dry-run (aucune modification)")
        apply = False

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 1. Lire le regime courant
    cur.execute("""
        SELECT trend, strength, snapshot_ts, btc_change_24h_pct, breadth_pct
        FROM market_regime_snapshots ORDER BY snapshot_ts DESC LIMIT 1;
    """)
    row = cur.fetchone()
    if not row:
        print("Pas de regime en DB. Lance le scanner d'abord.")
        return

    trend = row["trend"]
    strength = row["strength"]
    print("=" * 70)
    print(f"REGIME COURANT : {trend} strength={strength:.2f}")
    print(f"  Snapshot @ {row['snapshot_ts']}")
    print(f"  BTC 24h={row['btc_change_24h_pct']:.2f}%, breadth={row['breadth_pct']:.1f}%")
    print("=" * 70)
    print()

    # 2. Lister TOUS les FORMING/ARMED
    cur.execute("""
        SELECT h.id, h.pattern_kind, h.side, h.state, h.entry_price,
               h.invalidation_price, h.target_price, h.created_at,
               s.base || '/' || s.quote as sym
        FROM hypotheses h
        JOIN symbols s ON s.id = h.symbol_id
        WHERE h.state IN ('FORMING', 'ARMED')
        ORDER BY h.created_at;
    """)
    candidates = [dict(r) for r in cur.fetchall()]
    print(f"Hypotheses FORMING/ARMED : {len(candidates)}")
    print()

    # 3. Determiner lesquelles ne passent plus
    to_invalidate = []
    keep = []
    for h in candidates:
        score = regime_score(h["pattern_kind"], trend, strength)
        if score < MIN_SCORE:
            to_invalidate.append({**h, "regime_score": score})
        else:
            keep.append({**h, "regime_score": score})

    # 4. Stats
    print("=" * 70)
    print("ANALYSE")
    print("=" * 70)
    print(f"  A INVALIDER  : {len(to_invalidate)} hypotheses counter-regime")
    print(f"  A GARDER     : {len(keep)} hypotheses compatibles")
    print()

    if to_invalidate:
        print("  Breakdown a invalider par pattern :")
        by_kind = defaultdict(list)
        for h in to_invalidate:
            by_kind[h["pattern_kind"]].append(h)
        for k, hs in sorted(by_kind.items(), key=lambda x: -len(x[1])):
            avg_score = sum(h["regime_score"] for h in hs) / len(hs)
            sides = defaultdict(int)
            states = defaultdict(int)
            for h in hs:
                sides[h["side"]] += 1
                states[h["state"]] += 1
            sides_s = " ".join(f"{k}={v}" for k, v in sides.items())
            states_s = " ".join(f"{k}={v}" for k, v in states.items())
            print(f"    {k:30} n={len(hs):4} score_avg={avg_score:.2f} | {sides_s} | {states_s}")

    print()
    if keep:
        print("  Breakdown a garder par pattern :")
        by_kind = defaultdict(list)
        for h in keep:
            by_kind[h["pattern_kind"]].append(h)
        for k, hs in sorted(by_kind.items(), key=lambda x: -len(x[1])):
            avg_score = sum(h["regime_score"] for h in hs) / len(hs)
            print(f"    {k:30} n={len(hs):4} score_avg={avg_score:.2f}")
    print()

    # 5. Apply
    if not apply:
        print("[DRY-RUN] Aucune modification. Relance avec --apply pour invalider.")
        return

    if not to_invalidate:
        print("Rien a invalider, exit.")
        return

    print(f"INVALIDATION de {len(to_invalidate)} hypotheses...")
    now_iso = datetime.now(timezone.utc).isoformat()
    transition = {
        "from_state": "ARMED",
        "to_state": "INVALIDATED",
        "timestamp": now_iso,
        "price": 0.0,
        "reason": f"manual cleanup: counter-regime in {trend} strength={strength:.2f}",
    }
    transition_json = json.dumps([transition])

    invalidated = 0
    for h in to_invalidate:
        # Update transitions: append au JSON existant
        cur.execute("SELECT transitions FROM hypotheses WHERE id = ?;", (h["id"],))
        r = cur.fetchone()
        existing = []
        if r and r["transitions"]:
            try:
                existing = json.loads(r["transitions"])
            except Exception:
                existing = []
        existing.append({**transition, "from_state": h["state"]})
        cur.execute("""
            UPDATE hypotheses
            SET state = 'INVALIDATED',
                closed_at = ?,
                updated_at = ?,
                transitions = ?
            WHERE id = ? AND state IN ('FORMING', 'ARMED');
        """, (now_iso, now_iso, json.dumps(existing), h["id"]))
        invalidated += cur.rowcount

    conn.commit()
    print(f"  -> {invalidated} hypotheses passees a INVALIDATED")
    conn.close()


if __name__ == "__main__":
    main()
