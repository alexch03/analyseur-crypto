"""Diagnostic complet : pourquoi sa marche moins bien depuis le regime adaptive.

Compare les trades AVANT et APRES integration du regime,
verifie que le lifecycle est bien tenu, et identifie les regressions.
"""

from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "analyseur.db"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def fmt_pct(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}%"


def main() -> None:
    if not DB_PATH.exists():
        print(f"DB introuvable: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # ============================================================
    # 1. INVENTAIRE TABLES
    # ============================================================
    print("=" * 70)
    print("INVENTAIRE DB")
    print("=" * 70)
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [r["name"] for r in cur.fetchall()]
    print(f"Tables: {tables}")
    print()

    # ============================================================
    # 2. ETAT HYPOTHESES (lifecycle global)
    # ============================================================
    print("=" * 70)
    print("HYPOTHESES PAR STATE (lifecycle)")
    print("=" * 70)
    cur.execute("""
        SELECT state, COUNT(*) as n,
               ROUND(AVG((JULIANDAY('now') - JULIANDAY(created_at)) * 24.0), 1) as age_h
        FROM hypotheses GROUP BY state ORDER BY n DESC;
    """)
    for r in cur.fetchall():
        print(f"  {r['state']:18} n={r['n']:6}  age_moy={r['age_h']:.1f}h")
    print()

    # ============================================================
    # 3. DATES DE COUPURE
    # ============================================================
    print("=" * 70)
    print("DATES DE COUPURE")
    print("=" * 70)
    cur.execute("SELECT MIN(snapshot_ts), MAX(snapshot_ts), COUNT(*) FROM market_regime_snapshots;")
    row = cur.fetchone()
    regime_start = row[0]
    print(f"  Snapshots regime  : {regime_start} -> {row[1]}  ({row[2]} snaps)")

    cur.execute("SELECT MIN(created_at), MAX(created_at) FROM hypotheses;")
    r = cur.fetchone()
    print(f"  Hypotheses        : {r[0]} -> {r[1]}")

    cur.execute("SELECT MIN(entry_timestamp), MAX(entry_timestamp), MIN(exit_timestamp), MAX(exit_timestamp) FROM unit_trades;")
    r = cur.fetchone()
    print(f"  unit_trades entry : {r[0]} -> {r[1]}")
    print(f"  unit_trades exit  : {r[2]} -> {r[3]}")
    print()

    # ============================================================
    # 4. UNIT_TRADES = SOURCE DE VERITE PNL
    # ============================================================
    print("=" * 70)
    print("UNIT_TRADES : OUTCOMES")
    print("=" * 70)
    cur.execute("""
        SELECT outcome, COUNT(*) n,
               ROUND(SUM(pct_gain), 2) as tot,
               ROUND(AVG(pct_gain), 3) as avg
        FROM unit_trades GROUP BY outcome ORDER BY n DESC;
    """)
    for r in cur.fetchall():
        print(f"  {str(r['outcome']):14} n={r['n']:6}  total={r['tot']}  avg={r['avg']}")
    print()

    # ============================================================
    # 5. STATS AVANT / APRES REGIME (sur unit_trades clos)
    # ============================================================
    print("=" * 70)
    print("COMPARAISON AVANT / APRES INTEGRATION REGIME")
    print("=" * 70)

    cur.execute("""
        SELECT u.id, u.side, u.pattern_kind, u.entry_price, u.exit_price,
               u.entry_timestamp, u.exit_timestamp, u.pct_gain, u.outcome,
               u.confluence_score,
               s.base || '/' || s.quote as sym
        FROM unit_trades u
        JOIN symbols s ON s.id = u.symbol_id
        WHERE u.exit_timestamp IS NOT NULL
        ORDER BY u.exit_timestamp;
    """)
    closed = [dict(r) for r in cur.fetchall()]
    print(f"  Total trades clos : {len(closed)}")

    if not closed:
        print("  AUCUN trade clos")
        conn.close()
        return

    before, after = [], []
    for t in closed:
        ts = t["exit_timestamp"]
        if regime_start and ts >= regime_start:
            after.append(t)
        else:
            before.append(t)

    def stats(trades, label):
        if not trades:
            print(f"\n  === {label} : aucun trade ===")
            return
        wins = [t for t in trades if (t["pct_gain"] or 0) > 0]
        losses = [t for t in trades if (t["pct_gain"] or 0) <= 0]
        wr = 100.0 * len(wins) / len(trades)
        tot = sum(t["pct_gain"] or 0 for t in trades)
        avg_w = sum(t["pct_gain"] for t in wins) / len(wins) if wins else 0.0
        avg_l = sum(t["pct_gain"] for t in losses) / len(losses) if losses else 0.0
        sum_w = sum(t["pct_gain"] for t in wins)
        sum_l = sum(t["pct_gain"] for t in losses)
        pf = abs(sum_w / sum_l) if sum_l != 0 else float("inf")
        first = min(t["exit_timestamp"] for t in trades)
        last = max(t["exit_timestamp"] for t in trades)
        print(f"\n  === {label} ===")
        print(f"    Periode    : {first} -> {last}")
        print(f"    Trades     : {len(trades)}  W:{len(wins)} L:{len(losses)}")
        print(f"    Winrate    : {wr:.1f}%")
        print(f"    PnL cumule : {fmt_pct(tot)}")
        print(f"    Avg win    : {fmt_pct(avg_w)}")
        print(f"    Avg loss   : {fmt_pct(avg_l)}")
        print(f"    Profit fac : {pf:.2f}")
        # Outcome split
        by_oc = defaultdict(list)
        for t in trades:
            by_oc[t["outcome"] or "NULL"].append(t)
        print(f"    OUTCOMES   :")
        for oc, ts in sorted(by_oc.items()):
            wr_oc = 100.0 * sum(1 for x in ts if (x['pct_gain'] or 0) > 0) / len(ts)
            print(f"      {oc:20} n={len(ts):4} wr={wr_oc:5.1f}% sum={fmt_pct(sum(x['pct_gain'] or 0 for x in ts))}")
        # Pattern breakdown
        by_k = defaultdict(list)
        for t in trades:
            by_k[t["pattern_kind"]].append(t)
        rows = []
        for k, ts in by_k.items():
            w = sum(1 for x in ts if (x['pct_gain'] or 0) > 0)
            p = sum(x['pct_gain'] or 0 for x in ts)
            rows.append((k, len(ts), w, 100.0 * w / len(ts), p))
        rows.sort(key=lambda x: x[4], reverse=True)
        print(f"    PATTERNS   :")
        for k, n, w, wr2, p in rows:
            print(f"      {k:30} n={n:4} W={w:4} WR={wr2:5.1f}% PnL={fmt_pct(p)}")
        # Long vs short
        longs = [t for t in trades if t["side"] == "LONG"]
        shorts = [t for t in trades if t["side"] == "SHORT"]
        if longs:
            lwr = 100.0 * sum(1 for x in longs if (x['pct_gain'] or 0) > 0) / len(longs)
            print(f"    LONGS  : n={len(longs):4} WR={lwr:5.1f}% PnL={fmt_pct(sum(x['pct_gain'] or 0 for x in longs))}")
        if shorts:
            swr = 100.0 * sum(1 for x in shorts if (x['pct_gain'] or 0) > 0) / len(shorts)
            print(f"    SHORTS : n={len(shorts):4} WR={swr:5.1f}% PnL={fmt_pct(sum(x['pct_gain'] or 0 for x in shorts))}")

    stats(before, "AVANT REGIME (baseline)")
    stats(after, "APRES REGIME")
    print()

    # ============================================================
    # 6. VERIFICATION CALCULS PNL (sample)
    # ============================================================
    print("=" * 70)
    print("VERIFICATION PNL (10 derniers trades clos)")
    print("=" * 70)
    cur.execute("""
        SELECT u.id, u.side, u.pattern_kind, u.entry_price, u.exit_price,
               u.pct_gain, u.outcome, u.exit_timestamp,
               s.base || '/' || s.quote as sym
        FROM unit_trades u
        JOIN symbols s ON s.id = u.symbol_id
        WHERE u.exit_timestamp IS NOT NULL
        ORDER BY u.exit_timestamp DESC LIMIT 10;
    """)
    for r in cur.fetchall():
        e, x, side = r["entry_price"], r["exit_price"], r["side"]
        if e and x:
            if side == "LONG":
                calc = (x / e - 1.0) * 100.0
            else:
                calc = (e / x - 1.0) * 100.0
            diff = abs(calc - (r['pct_gain'] or 0))
            ok = "OK" if diff < 0.05 else f"!! diff={diff:.3f}"
            print(f"  {r['pattern_kind']:24} {r['sym']:10} {side:5} "
                  f"E={e:.5g} X={x:.5g} stored={fmt_pct(r['pct_gain'] or 0):>9} "
                  f"calc={fmt_pct(calc):>9}  {ok}  outcome={r['outcome']}")
    print()

    # ============================================================
    # 7. DUREE DES STOPS (entries premature?)
    # ============================================================
    print("=" * 70)
    print("DUREE DES TRADES (du trigger au close)")
    print("=" * 70)
    cur.execute("""
        SELECT
          CASE
            WHEN (JULIANDAY(exit_timestamp) - JULIANDAY(entry_timestamp)) * 1440 < 5 THEN 'A: < 5 min'
            WHEN (JULIANDAY(exit_timestamp) - JULIANDAY(entry_timestamp)) * 1440 < 15 THEN 'B: 5-15 min'
            WHEN (JULIANDAY(exit_timestamp) - JULIANDAY(entry_timestamp)) * 1440 < 60 THEN 'C: 15-60 min'
            WHEN (JULIANDAY(exit_timestamp) - JULIANDAY(entry_timestamp)) * 24 < 4 THEN 'D: 1-4 h'
            WHEN (JULIANDAY(exit_timestamp) - JULIANDAY(entry_timestamp)) * 24 < 12 THEN 'E: 4-12 h'
            ELSE 'F: > 12 h'
          END as duree,
          COUNT(*) as n,
          ROUND(SUM(CASE WHEN pct_gain > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as wr,
          ROUND(SUM(pct_gain), 2) as pnl
        FROM unit_trades
        WHERE exit_timestamp IS NOT NULL
        GROUP BY duree
        ORDER BY duree;
    """)
    for r in cur.fetchall():
        print(f"  {r['duree']:14} n={r['n']:5}  wr={r['wr']:5}%  pnl={r['pnl']}")
    print()

    # ============================================================
    # 8. SCORES CONFLUENCE PAR OUTCOME
    # ============================================================
    print("=" * 70)
    print("CONFLUENCE SCORE vs OUTCOME (la qualite est-elle predictive?)")
    print("=" * 70)
    cur.execute("""
        SELECT
          CASE
            WHEN confluence_score < 0.3 THEN 'A: < 0.3'
            WHEN confluence_score < 0.5 THEN 'B: 0.3-0.5'
            WHEN confluence_score < 0.7 THEN 'C: 0.5-0.7'
            WHEN confluence_score < 0.85 THEN 'D: 0.7-0.85'
            ELSE 'E: > 0.85'
          END as bucket,
          COUNT(*) n,
          ROUND(SUM(CASE WHEN pct_gain > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as wr,
          ROUND(SUM(pct_gain), 2) as pnl
        FROM unit_trades
        WHERE exit_timestamp IS NOT NULL
        GROUP BY bucket
        ORDER BY bucket;
    """)
    for r in cur.fetchall():
        print(f"  {r['bucket']:12} n={r['n']:5}  wr={r['wr']:5}%  pnl={r['pnl']}")
    print()

    # ============================================================
    # 9. REGIME AU MOMENT DU TRADE (sur post-regime trades)
    # ============================================================
    if regime_start:
        print("=" * 70)
        print("REGIME AU CLOSE x SIDE x WR (apres integration)")
        print("=" * 70)
        cur.execute("""
            SELECT
              (SELECT trend FROM market_regime_snapshots r
               WHERE r.snapshot_ts <= u.exit_timestamp
               ORDER BY r.snapshot_ts DESC LIMIT 1) as regime,
              (SELECT strength FROM market_regime_snapshots r
               WHERE r.snapshot_ts <= u.exit_timestamp
               ORDER BY r.snapshot_ts DESC LIMIT 1) as strength,
              u.side,
              u.pattern_kind,
              u.pct_gain
            FROM unit_trades u
            WHERE u.exit_timestamp IS NOT NULL AND u.exit_timestamp >= ?;
        """, (regime_start,))
        rows = list(cur.fetchall())
        regime_side = defaultdict(list)
        for r in rows:
            key = f"{r['regime']}/{r['side']}"
            regime_side[key].append(r["pct_gain"] or 0)
        print(f"  Trades post-regime: {len(rows)}")
        print(f"  {'Bucket':18}{'N':>5}{'WR%':>7}{'PnL':>10}{'Avg':>8}")
        for k, pcs in sorted(regime_side.items()):
            wr = 100.0 * sum(1 for p in pcs if p > 0) / len(pcs)
            tot = sum(pcs)
            avg = tot / len(pcs)
            print(f"  {k:18}{len(pcs):>5}{wr:>7.1f}{fmt_pct(tot):>10}{avg:>+8.2f}")
        print()

    # ============================================================
    # 10. 15 derniers trades pour apercu visuel
    # ============================================================
    print("=" * 70)
    print("15 DERNIERS TRADES CLOS")
    print("=" * 70)
    cur.execute("""
        SELECT u.pattern_kind, u.side, u.pct_gain, u.outcome,
               u.entry_timestamp, u.exit_timestamp,
               s.base || '/' || s.quote as sym,
               u.confluence_score
        FROM unit_trades u
        JOIN symbols s ON s.id = u.symbol_id
        WHERE u.exit_timestamp IS NOT NULL
        ORDER BY u.exit_timestamp DESC LIMIT 15;
    """)
    for r in cur.fetchall():
        icon = "+" if (r['pct_gain'] or 0) > 0 else "-"
        print(f"  {icon} {r['pattern_kind']:25} {r['sym']:11} {r['side']:5} "
              f"pnl={fmt_pct(r['pct_gain'] or 0):>9}  out={str(r['outcome']):14} "
              f"cs={r['confluence_score']:.2f}  closed={r['exit_timestamp']}")

    conn.close()


if __name__ == "__main__":
    main()
