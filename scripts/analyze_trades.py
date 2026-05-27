"""Diagnostic profond des trades pour proposer des MODIFICATIONS aux patterns.

Charge la DB (par defaut analyseur_backtest.db via ANALYZE_DB),
recupere l'OHLCV de chaque trade depuis son entry_timestamp,
calcule MFE/MAE et autres metriques, puis agrege par pattern
et propose des changements concrets.

Usage :
    ANALYZE_DB=analyseur_backtest_p1.db python scripts/analyze_trades.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, timezone
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_DB = os.environ.get("ANALYZE_DB", "analyseur_backtest.db")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///./{_DB}"


@dataclass
class TradeAnalysis:
    pattern: str
    side: str
    outcome: str
    pct_gain: float
    entry_price: float
    target_price: float
    invalidation_price: float
    mfe_pct: float                  # max favorable excursion (signe dans notre sens)
    mae_pct: float                  # max adverse excursion (signe dans notre sens, negatif)
    pct_of_target_reached: float    # MFE / abs(target - entry)
    pct_of_sl_reached: float        # |MAE| / abs(invalidation - entry)
    bars_in_trade: int


def _utc(dt):
    """Normalise un datetime en UTC tz-aware."""
    if dt is None:
        return None
    if getattr(dt, "tzinfo", None) is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


async def _fetch_ohlcv_cache(
    fetcher,
    symbols_tfs: set[tuple[str, str]],
    since_ms: int | None = None,
) -> dict:
    """Cache OHLCV par (symbol, tf).

    Si ``since_ms`` est fourni, fetch depuis ce timestamp jusqu'a maintenant
    pour couvrir la periode du backtest. Sinon, fetche les 1500 dernieres bougies.
    """
    from app.services.period_utils import timeframe_bar_seconds
    from datetime import datetime

    cache = {}
    for symbol, tf in sorted(symbols_tfs):
        try:
            if since_ms is not None:
                bar_ms = max(1, int(timeframe_bar_seconds(tf) * 1000))
                now_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
                # +200 bougies de marge
                want = min(5000, max(500, (now_ms - since_ms) // bar_ms + 200))
                rows = await fetcher.fetch_ohlcv(
                    symbol, tf, since_ms=since_ms, limit=int(want)
                )
            else:
                rows = await fetcher.fetch_ohlcv(symbol, tf, limit=1500)
            if rows:
                cache[(symbol, tf)] = [
                    (_utc(r.ts_open), float(r.high), float(r.low), float(r.close))
                    for r in rows
                ]
                print(f"  {symbol} {tf}: {len(rows)} bougies chargées")
        except Exception as exc:
            print(f"  [WARN] OHLCV {symbol} {tf}: {exc}")
    return cache


def _window(cache_rows: list, t_start, t_end) -> list:
    """Filtre les bougies dans la fenetre [t_start, t_end]."""
    ts = _utc(t_start)
    te = _utc(t_end)
    if ts is None or te is None:
        return []
    return [r for r in cache_rows if ts <= r[0] <= te]


async def analyze():
    from datetime import timedelta

    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    from app.db.models import Hypothesis, Symbol, Timeframe, UnitTrade
    from app.ingestion.ccxt_fetcher import CCXTFetcher

    engine = create_async_engine(os.environ["DATABASE_URL"])
    Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    analyses: list[TradeAnalysis] = []
    trigger_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    fetcher = CCXTFetcher("binance")
    ohlcv_cache: dict = {}

    try:
        async with Session() as session:
            # 1. Hypotheses : trigger rate par pattern
            hyps = (await session.execute(select(Hypothesis))).scalars().all()
            for h in hyps:
                trigger_stats[h.pattern_kind][h.state] += 1

            # 2. Trades cloturés
            q = (
                select(UnitTrade, Symbol, Timeframe)
                .join(Symbol, Symbol.id == UnitTrade.symbol_id)
                .join(Timeframe, Timeframe.id == UnitTrade.timeframe_id)
                .where(UnitTrade.exit_price.is_not(None))
            )
            rows = (await session.execute(q)).all()

            if not rows:
                print("Aucun trade cloture dans la DB.")
                return [], trigger_stats

            # Trouve le timestamp minimum pour couvrir toute la periode du backtest
            min_entry_ts = None
            for ut, s, tf in rows:
                if ut.entry_timestamp:
                    ts_utc = _utc(ut.entry_timestamp)
                    if ts_utc and (min_entry_ts is None or ts_utc < min_entry_ts):
                        min_entry_ts = ts_utc

            since_ms: int | None = None
            if min_entry_ts:
                # 1 heure de marge avant le premier trade
                since_ms = int(min_entry_ts.timestamp() * 1000) - 3_600_000
                from datetime import datetime
                print(f"  Periode couverte depuis : {min_entry_ts.strftime('%Y-%m-%d %H:%M')} UTC")

            # Identifie les (symbol, tf) uniques et fetch en bulk
            pairs = {(f"{s.base}/{s.quote}", tf.code) for _, s, tf in rows}
            print(f"Fetching OHLCV pour {len(pairs)} pairs...")
            ohlcv_cache = await _fetch_ohlcv_cache(fetcher, pairs, since_ms=since_ms)

            skipped_no_hyp = 0
            skipped_no_bars = 0
            processed = 0

            for ut, s, tf in rows:
                h = await session.get(Hypothesis, ut.hypothesis_id)
                if h is None or h.triggered_price is None:
                    skipped_no_hyp += 1
                    continue

                entry = float(h.triggered_price)
                target = float(h.target_price)
                invalidation = float(h.invalidation_price)
                side = h.side

                t_start = ut.entry_timestamp
                t_end = ut.exit_timestamp
                if t_end is None:
                    skipped_no_hyp += 1
                    continue

                symbol_str = f"{s.base}/{s.quote}"
                key = (symbol_str, tf.code)
                cached = ohlcv_cache.get(key, [])
                bars = _window(cached, t_start, t_end + timedelta(minutes=1))

                if not bars:
                    # Fallback : utilise le pct_gain stocke
                    pct = float(ut.pct_gain or 0.0)
                    mfe_pct = max(0.0, pct)
                    mae_pct = min(0.0, pct)
                    bars_in_trade = 0
                    skipped_no_bars += 1
                else:
                    highest = max(b[1] for b in bars)
                    lowest = min(b[2] for b in bars)
                    if side == "LONG":
                        mfe_pct = (highest / entry - 1.0) * 100.0
                        mae_pct = (lowest / entry - 1.0) * 100.0
                    else:
                        mfe_pct = (entry / lowest - 1.0) * 100.0 if lowest > 0 else 0.0
                        mae_pct = -(highest / entry - 1.0) * 100.0
                    bars_in_trade = len(bars)

                # % distance target / SL atteinte
                target_dist = abs(target - entry) / entry * 100.0 if entry > 0 else 0.0
                sl_dist = abs(invalidation - entry) / entry * 100.0 if entry > 0 else 0.0
                pct_target = (mfe_pct / target_dist) if target_dist > 0 else 0.0
                pct_sl = (abs(mae_pct) / sl_dist) if sl_dist > 0 else 0.0

                analyses.append(TradeAnalysis(
                    pattern=h.pattern_kind,
                    side=side,
                    outcome=ut.outcome or "UNKNOWN",
                    pct_gain=float(ut.pct_gain or 0.0),
                    entry_price=entry,
                    target_price=target,
                    invalidation_price=invalidation,
                    mfe_pct=mfe_pct,
                    mae_pct=mae_pct,
                    pct_of_target_reached=pct_target,
                    pct_of_sl_reached=pct_sl,
                    bars_in_trade=bars_in_trade,
                ))
                processed += 1

            print(f"\nTrades: {processed} analyses, {skipped_no_hyp} sans hyp, "
                  f"{skipped_no_bars} sans OHLCV (fallback pct_gain)")

    finally:
        await fetcher.close()
    await engine.dispose()
    return analyses, trigger_stats


def pct_in_bucket(values: list[float], buckets: list[float]) -> dict[str, float]:
    """Retourne le % de valeurs dans chaque seau cumulatif."""
    out = {}
    if not values:
        return {f">={b}": 0.0 for b in buckets}
    n = len(values)
    for b in buckets:
        out[f">={b}"] = sum(1 for v in values if v >= b) / n * 100.0
    return out


def main():
    analyses, trigger_stats = asyncio.run(analyze())

    if not analyses:
        print("Aucun trade clôturé trouvé. Lance d'abord scripts/backtest_2weeks.py")
        return

    print("=" * 80)
    print(f"DIAGNOSTIC SUR {len(analyses)} TRADES CLOTURÉS")
    print("=" * 80)

    # ============================================================
    # 1. Trigger rate par pattern
    # ============================================================
    print("\n[1] TRIGGER RATE par pattern (hypotheses creees -> triggered)")
    print("-" * 80)
    print(f"{'Pattern':<32}{'Total':>8}{'Trig%':>8}{'Win%':>8}{'Invalid%':>10}{'Expired%':>10}")
    for pat, states in sorted(trigger_stats.items()):
        total = sum(states.values())
        triggered = (states.get("TRIGGERED", 0)
                     + states.get("TARGET_HIT", 0)
                     + states.get("STOPPED", 0))
        won = states.get("TARGET_HIT", 0)
        invalid = states.get("INVALIDATED", 0)
        expired = states.get("EXPIRED", 0)
        trig_pct = triggered / total * 100 if total else 0
        win_pct = won / triggered * 100 if triggered else 0
        inv_pct = invalid / total * 100 if total else 0
        exp_pct = expired / total * 100 if total else 0
        print(f"{pat:<32}{total:>8}{trig_pct:>7.0f}%{win_pct:>7.0f}%"
              f"{inv_pct:>9.0f}%{exp_pct:>9.0f}%")

    # ============================================================
    # 2. MFE/MAE par pattern × outcome
    # ============================================================
    print("\n[2] MFE/MAE par pattern × outcome")
    print("-" * 80)
    print(f"{'Pattern':<24}{'Outcome':<14}{'N':>5}{'avg pct':>9}"
          f"{'avg MFE':>9}{'avg MAE':>9}{'%target':>9}{'%sl':>7}")
    by_p_o: dict[tuple[str, str], list[TradeAnalysis]] = defaultdict(list)
    for a in analyses:
        by_p_o[(a.pattern, a.outcome)].append(a)
    for (pat, out), items in sorted(by_p_o.items()):
        avg_g = mean(a.pct_gain for a in items)
        avg_mfe = mean(a.mfe_pct for a in items)
        avg_mae = mean(a.mae_pct for a in items)
        avg_pct_t = mean(a.pct_of_target_reached for a in items) * 100
        avg_pct_s = mean(a.pct_of_sl_reached for a in items) * 100
        print(f"{pat:<24}{out:<14}{len(items):>5}{avg_g:>+8.2f}%"
              f"{avg_mfe:>+8.2f}%{avg_mae:>+8.2f}%{avg_pct_t:>8.0f}%{avg_pct_s:>6.0f}%")

    # ============================================================
    # 3. Distribution MFE des STOPPED (combien de BE trigger utile?)
    # ============================================================
    stopped = [a for a in analyses if a.outcome == "STOPPED"]
    targets = [a for a in analyses if a.outcome == "TARGET_HIT"]

    print("\n[3] ANALYSE BREAKEVEN (STOPPED trades)")
    print("-" * 80)
    if stopped:
        for be_trigger in [0.2, 0.3, 0.5, 0.7]:
            saved = sum(1 for a in stopped if a.pct_of_target_reached >= be_trigger)
            pct = saved / len(stopped) * 100
            print(f"  BE@{be_trigger:.0%} du target : sauverait {saved}/{len(stopped)} "
                  f"STOPPED ({pct:.0f}%)")
        mfes = [a.mfe_pct for a in stopped]
        med_mfe = median(mfes)
        avg_mfe_st = mean(mfes)
        print(f"  MFE median STOPPED: {med_mfe:+.2f}%  avg: {avg_mfe_st:+.2f}%")
        if med_mfe > 0.3:
            print(f"  → Profit fréquent avant le SL : BE~{med_mfe/2:.1f}% recommandé")
    else:
        print("  Aucun STOPPED.")

    print("\n[4] ANALYSE SL (TARGET_HIT trades)")
    print("-" * 80)
    if targets:
        maes = [a.pct_of_sl_reached for a in targets]
        avg_mae_pct = mean(maes) * 100
        med_mae_pct = median(maes) * 100
        print(f"  % du SL effleuré avant target — avg: {avg_mae_pct:.0f}%  "
              f"median: {med_mae_pct:.0f}%")
        close_sl = sum(1 for m in maes if m > 0.8)
        print(f"  Winners qui ont frôlé >80% du SL: {close_sl}/{len(targets)} "
              f"({close_sl/len(targets)*100:.0f}%)")
        if avg_mae_pct > 60:
            print("  → SL trop SERRÉ : wicks tapent le stop, buffer ATR x 0.5 recommandé")
        elif avg_mae_pct < 25:
            print("  → SL peut être resserré (peu de drawdown sur les winners)")
    else:
        print("  Aucun TARGET_HIT.")

    # ============================================================
    # 5. Recommandations par pattern
    # ============================================================
    print("\n[5] RECOMMANDATIONS PAR PATTERN")
    print("-" * 80)
    by_pattern: dict[str, list[TradeAnalysis]] = defaultdict(list)
    for a in analyses:
        by_pattern[a.pattern].append(a)

    reco_lines: list[tuple[float, str]] = []
    for pat, items in sorted(by_pattern.items()):
        if len(items) < 3:
            continue
        avg_g = mean(a.pct_gain for a in items)
        win_rate = sum(1 for a in items if a.pct_gain > 0) / len(items) * 100
        st = [a for a in items if a.outcome == "STOPPED"]
        tg = [a for a in items if a.outcome == "TARGET_HIT"]

        st_recovered_pct = 0.0
        if st:
            st_recovered_pct = (
                sum(1 for a in st if a.pct_of_target_reached >= 0.5) / len(st) * 100
            )

        tg_mae_pct = 0.0
        if tg:
            tg_mae_pct = mean(a.pct_of_sl_reached for a in tg) * 100

        avg_mfe_all = mean(a.mfe_pct for a in items)
        avg_mae_all = mean(a.mae_pct for a in items)

        verdicts: list[str] = []
        if avg_g < -0.5:
            verdicts.append("PERDANT → filtrer ou changer entree")
        elif avg_g < 0:
            verdicts.append("marginalement negatif → tuner")

        if st and st_recovered_pct > 40:
            verdicts.append(f"BE@50% sauverait {st_recovered_pct:.0f}% des stops")

        if tg and tg_mae_pct > 65:
            verdicts.append(f"SL trop serre (MAE={tg_mae_pct:.0f}%) → +buffer")

        if avg_mfe_all < 0.2 and avg_g < 0:
            verdicts.append("MFE faible → detection trop prematuree")

        if not verdicts:
            verdicts = ["OK ✓"]

        rr = 0.0
        if items:
            rrs = []
            for a in items:
                risk = abs(a.entry_price - a.invalidation_price) / a.entry_price * 100
                reward = abs(a.target_price - a.entry_price) / a.entry_price * 100
                if risk > 0:
                    rrs.append(reward / risk)
            if rrs:
                rr = mean(rrs)

        line = (
            f"  {pat:<28} N={len(items):>4}  win={win_rate:>4.0f}%  "
            f"avg={avg_g:+5.2f}%  RR={rr:>4.2f}  "
            f"MFE={avg_mfe_all:+5.2f}%  MAE={avg_mae_all:+5.2f}%"
            f"  →  {' | '.join(verdicts)}"
        )
        reco_lines.append((avg_g, line))

    # Affiche du pire au meilleur
    for _, line in sorted(reco_lines):
        print(line)

    # ============================================================
    # 6. Recommendations .env
    # ============================================================
    print("\n[6] REGLAGES RECOMMANDES")
    print("-" * 80)
    # Calcule le meilleur BE trigger global
    if stopped:
        best_be = 0.0
        best_saved = 0
        for be in [0.2, 0.3, 0.5]:
            saved = sum(1 for a in stopped if a.pct_of_target_reached >= be)
            if saved > best_saved:
                best_saved = saved
                best_be = be
        if best_be > 0 and best_saved / len(stopped) > 0.2:
            print(f"  BREAKEVEN_TRIGGER_PCT={best_be}")

    # Patterns a filtrer (avg_g < -1%)
    bad_patterns = [p for p, it in by_pattern.items()
                    if len(it) >= 3 and mean(a.pct_gain for a in it) < -1.0]
    if bad_patterns:
        print(f"  # Patterns systématiquement perdants (avg < -1%) :")
        for p in sorted(bad_patterns):
            n = len(by_pattern[p])
            g = mean(a.pct_gain for a in by_pattern[p])
            print(f"    {p}: N={n}, avg={g:+.2f}% → envisager DISABLE")


if __name__ == "__main__":
    main()
