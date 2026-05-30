"""Boucle iterative backfill → analyse → amelioration → repeat.

Strategie : UNE SEULE DB par iteration, filtres virtuels (pas de re-backfill
pour chaque config testee) + analyse MFE/MAE. Chaque iteration utilise
la meilleure config trouvee precedemment comme base.

Phases par iteration :
  A. Reset DB + backfill (N jours, filtres actuels)
  B. Analytics : breakdown + optimize_filters (grid search virtuel)
  C. Analyse MFE/MAE : BE optimal + SL tighness + patterns perdants
  D. Rapport : variables .env recommandees

Usage :
    python scripts/run_loop.py                  # 14 jours, 10 symboles, 2 iterations
    python scripts/run_loop.py --days 7 --quick # 7 jours, 4 symboles
    python scripts/run_loop.py --iterations 3   # 3 boucles d'amelioration
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Force UTF-8 sur stdout/stderr Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ============================================================
# Configuration
# ============================================================

SYMBOLS_FULL = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT",
    "ADA/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT", "DOGE/USDT",
]
SYMBOLS_QUICK = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]


def resolve_universe(name: str) -> list[str]:
    """Selectionne l'univers : "10" / "4" / nom universe (50/100/200/bitget_...)"""
    if name == "10":
        return SYMBOLS_FULL
    if name == "4":
        return SYMBOLS_QUICK
    from app.universe import get_universe
    u = get_universe(name)
    return u if u else SYMBOLS_FULL

# Bars par jour pour les TF supportes
BARS_PER_DAY_MAP = {
    "1m": 60 * 24,    # 1440 bougies/jour - plus precis
    "5m": 12 * 24,    # 288
    "15m": 4 * 24,    # 96
    "1h": 24,
    "4h": 6,
}


# ============================================================
# DB management — reload propre des modules session
# ============================================================

def _set_db_url(db_name: str) -> str:
    url = f"sqlite+aiosqlite:///./{db_name}"
    os.environ["DATABASE_URL"] = url
    return url


async def _reload_session_modules() -> None:
    """Recharge app.config + app.db.session + app.services.continuous_scanner
    pour que le nouveau DATABASE_URL soit pris en compte partout."""
    # 1. Config (re-instancie Settings() avec le nouvel env var)
    if "app.config" in sys.modules:
        importlib.reload(sys.modules["app.config"])

    # 2. Session (re-crée engine avec nouveau settings.database_url)
    if "app.db.session" in sys.modules:
        sm = sys.modules["app.db.session"]
        try:
            await sm.engine.dispose()
        except Exception:
            pass
        importlib.reload(sm)

    # 3. Scanner (re-lie async_session_factory)
    if "app.services.continuous_scanner" in sys.modules:
        importlib.reload(sys.modules["app.services.continuous_scanner"])


async def _reset_db(db_name: str) -> None:
    """Supprime + recrée la DB, recharge les modules sessions."""
    url = _set_db_url(db_name)
    await _reload_session_modules()

    db_file = Path(db_name)
    if db_file.exists():
        try:
            db_file.unlink()
        except PermissionError:
            print(f"  [WARN] Ne peut pas supprimer {db_name}")

    from sqlalchemy.ext.asyncio import create_async_engine
    from app.db.models import Base
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    print(f"  [OK] DB reset → {db_name}")


# ============================================================
# Backfill
# ============================================================

async def _run_backfill(
    symbols: list[str],
    history_bars: int,
    *,
    timeframe: str = "15m",
    min_conf: float = 0.0,
    reject_counter: bool = False,
    require_volume: bool = False,
    breakeven_pct: float = 0.0,
    min_rr: float = 0.0,
    excluded_patterns: list[str] | None = None,
    reject_volume_weak: bool = False,
    trailing_stop_atr_mult: float = 0.0,
) -> dict:
    # Import apres le reload des modules
    from app.services.continuous_scanner import ContinuousScanner, ScanPlan
    from app.services.hypothesis_engine import HypothesisEngine, ConfluenceScorer
    from app.patterns._quality import QualityWrappedDetector
    from app.patterns.channels import ChannelDetector
    from app.patterns.flags import FlagDetector
    from app.patterns.rectangles import RectangleDetector
    from app.patterns.reversal import ReversalDetector
    from app.patterns.triangles import TriangleDetector
    from app.patterns.wedges import WedgeDetector

    # Adapte les params des detecteurs au TF (1m = moves plus petits + window plus large)
    if timeframe == "1m":
        reversal = ReversalDetector(
            window_bars=360,           # 6h de structure
            twin_tol_pct=0.008,        # tops/bottoms dans 0.8%
            min_neck_distance_pct=0.008,
            min_head_prominence_pct=0.005,
        )
        triangle = TriangleDetector(window_bars=360, min_height_pct=0.005)
        rectangle = RectangleDetector(window_bars=360, min_range_pct=0.005)
        channel = ChannelDetector(window_bars=360)
        wedge = WedgeDetector(window_bars=360, min_height_pct=0.005)
        flag = FlagDetector()  # pas de window_bars
    elif timeframe == "5m":
        reversal = ReversalDetector(
            window_bars=240, twin_tol_pct=0.012, min_neck_distance_pct=0.012,
        )
        triangle = TriangleDetector(window_bars=240)
        rectangle = RectangleDetector(window_bars=240)
        channel = ChannelDetector(window_bars=240)
        wedge = WedgeDetector(window_bars=240)
        flag = FlagDetector()
    else:
        reversal = ReversalDetector()
        triangle = TriangleDetector()
        rectangle = RectangleDetector()
        channel = ChannelDetector()
        wedge = WedgeDetector()
        flag = FlagDetector()

    plan = ScanPlan(
        symbols=symbols,
        timeframes=[timeframe],
        candles_per_fetch=history_bars + 60,
        # Wrappe chaque detecteur pour appliquer pre-trend + RSI alignment
        detectors=[
            QualityWrappedDetector(triangle),
            QualityWrappedDetector(rectangle),
            QualityWrappedDetector(channel),
            QualityWrappedDetector(wedge),
            QualityWrappedDetector(flag),
            QualityWrappedDetector(reversal),  # reversal a sa validation interne, skip
        ],
    )
    scanner = ContinuousScanner(plan=plan)
    # Expiry adapte au TF : 40 bars = 10h en 15m mais 40min en 1m ! On scale.
    expiry_bars_per_tf = {"1m": 240, "5m": 96, "15m": 40, "1h": 24, "4h": 12}
    expiry = expiry_bars_per_tf.get(timeframe, 40)

    scanner._engine = HypothesisEngine(
        confluence_scorer=ConfluenceScorer(),
        min_confluence_score=min_conf,
        reject_trend_counter=reject_counter,
        require_volume_expansion=require_volume,
        require_volume_weak_reject=reject_volume_weak,
        breakeven_trigger_pct=breakeven_pct,
        min_rr_ratio=min_rr,
        excluded_patterns=tuple(excluded_patterns or ()),
        trailing_stop_atr_mult=trailing_stop_atr_mult,
        expiry_bars=expiry,
    )
    try:
        result = await scanner.backfill(
            history_bars=history_bars,
            bars_per_step=1,
        )
    finally:
        await scanner.stop()
    return result


# ============================================================
# Analytics
# ============================================================

async def _compute_stats(db_url: str) -> dict:
    """Calcule les stats directement sur la DB specifiee (evite le cache module)."""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from app.services.analytics import compute_breakdowns, optimize_filters

    engine = create_async_engine(db_url)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with Session() as session:
            return {
                "breakdowns": await compute_breakdowns(session),
                "optimize": await optimize_filters(session, top_n=15),
            }
    finally:
        await engine.dispose()


# ============================================================
# MFE/MAE analysis
# ============================================================

async def _mfe_analysis(db_url: str) -> dict:
    """Analyse MFE/MAE, retourne recommandations."""
    from datetime import timezone, timedelta, UTC, datetime
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from app.db.models import Hypothesis, Symbol, Timeframe, UnitTrade
    from app.ingestion.ccxt_fetcher import CCXTFetcher
    from app.services.period_utils import timeframe_bar_seconds

    def _utc(dt):
        if dt is None:
            return None
        if getattr(dt, "tzinfo", None) is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    engine = create_async_engine(db_url)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    fetcher = CCXTFetcher("binance")

    pct_target: dict[str, list[float]] = defaultdict(list)   # par pattern
    pct_sl: dict[str, list[float]] = defaultdict(list)
    mfe_vals: dict[str, list[float]] = defaultdict(list)
    outcomes: dict[str, list[str]] = defaultdict(list)
    gains: dict[str, list[float]] = defaultdict(list)

    try:
        async with Session() as session:
            q = (
                select(UnitTrade, Symbol, Timeframe)
                .join(Symbol, Symbol.id == UnitTrade.symbol_id)
                .join(Timeframe, Timeframe.id == UnitTrade.timeframe_id)
                .where(UnitTrade.exit_price.is_not(None))
            )
            rows = (await session.execute(q)).all()
            if not rows:
                return {}

            # Min timestamp pour couvrir la periode
            min_ts = None
            for ut, s, tf in rows:
                ts = _utc(ut.entry_timestamp)
                if ts and (min_ts is None or ts < min_ts):
                    min_ts = ts
            since_ms = int(min_ts.timestamp() * 1000) - 3_600_000 if min_ts else None

            # Fetch OHLCV
            pairs = {(f"{s.base}/{s.quote}", tf.code) for _, s, tf in rows}
            cache: dict = {}
            fetch_failures: list[tuple[str, str, str]] = []
            for symbol, tfc in sorted(pairs):
                last_err: Exception | None = None
                for attempt in range(3):
                    try:
                        if since_ms:
                            bms = max(1, int(timeframe_bar_seconds(tfc) * 1000))
                            now_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
                            want = min(5000, max(500, (now_ms - since_ms) // bms + 200))
                            r = await fetcher.fetch_ohlcv(symbol, tfc, since_ms=since_ms, limit=int(want))
                        else:
                            r = await fetcher.fetch_ohlcv(symbol, tfc, limit=1500)
                        if r:
                            cache[(symbol, tfc)] = [
                                (_utc(x.ts_open), float(x.high), float(x.low), float(x.close))
                                for x in r
                            ]
                        last_err = None
                        break
                    except Exception as e:
                        last_err = e
                        # Backoff exponentiel : 1s, 2s, 4s
                        await asyncio.sleep(2 ** attempt)
                if last_err is not None:
                    fetch_failures.append((symbol, tfc, str(last_err)))
            if fetch_failures:
                print(f"  [WARN] {len(fetch_failures)} fetch OHLCV en echec apres 3 tentatives :")
                for sym, tfc, err in fetch_failures[:5]:
                    print(f"    - {sym} {tfc}: {err[:120]}")

            for ut, s, tf in rows:
                h = await session.get(Hypothesis, ut.hypothesis_id)
                if h is None or h.triggered_price is None or ut.exit_timestamp is None:
                    continue
                entry = float(h.triggered_price)
                target = float(h.target_price)
                inv = float(h.invalidation_price)
                side = h.side
                pat = h.pattern_kind

                key = (f"{s.base}/{s.quote}", tf.code)
                cached = cache.get(key, [])
                ts_start = _utc(ut.entry_timestamp)
                ts_end = _utc(ut.exit_timestamp)
                bars = [r for r in cached
                        if ts_start and ts_end and ts_start <= r[0] <= ts_end + timedelta(minutes=1)]

                if not bars:
                    g = float(ut.pct_gain or 0.0)
                    mfe = max(0.0, g)
                    mae = abs(min(0.0, g))
                else:
                    highest = max(b[1] for b in bars)
                    lowest = min(b[2] for b in bars)
                    if side == "LONG":
                        mfe = max(0.0, (highest / entry - 1.0) * 100.0)
                        mae = abs(min(0.0, (lowest / entry - 1.0) * 100.0))
                    else:
                        mfe = max(0.0, (entry / lowest - 1.0) * 100.0) if lowest > 0 else 0.0
                        mae = abs((highest / entry - 1.0) * 100.0)

                td = abs(target - entry) / entry * 100.0 if entry > 0 else 0.0
                sd = abs(inv - entry) / entry * 100.0 if entry > 0 else 0.0
                pt = mfe / td if td > 0 else 0.0
                ps = mae / sd if sd > 0 else 0.0

                pct_target[pat].append(pt)
                pct_sl[pat].append(ps)
                mfe_vals[pat].append(mfe)
                outcomes[pat].append(ut.outcome or "UNKNOWN")
                gains[pat].append(float(ut.pct_gain or 0.0))

    finally:
        await fetcher.close()
    await engine.dispose()

    if not gains:
        return {}

    # --- Recommandations ---
    reco: dict = {}

    # BE trigger optimal (global)
    all_stopped_pt = [
        pt for pat, pts in pct_target.items()
        for i, pt in enumerate(pts)
        if i < len(outcomes[pat]) and outcomes[pat][i] == "STOPPED"
    ]
    if all_stopped_pt:
        best_be, best_saved = 0.0, 0
        for be in [0.2, 0.3, 0.5, 0.7]:
            saved = sum(1 for pt in all_stopped_pt if pt >= be)
            if saved > best_saved:
                best_saved = saved
                best_be = be
        saved_ratio = best_saved / len(all_stopped_pt)
        if saved_ratio > 0.15:
            reco["breakeven_trigger_pct"] = best_be
            reco["be_saves_pct"] = round(saved_ratio * 100, 1)
            reco["stopped_count"] = len(all_stopped_pt)

    # Patterns perdants
    bad: dict[str, dict] = {}
    for pat, gs in gains.items():
        if len(gs) < 3:
            continue
        avg = mean(gs)
        wr = sum(1 for g in gs if g > 0) / len(gs)
        if avg < -0.5:
            bad[pat] = {
                "avg_gain": round(avg, 3),
                "winrate": round(wr * 100, 1),
                "n": len(gs),
            }
    if bad:
        reco["bad_patterns"] = bad

    # SL trop serre : winners qui ont failli toucher le SL
    tight: dict[str, float] = {}
    for pat, pts in pct_sl.items():
        outs = outcomes.get(pat, [])
        winner_sl = [pts[i] for i, o in enumerate(outs)
                     if o == "TARGET_HIT" and i < len(pts)]
        if len(winner_sl) >= 3 and mean(winner_sl) > 0.65:
            tight[pat] = round(mean(winner_sl) * 100, 1)
    if tight:
        reco["tight_sl_patterns"] = tight

    return reco


# ============================================================
# Main loop
# ============================================================

def _seg(s: dict) -> str:
    prefix = "+" if s.get("cumul_compound_pct", 0) >= 0 else ""
    return (
        f"N={s['count']:4}  win={s['win_rate_pct']:5.1f}%  "
        f"exp={s['expectancy_pct']:+6.3f}%  cumul={prefix}{s['cumul_compound_pct']:+8.2f}%"
    )


async def main(args) -> None:
    from app.logging_setup import setup_logging
    setup_logging(level_console=logging.WARNING, disable_file=True)

    if args.universe:
        symbols = resolve_universe(args.universe)
    elif args.quick:
        symbols = SYMBOLS_QUICK
    else:
        symbols = SYMBOLS_FULL
    if args.tf not in BARS_PER_DAY_MAP:
        print(f"TF {args.tf} non supporte. Choix: {list(BARS_PER_DAY_MAP.keys())}")
        return
    bars_per_day = BARS_PER_DAY_MAP[args.tf]
    history_bars = args.days * bars_per_day

    print("=" * 72)
    print(f"BOUCLE OPTIMISATION : {args.days} jours × {len(symbols)} symboles × {args.tf}")
    print(f"  {history_bars} bougies | {args.iterations} iteration(s)")
    print("=" * 72)

    # Config initiale RAW (zero filtre) - le grid search trouvera la meilleure
    # config a appliquer en iter 2
    cur_cfg: dict = {
        "min_conf": 0.0,
        "reject_counter": False,
        "require_volume": False,
        "breakeven_pct": 0.0,
        "min_rr": 0.0,
        "excluded_patterns": [],
        "reject_volume_weak": False,
        "trailing_stop_atr_mult": 0.0,
    }

    # None = aucune iteration n'a produit assez de trades pour calculer un compound.
    # On ne reutilise plus la sentinelle -9999.0 (confondue avec un vrai resultat dans le rapport).
    best_compound: float | None = None
    best_env_vars: list[str] = []
    all_recos: list[str] = []
    iterations_with_data = 0
    iterations_skipped: list[tuple[int, int]] = []  # (iteration, trade_count)

    for it in range(1, args.iterations + 1):
        db_name = f"loop_iter{it}.db"
        db_url = f"sqlite+aiosqlite:///./{db_name}"

        print(f"\n{'-'*72}")
        print(f"ITERATION {it}/{args.iterations}  →  {db_name}")
        print(f"  Config: score>={cur_cfg['min_conf']}  "
              f"reject_ctr={cur_cfg['reject_counter']}  "
              f"require_vol={cur_cfg['require_volume']}  "
              f"BE={cur_cfg['breakeven_pct']:.0%}")
        print(f"{'-'*72}")

        # A. Reset + backfill
        print(f"\n[A] Reset + backfill ({history_bars} bougies de {args.tf}) ...")
        await _reset_db(db_name)
        bf = await _run_backfill(symbols, history_bars, timeframe=args.tf, **cur_cfg)
        print(f"    {bf.get('total_steps', 0)} steps | "
              f"{bf.get('total_patterns_detected', 0)} patterns")

        # B. Stats
        print(f"\n[B] Analytics ...")
        stats = await _compute_stats(db_url)
        base = stats["breakdowns"]["overall"][0]
        print(f"    Baseline  : {_seg(base)}")

        if base["count"] < 10:
            print(f"    [!] Seulement {base['count']} trades — augmente --days ou utilise plus de symboles")
            iterations_skipped.append((it, base["count"]))
            if it == args.iterations:
                break
            # Prochaine iteration avec les memes params
            continue
        iterations_with_data += 1

        # Affiche top 5 patterns
        by_pat = stats["breakdowns"].get("by_pattern", [])
        if by_pat:
            print(f"    Top patterns par expectancy :")
            for seg in by_pat[:5]:
                print(f"      {seg['label']:<26} {_seg(seg)}")

        # C. Grid search virtuel
        opt = stats["optimize"]
        new_cfg = dict(cur_cfg)

        if opt.get("top_configs"):
            best_virt = opt["top_configs"][0]
            cfg_v = best_virt["config"]
            excluded = cfg_v.get("excluded_patterns", [])
            print(f"\n[C] Meilleure config virtuelle :")
            print(f"    score>={cfg_v['min_confluence_score']}  "
                  f"reject={cfg_v['reject_tags']}  require={cfg_v['required_tags']}")
            if excluded:
                print(f"    EXCLUDE patterns : {excluded}")
            if opt.get("losing_patterns_detected"):
                print(f"    Patterns perdants detectes : {opt['losing_patterns_detected']}")
            print(f"    Simul : {_seg(best_virt)}")
            new_cfg["min_conf"] = float(cfg_v["min_confluence_score"])
            new_cfg["reject_counter"] = "trend_counter" in cfg_v.get("reject_tags", [])
            new_cfg["require_volume"] = "volume_expansion" in cfg_v.get("required_tags", [])
            new_cfg["reject_volume_weak"] = "volume_weak" in cfg_v.get("reject_tags", [])
            new_cfg["excluded_patterns"] = list(excluded)
            # Active le trailing stop ATR en iter 2 pour faire courir les winners
            new_cfg["trailing_stop_atr_mult"] = 2.0
        else:
            print(f"\n[C] Grid search : pas assez de trades (min 15)")

        # D. MFE/MAE
        print(f"\n[D] Analyse MFE/MAE ...")
        mfe_reco = await _mfe_analysis(db_url)
        if mfe_reco:
            be = float(mfe_reco.get("breakeven_trigger_pct", 0.0))
            if be > 0:
                saves = mfe_reco.get("be_saves_pct", 0)
                stopped = mfe_reco.get("stopped_count", 0)
                print(f"    INFO BE@{be:.0%} sauverait {saves:.0f}% des {stopped} stops (theorique)")
                print(f"    NOTE: BE non auto-applique (peut convertir winners en BE-exits)")
            if mfe_reco.get("bad_patterns"):
                print(f"    → Patterns perdants : {list(mfe_reco['bad_patterns'].keys())}")
            if mfe_reco.get("tight_sl_patterns"):
                print(f"    → SL trop serre : {list(mfe_reco['tight_sl_patterns'].keys())}")
        else:
            print("    Pas assez de données pour MFE/MAE (pas de trades avec OHLCV)")

        # E. Simulation de la meilleure config sur les trades existants
        if opt.get("top_configs"):
            virt_compound = best_virt["cumul_compound_pct"]
            if best_compound is None or virt_compound > best_compound:
                best_compound = virt_compound
                # Construit les env vars recommandées
                env_vars = [f"MIN_CONFLUENCE_SCORE={new_cfg['min_conf']}"]
                if new_cfg["reject_counter"]:
                    env_vars.append("REJECT_TREND_COUNTER=true")
                if new_cfg["require_volume"]:
                    env_vars.append("REQUIRE_VOLUME_EXPANSION=true")
                if new_cfg["breakeven_pct"] > 0:
                    env_vars.append(f"BREAKEVEN_TRIGGER_PCT={new_cfg['breakeven_pct']}")
                best_env_vars = env_vars

            # Recommendations textuelles
            if mfe_reco.get("bad_patterns"):
                for pat, info in mfe_reco["bad_patterns"].items():
                    all_recos.append(
                        f"  Pattern {pat}: avg={info['avg_gain']:+.2f}%  win={info['winrate']:.0f}%  "
                        f"N={info['n']} → Filtrer ou augmenter min_conf"
                    )
            if mfe_reco.get("tight_sl_patterns"):
                for pat, pct in mfe_reco["tight_sl_patterns"].items():
                    all_recos.append(
                        f"  Pattern {pat}: SL effleuré à {pct:.0f}% chez les winners "
                        f"→ Elargir SL (buffer ATR)"
                    )

        # Prepare prochaine iteration avec la config amelioree
        cur_cfg = new_cfg
        print(f"\n    Config pour prochaine iteration : {cur_cfg}")

        # Nettoyage DB d'iteration (garde seulement la derniere)
        if it < args.iterations:
            db_path = Path(db_name)
            if db_path.exists():
                try:
                    db_path.unlink()
                except Exception:
                    pass

    # ============================================================
    # Résumé final
    # ============================================================
    print("\n" + "=" * 72)
    print("RÉSULTAT FINAL")
    print("=" * 72)
    if best_compound is None:
        print(f"  Meilleur compound virtuel : N/A (donnees insuffisantes)")
        if iterations_skipped:
            print(f"  Iterations sans assez de trades (<10) :")
            for it_idx, n in iterations_skipped:
                print(f"    - iter {it_idx} : {n} trades")
        print(f"  Conseils : augmente --days, ajoute des symboles, "
              f"ou desactive les filtres dans cur_cfg.")
    else:
        print(f"  Meilleur compound virtuel : {best_compound:+.2f}% "
              f"(sur {iterations_with_data} iteration(s) exploitable(s))")

    print(f"\n  Variables .env recommandées :")
    if best_env_vars:
        for v in best_env_vars:
            print(f"    {v}")
    else:
        print("    (pas assez de données pour optimiser)")

    if all_recos:
        print(f"\n  Recommendations pattern :")
        for r in all_recos[:10]:
            print(r)

    # Sauvegarde
    reco_file = ROOT / "recommendations.txt"
    with open(reco_file, "w", encoding="utf-8") as f:
        f.write(f"# Recommendations — run_loop.py\n")
        f.write(f"# Jours: {args.days}  Symboles: {len(symbols)}  "
                f"Iterations: {args.iterations}\n\n")
        if best_compound is None:
            f.write("Meilleur compound virtuel : N/A (donnees insuffisantes)\n")
            f.write(f"Iterations exploitables : {iterations_with_data} / {args.iterations}\n")
            if iterations_skipped:
                f.write("Iterations sans assez de trades (<10) :\n")
                for it_idx, n in iterations_skipped:
                    f.write(f"  - iter {it_idx} : {n} trades\n")
            f.write("\nAucune variable .env recommandee : relancer avec --days plus eleve "
                    "ou plus de symboles.\n")
        else:
            f.write(f"Meilleur compound virtuel : {best_compound:+.2f}% "
                    f"(sur {iterations_with_data} iteration(s) exploitable(s))\n\n")
            f.write("Variables .env a appliquer :\n")
            for v in best_env_vars:
                f.write(f"  {v}\n")
        if all_recos:
            f.write("\nRecommendations patterns :\n")
            for r in all_recos:
                f.write(f"{r}\n")

    print(f"\n  Recommendations sauvegardées → {reco_file.name}")
    print("\n  Prochaines etapes :")
    print("  1. Applique les variables ci-dessus dans .env")
    print("  2. Lance start.bat pour redemarrer le scanner avec la config optimisee")
    print("  3. Apres 24-48h de scan en live, compare les resultats")


def parse_args():
    p = argparse.ArgumentParser(description="Boucle backfill→analyse→optimisation")
    p.add_argument("--days", type=int, default=14,
                   help="Nombre de jours d'historique (defaut: 14)")
    p.add_argument("--iterations", type=int, default=2,
                   help="Nombre de boucles (defaut: 2)")
    p.add_argument("--quick", action="store_true",
                   help="Seulement 4 symboles (BTC/ETH/SOL/XRP) — plus rapide")
    p.add_argument("--tf", default="15m", choices=["1m", "5m", "15m", "1h", "4h"],
                   help="Timeframe pour le scan (defaut: 15m, 1m = plus precis)")
    p.add_argument("--universe", default=None,
                   help="Univers (4, 10, 50, 100, 200, bitget_futures_top100, bitget_futures_top200, ...)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # IMPORTANT: set DATABASE_URL AVANT tout import app
    _set_db_url(f"loop_iter1.db")
    asyncio.run(main(args))
