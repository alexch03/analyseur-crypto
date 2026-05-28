"""Scanner continu de patterns chartistes : 50 cryptos × 15m / 1h / 4h.

Boucle async qui :
  1. Pour chaque paire (symbol, timeframe), à chaque clôture de bougie,
  2. Fetch les dernières candles via ccxt,
  3. Calcule les swings, lance les détecteurs de patterns enregistrés,
  4. Charge les hypothèses actives, fait avancer le moteur de cycle de vie,
  5. Persiste les nouveautés (hypothèses + unit_trades + scan_run).

Le scanner reste **idempotent** sur la même barre : tant qu'aucune nouvelle bougie
n'a clôturé, ré-exécuter ne crée pas de doublons (dédup via le moteur).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import Exchange
from app.db.session import async_session_factory
from app.ingestion.ccxt_fetcher import CCXTFetcher
from app.market_structure.swings import detect_swings
from app.patterns._quality import QualityWrappedDetector
from app.patterns.channels import ChannelDetector
from app.patterns.flags import FlagDetector
from app.patterns.interfaces import PatternDetector
from app.patterns.rectangles import RectangleDetector
from app.patterns.reversal import ReversalDetector
from app.patterns.triangles import TriangleDetector
from app.patterns.wedges import WedgeDetector
from app.paper.unit_tracker import reconcile_with_engine_step
from app.services.hypothesis_engine import ConfluenceScorer, HypothesisEngine
from app.services.hypothesis_repository import (
    ensure_symbol_id,
    ensure_timeframe_id,
    insert_unit_trade,
    load_active_hypotheses,
    load_open_unit_trades_for_symbol_tf,
    record_scan_run,
    upsert_hypothesis,
)
from app.services.period_utils import timeframe_bar_seconds

logger = logging.getLogger(__name__)

# Fingerprint affiche au demarrage pour verifier qu'on tourne bien le bon code.
_CODE_VERSION = "scanner.v2-tz-fix"


def default_detectors() -> list[PatternDetector]:
    """Detecteurs par defaut wrappes par QualityWrappedDetector.

    Le wrapper applique :
      - Pre-trend context (continuation: meme sens, reversal: sens oppose)
      - RSI alignment (RSI > 50 pour breakout UP, < 50 pour DOWN)

    Patterns reversal (DOUBLE_TOP/BOTTOM, H&S, IHS) ont leur propre validation
    interne avancee (RSI div + volume + body), donc le wrapper les passe-through.
    """
    return [
        QualityWrappedDetector(TriangleDetector()),
        QualityWrappedDetector(RectangleDetector()),
        QualityWrappedDetector(ChannelDetector()),
        QualityWrappedDetector(WedgeDetector()),
        QualityWrappedDetector(FlagDetector()),
        QualityWrappedDetector(ReversalDetector()),
    ]


@dataclass
class ExchangeTarget:
    """Configuration d'un exchange data source (Binance, Bitget, etc.)."""
    exchange_id: str               # "binance", "bitget"
    symbols: list[str]
    timeframes: list[str]


@dataclass
class ScanPlan:
    """Plan de scan : 1 ou plusieurs exchanges, leurs symboles et TF."""
    symbols: list[str]              # gardes pour retro-compat (mono-exchange)
    timeframes: list[str]
    interval_seconds: int = 60
    candles_per_fetch: int = 200
    detectors: list[PatternDetector] = field(default_factory=default_detectors)
    # NEW : si exchange_targets est non vide, scanne CES exchanges en parallele.
    # Sinon, retombe sur l'exchange unique de settings (binance par defaut).
    exchange_targets: list[ExchangeTarget] = field(default_factory=list)


@dataclass
class _LastBarTracker:
    """État interne : dernière barre traitée par (symbol, tf)."""
    last_bar_ts: dict[tuple[str, str], datetime] = field(default_factory=dict)

    def should_process(self, symbol: str, tf: str, latest_close_ts: datetime) -> bool:
        prev = self.last_bar_ts.get((symbol, tf))
        return prev is None or latest_close_ts > prev

    def mark(self, symbol: str, tf: str, ts: datetime) -> None:
        self.last_bar_ts[(symbol, tf)] = ts


class ContinuousScanner:
    """Scanner long-running : appeler ``run()`` (async) pour démarrer la boucle."""

    def __init__(self, plan: ScanPlan | None = None) -> None:
        self._plan = plan or ScanPlan(
            symbols=settings.effective_scan_symbols(),
            timeframes=settings.effective_scan_timeframes(),
            interval_seconds=int(settings.scan_interval_seconds),
        )
        self._fetcher = CCXTFetcher(settings.exchange_id)
        self._engine = HypothesisEngine(
            confluence_scorer=ConfluenceScorer(),
            min_confluence_score=float(settings.min_confluence_score),
            min_rr_ratio=float(settings.min_rr_ratio),
            reject_trend_counter=bool(settings.reject_trend_counter),
            require_volume_expansion=bool(settings.require_volume_expansion),
            breakeven_trigger_pct=float(settings.breakeven_trigger_pct),
        )
        self._tracker = _LastBarTracker()
        self._stop_event = asyncio.Event()

    async def stop(self) -> None:
        self._stop_event.set()
        await self._fetcher.close()

    async def run(self) -> None:
        logger.info(
            "ContinuousScanner [%s]: %d symbols x %d timeframes, every %ds",
            _CODE_VERSION,
            len(self._plan.symbols),
            len(self._plan.timeframes),
            self._plan.interval_seconds,
        )
        try:
            while not self._stop_event.is_set():
                await self._scan_cycle()
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self._plan.interval_seconds
                    )
                except TimeoutError:
                    continue
        finally:
            await self._fetcher.close()

    async def _detect_market_regime(self) -> None:
        """Detecte le regime de marche : BTC 1h + breadth des symbols scannes.

        Appele en debut de chaque cycle. Le regime est ensuite utilise par
        HypothesisEngine pour adapter le filtrage des patterns.
        """
        try:
            from app.services.regime_tracker import get_regime_tracker
            # Fetch BTC 1h pour analyse trend macro (300 bars = 12.5j)
            btc_rows = await self._fetcher.fetch_ohlcv("BTC/USDT", "1h", limit=300)
            if not btc_rows or len(btc_rows) < 50:
                return
            df = _rows_to_df(btc_rows)
            # Breadth : on calcule sur les pairs deja scannees ce cycle
            # (necessite un cache; pour l'instant fallback None -> derive de BTC seul)
            tracker = get_regime_tracker()
            async with async_session_factory() as session:
                regime = await tracker.refresh(session, df, breadth_pct=None)
                await session.commit()
            # Communique au moteur pour adaptation
            if hasattr(self._engine, "set_market_regime"):
                self._engine.set_market_regime(regime)
        except Exception:
            logger.exception("Regime detection failed (continue scan)")

    async def _scan_cycle(self) -> None:
        """Une passe complète sur toutes les paires."""
        # 0. Detecte le regime au debut du cycle pour adapter l'engine
        await self._detect_market_regime()

        ok = 0
        failed = 0
        import time as _t
        t_start = _t.monotonic()
        delay_s = max(0.0, float(settings.scan_pair_delay_ms) / 1000.0)
        for symbol in self._plan.symbols:
            for tf in self._plan.timeframes:
                try:
                    await self._scan_pair(symbol, tf)
                    ok += 1
                except Exception as e:
                    failed += 1
                    logger.exception(
                        "Scan failed for %s %s | error_type=%s | msg=%s",
                        symbol, tf, type(e).__name__, str(e)[:200],
                    )
                if delay_s > 0:
                    await asyncio.sleep(delay_s)
        elapsed = _t.monotonic() - t_start
        logger.info("Cycle done: ok=%d failed=%d in %.1fs (%d pairs, delay=%dms)",
                     ok, failed, elapsed,
                     len(self._plan.symbols) * len(self._plan.timeframes),
                     settings.scan_pair_delay_ms)

    async def scan_once(self) -> dict:
        """Une passe immédiate, retourne un résumé. Idempotent."""
        import time
        t0 = time.monotonic()
        before_state = dict(self._tracker.last_bar_ts)
        self._tracker.last_bar_ts.clear()   # force re-process même si déjà vu
        try:
            await self._scan_cycle()
        finally:
            # On garde l'état mis à jour pour ne pas re-traiter au prochain tick
            pass
        return {
            "elapsed_seconds": round(time.monotonic() - t0, 2),
            "symbols": len(self._plan.symbols),
            "timeframes": len(self._plan.timeframes),
            "pairs_processed": len(self._tracker.last_bar_ts),
        }

    async def backfill(
        self,
        *,
        bars_per_step: int = 1,
        history_bars: int = 250,
        symbols: list[str] | None = None,
        timeframes: list[str] | None = None,
    ) -> dict:
        """Reconstruit l'historique des hypothèses en rejouant les ``history_bars`` dernières
        bougies bar-par-bar. Bcp plus lent qu'un scan ponctuel mais produit un cumul %
        immédiatement (winrate, best/worst, etc.).

        ``bars_per_step`` > 1 saute des bougies (gain de vitesse, perd un peu de précision).
        """
        import time
        t0 = time.monotonic()
        target_symbols = symbols or self._plan.symbols
        target_tfs = timeframes or self._plan.timeframes
        total_steps = 0
        total_patterns = 0
        for symbol in target_symbols:
            for tf in target_tfs:
                try:
                    steps, patterns = await self._backfill_pair(
                        symbol, tf, history_bars=history_bars, bars_per_step=bars_per_step
                    )
                    total_steps += steps
                    total_patterns += patterns
                except Exception:
                    logger.exception("Backfill failed for %s %s", symbol, tf)
                await asyncio.sleep(0.1)
        return {
            "elapsed_seconds": round(time.monotonic() - t0, 1),
            "symbols": len(target_symbols),
            "timeframes": len(target_tfs),
            "history_bars": history_bars,
            "bars_per_step": bars_per_step,
            "total_steps": total_steps,
            "total_patterns_detected": total_patterns,
        }

    async def _backfill_pair(
        self, symbol: str, tf: str, *, history_bars: int, bars_per_step: int
    ) -> tuple[int, int]:
        rows = await self._fetcher.fetch_ohlcv(symbol, tf, limit=history_bars + 50)
        if not rows or len(rows) < 60:
            return 0, 0
        df_full = _rows_to_df(rows)
        n = len(df_full)
        # Démarre à 50 bars pour avoir assez d'historique pour les détecteurs
        start_idx = max(50, n - history_bars)

        steps = 0
        patterns_total = 0
        async with async_session_factory() as session:
            exchange_id = await _ensure_exchange_id(session, settings.exchange_id)
            symbol_id = await ensure_symbol_id(session, exchange_id, symbol)
            timeframe_id = await ensure_timeframe_id(session, tf)

            for i in range(start_idx, n, bars_per_step):
                df_slice = df_full.iloc[: i + 1].reset_index(drop=True)
                swings = detect_swings(df_slice, left=3, right=3)
                patterns = []
                for det in self._plan.detectors:
                    patterns.extend(det.detect(df_slice, swings, symbol=symbol, timeframe=tf))
                patterns_total += len(patterns)

                existing = await load_active_hypotheses(session, symbol_id, timeframe_id)
                result = self._engine.step(df_slice, patterns, existing)

                for h in result.created + result.updated:
                    await upsert_hypothesis(
                        session, h, symbol_id=symbol_id, timeframe_id=timeframe_id
                    )

                open_trades = await load_open_unit_trades_for_symbol_tf(
                    session, symbol_id, timeframe_id
                )
                open_trades = [replace(t, symbol=symbol, timeframe=tf) for t in open_trades]
                all_h = result.created + result.updated
                _, newly_closed, newly_opened = reconcile_with_engine_step(open_trades, all_h)
                for t in newly_opened + newly_closed:
                    await insert_unit_trade(
                        session, t, symbol_id=symbol_id, timeframe_id=timeframe_id
                    )
                steps += 1
                if steps % 50 == 0:
                    await session.commit()

            await session.commit()

        # Mémorise le dernier ts pour ne pas re-traiter ces bougies en live
        latest_close = df_full["timestamp"].iloc[-1].to_pydatetime()
        self._tracker.mark(symbol, tf, latest_close)
        logger.info("Backfill %s %s: %d steps, %d patterns", symbol, tf, steps, patterns_total)
        return steps, patterns_total

    async def _scan_pair(self, symbol: str, tf: str) -> None:
        rows = await self._fetcher.fetch_ohlcv(
            symbol, tf, limit=self._plan.candles_per_fetch
        )
        if not rows or len(rows) < 30:
            return

        df = _rows_to_df(rows)
        latest_close = df["timestamp"].iloc[-1].to_pydatetime()

        if not self._tracker.should_process(symbol, tf, latest_close):
            return
        # On évite de traiter une bougie pas encore clôturée : on s'assure que
        # l'horodatage de la dernière barre + sa durée <= maintenant UTC.
        bar_seconds = int(timeframe_bar_seconds(tf))
        now_utc = datetime.now(tz=UTC)
        if latest_close.timestamp() + bar_seconds > now_utc.timestamp():
            # Bougie courante non clôturée — on tronque la dernière ligne.
            df = df.iloc[:-1].reset_index(drop=True)
            if len(df) < 30:
                return
            latest_close = df["timestamp"].iloc[-1].to_pydatetime()
            if not self._tracker.should_process(symbol, tf, latest_close):
                return

        async with async_session_factory() as session:
            await self._process_with_session(session, symbol, tf, df)
            await session.commit()

        self._tracker.mark(symbol, tf, latest_close)

    async def _process_with_session(
        self,
        session: AsyncSession,
        symbol: str,
        tf: str,
        df: pd.DataFrame,
    ) -> None:
        ts_started = datetime.now(tz=UTC)
        exchange_id = await _ensure_exchange_id(session, settings.exchange_id)
        symbol_id = await ensure_symbol_id(session, exchange_id, symbol)
        timeframe_id = await ensure_timeframe_id(session, tf)

        swings = detect_swings(df, left=3, right=3)

        patterns = []
        for det in self._plan.detectors:
            patterns.extend(det.detect(df, swings, symbol=symbol, timeframe=tf))

        existing = await load_active_hypotheses(session, symbol_id, timeframe_id)
        result = self._engine.step(df, patterns, existing)

        for h in result.created + result.updated:
            await upsert_hypothesis(session, h, symbol_id=symbol_id, timeframe_id=timeframe_id)

        open_trades = await load_open_unit_trades_for_symbol_tf(session, symbol_id, timeframe_id)
        # Restituer symbol/timeframe perdus par la query (cf. note dans repository)
        open_trades = [replace(t, symbol=symbol, timeframe=tf) for t in open_trades]
        all_hypotheses_now = result.created + result.updated
        _, newly_closed, newly_opened = reconcile_with_engine_step(open_trades, all_hypotheses_now)

        for t in newly_opened + newly_closed:
            await insert_unit_trade(session, t, symbol_id=symbol_id, timeframe_id=timeframe_id)

        # Notifications Telegram + Execution exchange (silent fail si non configure)
        try:
            from app.tg_bot.notifier import (
                dispatch_hypothesis_triggered, dispatch_hypothesis_closed,
            )
            from app.execution.router import get_executor
            from app.execution.base import OrderRequest, CloseRequest
            from app.schemas.domain import Side
            import os

            executor = await get_executor()
            exec_mode = os.environ.get("EXECUTION_MODE", "disabled").lower()
            size_usd = float(os.environ.get("MAX_POSITION_USD", "50"))

            for h in result.created + result.updated:
                if not h.transitions:
                    continue
                last_t = h.transitions[-1]

                # TRIGGERED : nouvel ordre + notif
                if (last_t.to_state.value == "TRIGGERED"
                        and last_t.timestamp == h.triggered_at):
                    entry = float(h.triggered_price or h.entry_price)
                    dispatch_hypothesis_triggered(
                        symbol=symbol, timeframe=tf,
                        pattern=h.pattern.kind.value, side=h.side.value,
                        entry=entry, target=float(h.target_price),
                        invalidation=float(h.invalidation_price),
                        confluence_score=float(h.confluence_score),
                    )
                    if exec_mode != "disabled":
                        req = OrderRequest(
                            hypothesis_id=h.id, symbol=symbol, side=h.side.value,
                            entry_price=entry, target_price=float(h.target_price),
                            invalidation_price=float(h.invalidation_price),
                            size_usd=size_usd,
                        )
                        try:
                            res = await executor.open_position(req)
                            if res.ok:
                                logger.info("[%s] OPEN OK %s %s @ %s",
                                            executor.name, symbol, h.side.value, res.filled_price)
                            else:
                                logger.warning("[%s] OPEN REFUSED %s: %s",
                                                executor.name, symbol, res.error)
                        except Exception as exc:
                            logger.exception("Executor open failed for %s", h.id)

                # Close transition (TARGET_HIT / STOPPED / INVALIDATED / EXPIRED)
                elif (last_t.to_state.value in ("TARGET_HIT", "STOPPED", "INVALIDATED", "EXPIRED")
                        and last_t.timestamp == h.closed_at):
                    exit_p = float(h.outcome_price or 0.0)
                    entry_p = float(h.triggered_price or h.entry_price)
                    if entry_p > 0 and exit_p > 0:
                        if h.side == Side.LONG:
                            pct = (exit_p / entry_p - 1.0) * 100
                        else:
                            pct = (entry_p / exit_p - 1.0) * 100
                    else:
                        pct = 0.0
                    dispatch_hypothesis_closed(
                        symbol=symbol, timeframe=tf,
                        pattern=h.pattern.kind.value, side=h.side.value,
                        outcome=last_t.to_state.value,
                        entry=entry_p, exit_price=exit_p, pct_gain=pct,
                    )
                    if exec_mode != "disabled" and h.triggered_at is not None:
                        # Seulement si on a effectivement ouvert (triggered_at set)
                        req = CloseRequest(
                            hypothesis_id=h.id, symbol=symbol, side=h.side.value,
                            reason=last_t.to_state.value,
                        )
                        try:
                            res = await executor.close_position(req)
                            if res.ok:
                                logger.info("[%s] CLOSE OK %s reason=%s",
                                            executor.name, symbol, last_t.to_state.value)
                            # Enregistre PnL pour safety guard
                            try:
                                from app.execution.router import get_safety
                                pnl_usd = pct / 100.0 * size_usd
                                get_safety().record_close(pnl_usd=pnl_usd)
                            except Exception:
                                pass
                        except Exception:
                            logger.exception("Executor close failed for %s", h.id)
        except Exception as e:
            logger.debug("Telegram/execution skipped: %s", e)

        await record_scan_run(
            session,
            symbol_id=symbol_id,
            timeframe_id=timeframe_id,
            ts_started=ts_started,
            ts_finished=datetime.now(tz=UTC),
            candles_fetched=len(df),
            patterns_detected=len(patterns),
            hypotheses_active=sum(1 for h in result.updated if not h.is_terminal)
            + len(result.created),
        )

        if patterns or result.created or result.transitions:
            logger.info(
                "%s %s: %d patterns, %d hypotheses (created=%d, transitions=%d)",
                symbol, tf, len(patterns),
                sum(1 for h in result.updated if not h.is_terminal) + len(result.created),
                len(result.created),
                len(result.transitions),
            )


def _rows_to_df(rows) -> pd.DataFrame:
    """Convertit les CandleRow ccxt en DataFrame.

    Robuste aux differents formats d'horodatage qu'on peut recevoir :
    - datetime tz-aware (cas standard CCXTFetcher)
    - datetime naive (legacy)
    - pd.Timestamp deja construit
    Toujours normalise en UTC.
    """
    out_rows = []
    for r in rows:
        ts_raw = r.ts_open
        try:
            ts = pd.Timestamp(ts_raw)
        except Exception:
            # fallback : conversion via datetime
            ts = pd.Timestamp(str(ts_raw))
        # Normalise en UTC : si naive on localise, si aware on convertit.
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        out_rows.append({
            "timestamp": ts,
            "open": float(r.open),
            "high": float(r.high),
            "low": float(r.low),
            "close": float(r.close),
            "volume": float(r.volume),
        })
    return pd.DataFrame(out_rows)


async def _ensure_exchange_id(session: AsyncSession, code: str) -> int:
    res = await session.execute(select(Exchange.id).where(Exchange.code == code))
    eid = res.scalar_one_or_none()
    if eid is not None:
        return int(eid)
    ex = Exchange(code=code, name=code.capitalize())
    session.add(ex)
    await session.flush()
    return int(ex.id)
