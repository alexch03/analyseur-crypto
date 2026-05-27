"""Full analysis pipeline: fetch → analyse → setups → chart → telegram.

This service orchestrates the entire flow for a single symbol + timeframe.
It can be called from an API endpoint, a background worker, or a CLI command.

The module also exposes helpers for the **O(n) walk-forward backtest**:
- ``precompute_all_structures`` computes swings, BOS/CHOCH, FVG, OB etc.
  once over the whole OHLCV DataFrame.
- ``setups_at_bar`` produces scored setups for a given bar index using
  only the structures confirmed before that bar (no look-ahead).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.chart.renderer import render_chart
from app.config import settings
from app.market_structure.bos_choch import detect_bos_choch
from app.market_structure.fvg import detect_fvg, detect_ifvg
from app.market_structure.order_blocks import detect_order_blocks
from app.market_structure.rsi_divergence import rsi_bearish_divergence_recent, rsi_bullish_divergence_recent
from app.market_structure.support_resistance import detect_sr_levels
from app.market_structure.swings import detect_swings
from app.schemas.domain import MarketContextDTO, Trend, TradeSetupDTO
from app.strategy.scoring import RuleBasedScorer
from app.strategy.setup_engine import RuleBasedSetupEngine
from app.strategy.setup_filters import setup_passes_ifvg_filter, setup_passes_rsi_divergence_filter
from app.telegram.notifier import TelegramNotifier

logger = logging.getLogger(__name__)


def chart_render_kwargs(engine_params: dict[str, Any] | None) -> dict[str, Any]:
    """Options graphique (``render_chart``) depuis ``engine_params`` / état fusionné."""
    ep = engine_params or {}
    kw: dict[str, Any] = {}
    if "chart_focus_last_bars" in ep:
        kw["focus_last_bars"] = ep["chart_focus_last_bars"]
    if "chart_compact_overlays" in ep:
        kw["compact_overlays"] = bool(ep["chart_compact_overlays"])
    return kw


_RULE_ENGINE_KEYS = frozenset(
    {"rr_min", "max_setups", "fvg_proximity_pct", "ob_proximity_pct", "choch_max_age_bars"}
)


@dataclass
class PipelineResult:
    symbol: str
    timeframe: str
    context: MarketContextDTO
    setups: list[TradeSetupDTO]
    chart_png: bytes | None = None
    telegram_sent: bool = False


async def run_analysis(
    ohlcv_df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    *,
    swing_left: int = 3,
    swing_right: int = 3,
    send_telegram: bool = False,
    render_chart_img: bool = True,
    engine_params: dict[str, Any] | None = None,
) -> PipelineResult:
    """Run the full analysis pipeline on an OHLCV DataFrame.

    Parameters
    ----------
    ohlcv_df:
        DataFrame with columns: timestamp, open, high, low, close, volume
    symbol:
        E.g. "BTC/USDT"
    timeframe:
        E.g. "4h"
    swing_left/swing_right:
        Fractal pivot detection window.
    send_telegram:
        If True and Telegram is configured, send signals.
    render_chart_img:
        If True, generate the annotated chart PNG.
    """
    logger.info("Running analysis pipeline for %s %s (%d candles)", symbol, timeframe, len(ohlcv_df))

    ctx, setups = build_context_and_setups(
        ohlcv_df=ohlcv_df,
        symbol=symbol,
        timeframe=timeframe,
        swing_left=swing_left,
        swing_right=swing_right,
        engine_params=engine_params,
    )
    logger.info("  Generated %d trade setups", len(setups))

    # 9. Render chart
    chart_png = None
    if render_chart_img:
        chart_png = render_chart(
            ctx,
            setups,
            title=f"{symbol} | {timeframe} — Analysis",
            **chart_render_kwargs(engine_params),
        )
        logger.info("  Chart rendered (%d bytes)", len(chart_png))

    # 10. Send to Telegram (single grouped message)
    telegram_sent = False
    if send_telegram and settings.telegram_bot_token and settings.telegram_chat_id and setups:
        notifier = TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id)
        receipt = await notifier.send_summary(
            setups,
            chart_png=chart_png,
            trend=ctx.trend.value,
        )
        if receipt.success:
            telegram_sent = True
            logger.info("  Telegram: sent summary (%d setups) to %s", len(setups), settings.telegram_chat_id)
        else:
            logger.warning("  Telegram: failed — %s", receipt.error)

    return PipelineResult(
        symbol=symbol,
        timeframe=timeframe,
        context=ctx,
        setups=setups,
        chart_png=chart_png,
        telegram_sent=telegram_sent,
    )


def build_context_and_setups(
    *,
    ohlcv_df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    swing_left: int = 3,
    swing_right: int = 3,
    engine_params: dict[str, Any] | None = None,
) -> tuple[MarketContextDTO, list[TradeSetupDTO]]:
    """Build market context and scored setups for a given OHLCV slice.

    This synchronous helper is used by both live analysis and replay backtests.
    """
    swings = detect_swings(ohlcv_df, left=swing_left, right=swing_right)
    sr_levels = detect_sr_levels(ohlcv_df, swings, atr_mult=0.5)
    structure_events = detect_bos_choch(swings, ohlcv_df["close"])
    fvgs = detect_fvg(ohlcv_df)
    ifvgs = detect_ifvg(fvgs)
    order_blocks = detect_order_blocks(ohlcv_df)

    trend = Trend.UNDEFINED
    if structure_events:
        trend = structure_events[-1].direction

    indicator_tags = {
        "rsi_bull_div_recent": rsi_bullish_divergence_recent(ohlcv_df),
        "rsi_bear_div_recent": rsi_bearish_divergence_recent(ohlcv_df),
    }

    ctx = MarketContextDTO(
        symbol=symbol,
        timeframe=timeframe,
        ohlcv=ohlcv_df,
        swings=swings,
        sr_levels=sr_levels,
        structure_events=structure_events,
        trend=trend,
        fvgs=fvgs,
        order_blocks=order_blocks,
        ifvgs=ifvgs,
        indicator_tags=indicator_tags,
    )

    ep = engine_params or {}
    rule_kw = {k: ep[k] for k in _RULE_ENGINE_KEYS if k in ep}
    if "rr_min" not in rule_kw:
        rule_kw["rr_min"] = 2.0
    if "max_setups" not in rule_kw:
        rule_kw["max_setups"] = 5
    if "fvg_proximity_pct" not in rule_kw:
        rule_kw["fvg_proximity_pct"] = 0.004
    if "ob_proximity_pct" not in rule_kw:
        rule_kw["ob_proximity_pct"] = 0.004
    if "choch_max_age_bars" not in rule_kw:
        rule_kw["choch_max_age_bars"] = 48

    engine = RuleBasedSetupEngine(**rule_kw)
    raw_setups = engine.propose(ctx)
    scorer = RuleBasedScorer()

    scored: list[TradeSetupDTO] = []
    for s in raw_setups:
        scored_conf = scorer.score(s, ctx)
        scored.append(
            TradeSetupDTO(
                symbol=s.symbol,
                timeframe=s.timeframe,
                side=s.side,
                entry=s.entry,
                stop_loss=s.stop_loss,
                take_profits=s.take_profits,
                risk_reward=s.risk_reward,
                confidence=round(scored_conf, 2),
                setup_type=s.setup_type,
                timestamp=s.timestamp,
                rationale=s.rationale,
                payload=s.payload,
            )
        )

    req_ifvg = bool(ep.get("require_ifvg_confluence"))
    ifvg_pct = float(ep.get("ifvg_confluence_pct", 0.008))
    req_div = bool(ep.get("require_rsi_divergence"))
    if req_ifvg or req_div:
        filtered: list[TradeSetupDTO] = []
        for s in scored:
            if req_ifvg and not setup_passes_ifvg_filter(s, ifvgs, ifvg_pct):
                continue
            if req_div and not setup_passes_rsi_divergence_filter(s, indicator_tags):
                continue
            filtered.append(s)
        scored = filtered

    # Après scoring, la confiance peut différer fortement de l’ordre ``engine.propose`` ;
    # le « top » (paper live, Telegram) doit suivre la confiance finale, pas l’ordre brut.
    scored.sort(key=lambda s: (-float(s.confidence), s.setup_type))

    return ctx, scored


@dataclass
class PrecomputedStructures:
    """All market-structure objects computed once over the full OHLCV series."""
    ohlcv_df: pd.DataFrame
    symbol: str
    timeframe: str
    swings: list
    sr_levels: list
    structure_events: list
    fvgs: list
    ifvgs: list
    order_blocks: list
    indicator_tags: dict[str, bool]


def precompute_all_structures(
    ohlcv_df: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    swing_left: int = 3,
    swing_right: int = 3,
) -> PrecomputedStructures:
    """Compute swings, BOS/CHOCH, FVG, OB, SR, RSI divergence once."""
    swings = detect_swings(ohlcv_df, left=swing_left, right=swing_right)
    sr_levels = detect_sr_levels(ohlcv_df, swings, atr_mult=0.5)
    structure_events = detect_bos_choch(swings, ohlcv_df["close"])
    fvgs = detect_fvg(ohlcv_df)
    ifvgs = detect_ifvg(fvgs)
    order_blocks = detect_order_blocks(ohlcv_df)
    indicator_tags = {
        "rsi_bull_div_recent": rsi_bullish_divergence_recent(ohlcv_df),
        "rsi_bear_div_recent": rsi_bearish_divergence_recent(ohlcv_df),
    }
    return PrecomputedStructures(
        ohlcv_df=ohlcv_df,
        symbol=symbol,
        timeframe=timeframe,
        swings=swings,
        sr_levels=sr_levels,
        structure_events=structure_events,
        fvgs=fvgs,
        ifvgs=ifvgs,
        order_blocks=order_blocks,
        indicator_tags=indicator_tags,
    )


def setups_at_bar(
    pre: PrecomputedStructures,
    bar_idx: int,
    *,
    engine_params: dict[str, Any] | None = None,
    swing_right: int = 3,
) -> list[TradeSetupDTO]:
    """Generate scored setups visible at *bar_idx* (no look-ahead).

    Uses pre-computed structures but filters them to only include events
    that were confirmed before or at *bar_idx*.  Swing confirmation needs
    ``swing_right`` additional bars, so a swing at index *s* is visible
    only when ``bar_idx >= s + swing_right``.
    """
    ep = engine_params or {}
    swing_right_val = int(ep.get("swing_right", swing_right))

    visible_swings = [s for s in pre.swings if s.index + swing_right_val <= bar_idx]
    visible_events = [e for e in pre.structure_events if e.index <= bar_idx]
    visible_fvgs = [f for f in pre.fvgs if f.index + 1 <= bar_idx]
    visible_ifvgs = [f for f in pre.ifvgs if f.index + 1 <= bar_idx]
    visible_obs = [o for o in pre.order_blocks if o.index + 1 <= bar_idx]

    trend = Trend.UNDEFINED
    if visible_events:
        trend = visible_events[-1].direction

    ohlcv_slice = pre.ohlcv_df.iloc[: bar_idx + 1]

    ctx = MarketContextDTO(
        symbol=pre.symbol,
        timeframe=pre.timeframe,
        ohlcv=ohlcv_slice,
        swings=visible_swings,
        sr_levels=pre.sr_levels,
        structure_events=visible_events,
        trend=trend,
        fvgs=visible_fvgs,
        order_blocks=visible_obs,
        ifvgs=visible_ifvgs,
        indicator_tags=pre.indicator_tags,
    )

    rule_kw = {k: ep[k] for k in _RULE_ENGINE_KEYS if k in ep}
    rule_kw.setdefault("rr_min", 2.0)
    rule_kw.setdefault("max_setups", 5)
    rule_kw.setdefault("fvg_proximity_pct", 0.004)
    rule_kw.setdefault("ob_proximity_pct", 0.004)
    rule_kw.setdefault("choch_max_age_bars", 48)

    engine = RuleBasedSetupEngine(**rule_kw)
    raw_setups = engine.propose(ctx)

    scorer = RuleBasedScorer()
    scored: list[TradeSetupDTO] = []
    for s in raw_setups:
        scored_conf = scorer.score(s, ctx)
        scored.append(TradeSetupDTO(
            symbol=s.symbol,
            timeframe=s.timeframe,
            side=s.side,
            entry=s.entry,
            stop_loss=s.stop_loss,
            take_profits=s.take_profits,
            risk_reward=s.risk_reward,
            confidence=round(scored_conf, 2),
            setup_type=s.setup_type,
            timestamp=s.timestamp,
            rationale=s.rationale,
            payload=s.payload,
        ))

    req_ifvg = bool(ep.get("require_ifvg_confluence"))
    ifvg_pct = float(ep.get("ifvg_confluence_pct", 0.008))
    req_div = bool(ep.get("require_rsi_divergence"))
    if req_ifvg or req_div:
        filtered: list[TradeSetupDTO] = []
        for s in scored:
            if req_ifvg and not setup_passes_ifvg_filter(s, visible_ifvgs, ifvg_pct):
                continue
            if req_div and not setup_passes_rsi_divergence_filter(s, pre.indicator_tags):
                continue
            filtered.append(s)
        scored = filtered

    scored.sort(key=lambda s: (-float(s.confidence), s.setup_type))

    return scored


async def run_full_scan(
    *,
    send_telegram: bool = False,
) -> list[PipelineResult]:
    """Run the pipeline for all configured symbols and timeframes."""
    from app.ingestion.ccxt_fetcher import CCXTFetcher

    fetcher = CCXTFetcher(settings.exchange_id)
    results: list[PipelineResult] = []

    try:
        for symbol in settings.symbols:
            for tf in settings.timeframes:
                try:
                    rows = await fetcher.fetch_ohlcv(symbol, tf, limit=500)
                    if not rows:
                        continue

                    df = pd.DataFrame(
                        [
                            {
                                "timestamp": r.ts_open,
                                "open": r.open,
                                "high": r.high,
                                "low": r.low,
                                "close": r.close,
                                "volume": r.volume,
                            }
                            for r in rows
                        ]
                    )

                    result = await run_analysis(
                        df, symbol, tf,
                        send_telegram=send_telegram,
                    )
                    results.append(result)

                except Exception:
                    logger.exception("Failed to analyze %s %s", symbol, tf)

    finally:
        await fetcher.close()

    return results
