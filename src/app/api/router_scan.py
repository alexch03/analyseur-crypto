"""Scan endpoint: triggers full analysis pipeline for configured pairs."""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import Response

from app.services.analysis_pipeline import run_analysis, run_full_scan

router = APIRouter(prefix="/scan", tags=["scan"])


@router.post("/run")
async def scan_all(
    send_telegram: bool = Query(False),
):
    """Run full scan on all configured symbols/timeframes."""
    results = await run_full_scan(send_telegram=send_telegram)
    return {
        "scanned": len(results),
        "setups": [
            {
                "symbol": r.symbol,
                "timeframe": r.timeframe,
                "trend": r.context.trend.value,
                "n_swings": len(r.context.swings),
                "n_fvgs": len(r.context.fvgs),
                "n_obs": len(r.context.order_blocks),
                "n_structure_events": len(r.context.structure_events),
                "setups": [
                    {
                        "type": s.setup_type,
                        "side": s.side.value,
                        "entry": s.entry,
                        "sl": s.stop_loss,
                        "tps": s.take_profits,
                        "rr": s.risk_reward,
                        "confidence": s.confidence,
                        "rationale": s.rationale,
                    }
                    for s in r.setups
                ],
                "chart_available": r.chart_png is not None,
                "telegram_sent": r.telegram_sent,
            }
            for r in results
        ],
    }


@router.post("/chart")
async def scan_chart(
    symbol: str = Query("BTC/USDT"),
    tf: str = Query("4h"),
    send_telegram: bool = Query(False),
):
    """Run analysis on a single pair and return the chart image."""
    from app.ingestion.ccxt_fetcher import CCXTFetcher
    from app.config import settings
    import pandas as pd

    fetcher = CCXTFetcher(settings.exchange_id)
    try:
        rows = await fetcher.fetch_ohlcv(symbol, tf, limit=500)
    finally:
        await fetcher.close()

    if not rows:
        return {"error": "No data returned"}

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

    result = await run_analysis(df, symbol, tf, send_telegram=send_telegram)

    if result.chart_png:
        return Response(content=result.chart_png, media_type="image/png")
    return {"error": "Chart generation failed"}
