"""Backtest and optimization endpoints."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Query

from app.config import settings
from app.ingestion.ccxt_fetcher import CCXTFetcher
from app.paper.engine_replay import ReplayBacktestEngine
from app.services.optimizer import optimize_setup_parameters

router = APIRouter(prefix="/backtest", tags=["backtest"])


async def _fetch_df(symbol: str, tf: str, limit: int) -> pd.DataFrame:
    fetcher = CCXTFetcher(settings.exchange_id)
    try:
        rows = await fetcher.fetch_ohlcv(symbol, tf, limit=limit)
    finally:
        await fetcher.close()

    return pd.DataFrame(
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


@router.post("/run")
async def run_backtest(
    symbol: str = Query("BTC/USDT"),
    tf: str = Query("4h"),
    limit: int = Query(2000, ge=300, le=10000),
    training_bars: int = Query(120, ge=20, le=3000),
    max_holding_bars: int = Query(120, ge=5, le=3000),
    unit_size: float = Query(1.0, gt=0),
    entry_fee_rate: float = Query(0.0004, ge=0),
    exit_fee_rate: float = Query(0.0004, ge=0),
    funding_rate_8h: float = Query(0.0, ge=0),
):
    df = await _fetch_df(symbol, tf, limit)
    engine = ReplayBacktestEngine(
        warmup_bars=training_bars,
        max_holding_bars=max_holding_bars,
        unit_size=unit_size,
        entry_fee_rate=entry_fee_rate,
        exit_fee_rate=exit_fee_rate,
        funding_rate_8h=funding_rate_8h,
    )
    report = engine.run_walkforward(df, symbol=symbol, timeframe=tf)
    return {
        "symbol": symbol,
        "timeframe": tf,
        "bars": len(df),
        "total_trades": report.total_trades,
        "wins": report.wins,
        "losses": report.losses,
        "win_rate": round(report.win_rate, 4),
        "profit_factor": round(report.profit_factor, 4) if report.profit_factor != float("inf") else "inf",
        "expectancy_r": round(report.expectancy_r, 4),
        "net_r": round(report.net_r, 4),
        "max_drawdown_r": round(report.max_drawdown_r, 4),
        "gross_pnl_quote": round(report.gross_pnl_quote, 4),
        "net_pnl_quote": round(report.net_pnl_quote, 4),
        "total_fees_quote": round(report.total_fees_quote, 4),
        "total_funding_quote": round(report.total_funding_quote, 4),
        "avg_trade_duration_bars": round(report.avg_trade_duration_bars, 2),
        "avg_time_in_negative_pct": round(report.avg_time_in_negative_pct, 4),
        "max_drawdown_quote": round(report.max_drawdown_quote, 4),
    }


@router.post("/optimize")
async def optimize_backtest(
    symbol: str = Query("BTC/USDT"),
    tf: str = Query("4h"),
    limit: int = Query(3000, ge=500, le=15000),
    top_n: int = Query(5, ge=1, le=20),
    objective: str = Query("net_pnl_quote"),
    training_bars: int = Query(120, ge=20, le=3000),
    max_holding_bars: int = Query(120, ge=5, le=3000),
    unit_size: float = Query(1.0, gt=0),
    entry_fee_rate: float = Query(0.0004, ge=0),
    exit_fee_rate: float = Query(0.0004, ge=0),
    funding_rate_8h: float = Query(0.0, ge=0),
):
    df = await _fetch_df(symbol, tf, limit)
    results = optimize_setup_parameters(
        df,
        symbol=symbol,
        timeframe=tf,
        objective=objective,
        backtest_config={
            "warmup_bars": training_bars,
            "max_holding_bars": max_holding_bars,
            "unit_size": unit_size,
            "entry_fee_rate": entry_fee_rate,
            "exit_fee_rate": exit_fee_rate,
            "funding_rate_8h": funding_rate_8h,
        },
    )
    best = results[:top_n]
    return {
        "symbol": symbol,
        "timeframe": tf,
        "bars": len(df),
        "evaluated": len(results),
        "objective": objective,
        "best": [
            {
                "rank": i + 1,
                "params": r.params,
                "total_trades": r.report.total_trades,
                "win_rate": round(r.report.win_rate, 4),
                "profit_factor": round(r.report.profit_factor, 4) if r.report.profit_factor != float("inf") else "inf",
                "expectancy_r": round(r.report.expectancy_r, 4),
                "net_r": round(r.report.net_r, 4),
                "max_drawdown_r": round(r.report.max_drawdown_r, 4),
                "net_pnl_quote": round(r.report.net_pnl_quote, 4),
                "max_drawdown_quote": round(r.report.max_drawdown_quote, 4),
            }
            for i, r in enumerate(best)
        ],
    }
