"""CLI worker: run the full scan pipeline from the command line.

Usage:
    python -m app.worker                         # analyze all pairs, no telegram
    python -m app.worker --telegram              # analyze + send to telegram
    python -m app.worker --symbol BTC/USDT --tf 4h  # single pair
    python -m app.worker --save-charts ./output  # save chart PNGs to directory
    python -m app.worker --backtest --symbol BTC/USDT --tf 4h
    python -m app.worker --optimize --symbol BTC/USDT --tf 4h
    python -m app.worker --scan-daemon           # boucle continue patterns × 50 cryptos
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyseur Crypto — CLI Scanner")
    parser.add_argument("--symbol", type=str, help="Single symbol to analyze (e.g. BTC/USDT)")
    parser.add_argument("--tf", type=str, help="Single timeframe (e.g. 4h)")
    parser.add_argument("--telegram", action="store_true", help="Send results to Telegram")
    parser.add_argument("--backtest", action="store_true", help="Run replay backtest")
    parser.add_argument("--optimize", action="store_true", help="Run parameter optimization")
    parser.add_argument("--save-charts", type=str, help="Directory to save chart PNGs")
    parser.add_argument("--limit", type=int, default=500, help="Number of candles to fetch")
    parser.add_argument(
        "--scan-daemon",
        action="store_true",
        help="Lance le scanner continu de patterns chartistes (50 cryptos × 15m/1h/4h)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    from app.logging_setup import setup_logging
    setup_logging(level_console=logging.DEBUG if args.verbose else logging.INFO)

    asyncio.run(async_main(args))


async def async_main(args) -> None:
    from app.config import settings
    from app.ingestion.ccxt_fetcher import CCXTFetcher
    from app.paper.engine_replay import ReplayBacktestEngine
    from app.services.analysis_pipeline import run_analysis
    from app.services.optimizer import optimize_setup_parameters

    if args.scan_daemon:
        from app.services.continuous_scanner import ContinuousScanner
        scanner = ContinuousScanner()
        logging.info(
            "Scanner daemon: %d symbols × %d timeframes (Ctrl+C pour stopper)",
            len(settings.effective_scan_symbols()),
            len(settings.effective_scan_timeframes()),
        )
        try:
            await scanner.run()
        finally:
            await scanner.stop()
        return

    symbols = [args.symbol] if args.symbol else settings.symbols
    timeframes = [args.tf] if args.tf else settings.timeframes

    if args.save_charts:
        Path(args.save_charts).mkdir(parents=True, exist_ok=True)

    fetcher = CCXTFetcher(settings.exchange_id)

    try:
        for symbol in symbols:
            for tf in timeframes:
                logging.info("Fetching %s %s ...", symbol, tf)
                rows = await fetcher.fetch_ohlcv(symbol, tf, limit=args.limit)
                if not rows:
                    logging.warning("No data for %s %s", symbol, tf)
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
                    send_telegram=args.telegram,
                    render_chart_img=True,
                )

                print(f"\n{'='*60}")
                print(f"  {symbol} | {tf} - Trend: {result.context.trend.value}")
                print(f"  Swings: {len(result.context.swings)} | "
                      f"S/R: {len(result.context.sr_levels)} | "
                      f"BOS/CHOCH: {len(result.context.structure_events)} | "
                      f"FVGs: {len(result.context.fvgs)} | "
                      f"OBs: {len(result.context.order_blocks)}")
                print(f"  Setups found: {len(result.setups)}")

                for s in result.setups:
                    print(f"    -> {s.side.value} {s.setup_type}: "
                          f"entry={s.entry:.2f} SL={s.stop_loss:.2f} "
                          f"TP={[f'{t:.2f}' for t in s.take_profits]} "
                          f"R:R={s.risk_reward:.1f} conf={s.confidence:.0%}")
                    print(f"      {s.rationale}")

                if result.chart_png and args.save_charts:
                    safe_name = symbol.replace("/", "_")
                    path = Path(args.save_charts) / f"{safe_name}_{tf}.png"
                    path.write_bytes(result.chart_png)
                    print(f"  Chart saved: {path}")

                if result.telegram_sent:
                    print("  [OK] Sent to Telegram")

                if args.backtest:
                    bt = ReplayBacktestEngine().run_walkforward(df, symbol=symbol, timeframe=tf)
                    print("  Backtest:")
                    print(f"    trades={bt.total_trades} winrate={bt.win_rate:.1%} PF={bt.profit_factor:.2f} "
                          f"netR={bt.net_r:.2f} maxDD={bt.max_drawdown_r:.2f}")

                if args.optimize:
                    opt = optimize_setup_parameters(df, symbol=symbol, timeframe=tf)[:3]
                    print("  Optimization Top 3:")
                    for rank, item in enumerate(opt, start=1):
                        r = item.report
                        print(f"    #{rank} params={item.params} | trades={r.total_trades} "
                              f"winrate={r.win_rate:.1%} PF={r.profit_factor:.2f} netR={r.net_r:.2f}")

                print(f"{'='*60}")

    finally:
        await fetcher.close()


if __name__ == "__main__":
    main()
