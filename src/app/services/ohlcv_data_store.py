"""Local CSV OHLCV cache + DB reads for backtest / optimisation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models.candle import Candle, Exchange, Symbol, Timeframe


def ohlcv_dir() -> Path:
    p = Path(settings.ohlcv_data_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ohlcv_csv_path(symbol: str, timeframe: str) -> Path:
    stem = symbol.replace("/", "_").replace(" ", "")
    return ohlcv_dir() / f"{stem}__{timeframe}.csv"


def save_ohlcv_csv(df: pd.DataFrame, symbol: str, timeframe: str) -> str:
    """Write OHLCV to CSV under configured data dir. Returns absolute path string."""
    path = ohlcv_csv_path(symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if "timestamp" not in out.columns:
        raise ValueError("DataFrame must have a timestamp column")
    out = out.sort_values("timestamp")
    out.to_csv(path, index=False)
    return str(path.resolve())


def load_ohlcv_csv(symbol: str, timeframe: str, limit: int | None = None) -> pd.DataFrame:
    path = ohlcv_csv_path(symbol, timeframe)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if limit is not None and limit > 0 and len(df) > limit:
        df = df.sort_values("timestamp").tail(int(limit)).reset_index(drop=True)
    return df


def file_dataset_status(symbol: str, timeframe: str) -> dict[str, Any]:
    path = ohlcv_csv_path(symbol, timeframe)
    if not path.exists():
        return {
            "exists": False,
            "path": str(path.resolve()),
            "bars": 0,
            "file_mtime_utc": None,
            "last_candle_open_utc": None,
            "age_seconds_since_last_candle": None,
        }
    stat = path.stat()
    mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df.empty or "timestamp" not in df.columns:
        return {
            "exists": True,
            "path": str(path.resolve()),
            "bars": 0,
            "file_mtime_utc": mtime.isoformat(),
            "last_candle_open_utc": None,
            "age_seconds_since_last_candle": None,
        }
    last_ts = pd.Timestamp(df["timestamp"].max()).to_pydatetime()
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=UTC)
    now = datetime.now(tz=UTC)
    age = max(0.0, (now - last_ts.astimezone(UTC)).total_seconds())
    return {
        "exists": True,
        "path": str(path.resolve()),
        "bars": int(len(df)),
        "file_mtime_utc": mtime.isoformat(),
        "last_candle_open_utc": last_ts.astimezone(UTC).isoformat(),
        "age_seconds_since_last_candle": round(age, 1),
    }


async def _symbol_on_exchange(
    session: AsyncSession, exchange_code: str, base: str, quote: str
) -> Symbol | None:
    ex = (await session.execute(select(Exchange).where(Exchange.code == exchange_code))).scalar_one_or_none()
    if ex is None:
        return None
    return (
        await session.execute(
            select(Symbol).where(
                Symbol.exchange_id == ex.id,
                Symbol.base == base,
                Symbol.quote == quote,
            )
        )
    ).scalar_one_or_none()


async def load_ohlcv_from_db(session: AsyncSession, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    base, quote = symbol.upper().split("/")
    sym = await _symbol_on_exchange(session, settings.exchange_id, base, quote)
    if sym is None:
        return pd.DataFrame()

    tf_obj = (await session.execute(select(Timeframe).where(Timeframe.code == timeframe))).scalar_one_or_none()
    if tf_obj is None:
        return pd.DataFrame()

    q = (
        select(Candle)
        .where(Candle.symbol_id == sym.id, Candle.timeframe_id == tf_obj.id)
        .order_by(Candle.ts_open.desc())
        .limit(int(limit))
    )
    rows = (await session.execute(q)).scalars().all()
    if not rows:
        return pd.DataFrame()
    rows = list(reversed(rows))
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


async def db_dataset_status(session: AsyncSession, symbol: str, timeframe: str) -> dict[str, Any]:
    base, quote = symbol.upper().split("/")
    sym = await _symbol_on_exchange(session, settings.exchange_id, base, quote)
    if sym is None:
        return {
            "has_symbol": False,
            "bars": 0,
            "last_candle_open_utc": None,
            "age_seconds_since_last_candle": None,
        }

    tf_obj = (await session.execute(select(Timeframe).where(Timeframe.code == timeframe))).scalar_one_or_none()
    if tf_obj is None:
        return {
            "has_symbol": True,
            "has_timeframe": False,
            "bars": 0,
            "last_candle_open_utc": None,
            "age_seconds_since_last_candle": None,
        }

    cnt = (
        await session.execute(
            select(func.count()).select_from(Candle).where(
                Candle.symbol_id == sym.id, Candle.timeframe_id == tf_obj.id
            )
        )
    ).scalar_one()

    last_q = (
        select(Candle.ts_open)
        .where(Candle.symbol_id == sym.id, Candle.timeframe_id == tf_obj.id)
        .order_by(Candle.ts_open.desc())
        .limit(1)
    )
    last_row = (await session.execute(last_q)).scalar_one_or_none()
    if last_row is None:
        return {
            "has_symbol": True,
            "has_timeframe": True,
            "bars": 0,
            "last_candle_open_utc": None,
            "age_seconds_since_last_candle": None,
        }

    last_ts = last_row
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=UTC)
    now = datetime.now(tz=UTC)
    age = max(0.0, (now - last_ts.astimezone(UTC)).total_seconds())
    return {
        "has_symbol": True,
        "has_timeframe": True,
        "bars": int(cnt or 0),
        "last_candle_open_utc": last_ts.astimezone(UTC).isoformat(),
        "age_seconds_since_last_candle": round(age, 1),
    }
