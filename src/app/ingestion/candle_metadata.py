"""Ensure exchange / symbol / timeframe rows exist for candle storage."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.candle import Exchange, Symbol, Timeframe


async def ensure_symbol_timeframe_ids(
    session: AsyncSession, exchange_code: str, symbol_str: str, tf_code: str
) -> tuple[int, int]:
    """Return (symbol_id, timeframe_id), creating rows if needed."""
    ex = (await session.execute(select(Exchange).where(Exchange.code == exchange_code))).scalar_one_or_none()
    if ex is None:
        ex = Exchange(code=exchange_code, name=exchange_code)
        session.add(ex)
        await session.flush()

    base, quote = symbol_str.upper().split("/")
    sym = (
        await session.execute(
            select(Symbol).where(Symbol.exchange_id == ex.id, Symbol.base == base, Symbol.quote == quote)
        )
    ).scalar_one_or_none()
    if sym is None:
        sym = Symbol(exchange_id=ex.id, base=base, quote=quote)
        session.add(sym)
        await session.flush()

    tf = (await session.execute(select(Timeframe).where(Timeframe.code == tf_code))).scalar_one_or_none()
    if tf is None:
        tf = Timeframe(code=tf_code)
        session.add(tf)
        await session.flush()

    return sym.id, tf.id
