"""PaperExecutor : simulation interne (DB only).

Sert de comportement par defaut. Ne place aucun ordre reel.
Sert aussi de fallback si Bitget est inaccessible.
"""

from __future__ import annotations

import logging
from datetime import datetime

from app.execution.base import (
    CloseRequest, Executor, OrderRequest, OrderResult,
)

logger = logging.getLogger(__name__)


class PaperExecutor(Executor):
    name = "paper"

    def __init__(self) -> None:
        self._open: dict[str, dict] = {}   # hypothesis_id -> {symbol, side, entry, qty}
        self._fake_balance = 10000.0       # 10k$ initial paper balance

    async def open_position(self, req: OrderRequest) -> OrderResult:
        if req.hypothesis_id in self._open:
            return OrderResult(ok=False, error="position already open for this hypothesis")
        self._open[req.hypothesis_id] = {
            "symbol": req.symbol,
            "side": req.side,
            "entry": req.entry_price,
            "size_usd": req.size_usd,
            "opened_at": datetime.utcnow(),
        }
        logger.info("[PAPER] OPEN %s %s @ %s size=%s$",
                    req.symbol, req.side, req.entry_price, req.size_usd)
        return OrderResult(
            ok=True,
            order_id=f"paper-{req.hypothesis_id}",
            filled_price=req.entry_price,
            filled_qty=req.size_usd / req.entry_price if req.entry_price > 0 else 0.0,
            exchange="paper",
        )

    async def close_position(self, req: CloseRequest) -> OrderResult:
        pos = self._open.pop(req.hypothesis_id, None)
        if pos is None:
            return OrderResult(ok=False, error="no open position")
        logger.info("[PAPER] CLOSE %s %s reason=%s", req.symbol, req.side, req.reason)
        return OrderResult(
            ok=True,
            order_id=f"paper-close-{req.hypothesis_id}",
            exchange="paper",
        )

    async def fetch_open_positions(self) -> list[dict]:
        return [
            {"hypothesis_id": hid, **info}
            for hid, info in self._open.items()
        ]

    async def fetch_balance(self) -> dict:
        return {"free": self._fake_balance, "used": 0.0, "total": self._fake_balance}
