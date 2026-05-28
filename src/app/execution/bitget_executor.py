"""BitgetExecutor : execute des trades reels sur Bitget Demo ou Live.

Avant chaque ouverture : SafetyGuard.can_open() doit retourner True.
"""

from __future__ import annotations

import logging

from app.execution.base import (
    CloseRequest, Executor, OrderRequest, OrderResult,
)
from app.execution.bitget_client import BitgetClient
from app.execution.safety import SafetyGuard

logger = logging.getLogger(__name__)


class BitgetExecutor(Executor):
    """Execute via Bitget (demo ou live selon le client)."""

    def __init__(self, *, client: BitgetClient, safety: SafetyGuard,
                  default_leverage: int = 1) -> None:
        self._client = client
        self._safety = safety
        self._leverage = int(default_leverage)
        self.name = client.label

    async def open_position(self, req: OrderRequest) -> OrderResult:
        # 1. Verifie safety
        balance = (await self._client.fetch_balance()).get("free", 0.0)
        open_positions = await self._client.fetch_open_positions()
        ok, reason = self._safety.can_open(
            symbol=req.symbol, side=req.side, size_usd=req.size_usd,
            balance_usd=balance, open_positions_count=len(open_positions),
        )
        if not ok:
            logger.warning("[%s] REFUSE OPEN %s %s : %s",
                            self.name, req.symbol, req.side, reason)
            return OrderResult(ok=False, error=reason, exchange=self.name)

        if req.entry_price <= 0:
            return OrderResult(ok=False, error="invalid entry_price", exchange=self.name)

        # 2. Calcule qty depuis size_usd
        qty = req.size_usd / req.entry_price

        # 3. Place l'ordre
        try:
            order = await self._client.place_market_order(
                symbol=req.symbol, side=req.side, qty=qty,
                leverage=self._leverage,
                sl=req.invalidation_price, tp=req.target_price,
            )
            logger.info("[%s] OPEN %s %s qty=%s @ ~%s$",
                        self.name, req.symbol, req.side, qty, req.entry_price)
            return OrderResult(
                ok=True,
                order_id=str(order.get("id") or "?"),
                filled_price=float(order.get("price") or req.entry_price),
                filled_qty=float(order.get("amount") or qty),
                exchange=self.name,
            )
        except Exception as e:
            logger.exception("[%s] OPEN FAILED %s %s", self.name, req.symbol, req.side)
            return OrderResult(ok=False, error=str(e)[:200], exchange=self.name)

    async def close_position(self, req: CloseRequest) -> OrderResult:
        try:
            order = await self._client.close_market_position(
                symbol=req.symbol, side=req.side,
            )
            logger.info("[%s] CLOSE %s %s reason=%s",
                        self.name, req.symbol, req.side, req.reason)
            return OrderResult(
                ok=True,
                order_id=str(order.get("id") or "?"),
                exchange=self.name,
            )
        except Exception as e:
            logger.exception("[%s] CLOSE FAILED %s", self.name, req.symbol)
            return OrderResult(ok=False, error=str(e)[:200], exchange=self.name)

    async def fetch_open_positions(self) -> list[dict]:
        return await self._client.fetch_open_positions()

    async def fetch_balance(self) -> dict:
        return await self._client.fetch_balance()

    async def close(self) -> None:
        await self._client.close()
