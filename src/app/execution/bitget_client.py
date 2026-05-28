"""Client Bitget USDT-FUTURES via ccxt.

Modes :
    - demo=True  : sandbox testnet (env BITGET_DEMO_*)
    - demo=False : mainnet LIVE (env BITGET_LIVE_*)

Methodes async pour ne pas bloquer le scanner.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import ccxt.async_support as ccxt_async

logger = logging.getLogger(__name__)


def _normalize_futures_symbol(symbol: str) -> str:
    """BTC/USDT -> BTC/USDT:USDT (notation ccxt pour futures perpetuels)."""
    if ":" in symbol:
        return symbol
    if "/" not in symbol:
        return f"{symbol}/USDT:USDT"
    return f"{symbol}:USDT"


class BitgetClient:
    """Wrapper ccxt.bitget pour USDT-FUTURES."""

    def __init__(self, *, demo: bool = True) -> None:
        self._demo = demo
        prefix = "BITGET_DEMO_" if demo else "BITGET_LIVE_"
        api_key = os.environ.get(f"{prefix}API_KEY", "")
        secret = os.environ.get(f"{prefix}SECRET", "")
        passphrase = os.environ.get(f"{prefix}PASSPHRASE", "")
        if not (api_key and secret and passphrase):
            raise RuntimeError(
                f"Bitget credentials manquants : {prefix}API_KEY / SECRET / PASSPHRASE dans .env"
            )
        self._ex = ccxt_async.bitget({
            "apiKey": api_key,
            "secret": secret,
            "password": passphrase,
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap",          # perpetual futures
                "defaultSubType": "linear",     # USDT-margined
            },
        })
        if demo:
            # Bitget demo via sandbox flag ccxt
            try:
                self._ex.set_sandbox_mode(True)
            except Exception:
                # Pour Bitget specifiquement, demo trading se passe par productType
                self._ex.options["sandboxMode"] = True

    @property
    def demo(self) -> bool:
        return self._demo

    @property
    def label(self) -> str:
        return "bitget-demo" if self._demo else "bitget-live"

    async def close(self) -> None:
        try:
            await self._ex.close()
        except Exception:
            pass

    async def fetch_balance(self) -> dict[str, Any]:
        try:
            bal = await self._ex.fetch_balance(params={"productType": "USDT-FUTURES"})
            usdt = bal.get("USDT", {})
            return {
                "free": float(usdt.get("free") or 0.0),
                "used": float(usdt.get("used") or 0.0),
                "total": float(usdt.get("total") or 0.0),
            }
        except Exception as e:
            logger.warning("Bitget fetch_balance failed: %s", e)
            return {"free": 0.0, "used": 0.0, "total": 0.0}

    async def fetch_open_positions(self) -> list[dict]:
        try:
            positions = await self._ex.fetch_positions(
                params={"productType": "USDT-FUTURES"}
            )
            return [p for p in positions if abs(float(p.get("contracts") or 0)) > 0]
        except Exception as e:
            logger.warning("Bitget fetch_positions failed: %s", e)
            return []

    async def place_market_order(
        self, *, symbol: str, side: str, qty: float, leverage: int = 1,
        sl: float | None = None, tp: float | None = None,
    ) -> dict:
        """Place un ordre market avec SL/TP optionnels.

        side : "LONG" ou "SHORT" (ccxt utilise "buy"/"sell").
        """
        sym = _normalize_futures_symbol(symbol)
        ccxt_side = "buy" if side.upper() == "LONG" else "sell"
        hold_side = "long" if side.upper() == "LONG" else "short"

        # Set leverage avant placement
        try:
            await self._ex.set_leverage(int(leverage), sym, params={
                "marginCoin": "USDT", "holdSide": hold_side,
            })
        except Exception as e:
            logger.debug("set_leverage: %s", e)

        params = {
            "productType": "USDT-FUTURES",
            "marginCoin": "USDT",
            "holdSide": hold_side,
            "tradeSide": "open",
            "marginMode": "isolated",
        }
        if sl is not None:
            params["stopLoss"] = {"triggerPrice": float(sl)}
        if tp is not None:
            params["takeProfit"] = {"triggerPrice": float(tp)}

        order = await self._ex.create_order(sym, "market", ccxt_side, qty, params=params)
        return order

    async def close_market_position(self, *, symbol: str, side: str) -> dict:
        """Ferme la position market (reduceOnly)."""
        sym = _normalize_futures_symbol(symbol)
        positions = await self._ex.fetch_positions([sym], params={"productType": "USDT-FUTURES"})
        for p in positions:
            if p["symbol"] != sym:
                continue
            qty = abs(float(p.get("contracts") or 0))
            if qty == 0:
                return {"info": "no position"}
            hold_side = p.get("side", "long")
            ccxt_side = "sell" if hold_side == "long" else "buy"
            return await self._ex.create_order(
                sym, "market", ccxt_side, qty,
                params={
                    "productType": "USDT-FUTURES",
                    "marginCoin": "USDT",
                    "holdSide": hold_side,
                    "tradeSide": "close",
                    "reduceOnly": True,
                },
            )
        return {"info": f"no position on {sym}"}

    async def close_all_positions(self) -> list[dict]:
        """EMERGENCY : ferme TOUTES les positions ouvertes."""
        positions = await self.fetch_open_positions()
        results = []
        for p in positions:
            try:
                sym = p["symbol"]
                hold_side = p.get("side", "long")
                qty = abs(float(p.get("contracts") or 0))
                if qty == 0:
                    continue
                ccxt_side = "sell" if hold_side == "long" else "buy"
                r = await self._ex.create_order(
                    sym, "market", ccxt_side, qty,
                    params={
                        "productType": "USDT-FUTURES",
                        "marginCoin": "USDT",
                        "holdSide": hold_side,
                        "tradeSide": "close",
                        "reduceOnly": True,
                    },
                )
                results.append({"symbol": sym, "ok": True, "order": r})
            except Exception as e:
                results.append({"symbol": p.get("symbol"), "ok": False, "error": str(e)})
        return results
