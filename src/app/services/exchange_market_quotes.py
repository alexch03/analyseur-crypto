"""Fetch taker fee and funding rate from the configured CCXT exchange (e.g. Bitget)."""

from __future__ import annotations

from typing import Any

import ccxt.async_support as ccxt_async


def _to_linear_swap_symbol(spot_like: str) -> str | None:
    """Map BTC/USDT -> BTC/USDT:USDT for USDT linear perps in unified CCXT."""
    s = spot_like.strip().upper()
    if ":" in s:
        return spot_like.strip()
    if s.endswith("/USDT"):
        return f"{spot_like.strip()}:USDT"
    return None


def _market_fee_side(m: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": m["symbol"],
        "taker": float(m["taker"]) if m.get("taker") is not None else None,
        "maker": float(m["maker"]) if m.get("maker") is not None else None,
    }


async def fetch_taker_fee_rate(
    exchange_id: str,
    symbol: str,
    *,
    market_kind: str,
) -> dict[str, Any]:
    """Return taker fee (fraction CCXT : 0.0006 = 0,06 % du notionnel) pour spot et/ou swap."""
    bundle = await fetch_market_info_bundle(exchange_id, symbol)
    out = {**bundle, "taker_applied": None}
    if market_kind == "swap" and bundle.get("swap") and bundle["swap"].get("taker") is not None:
        out["taker_applied"] = bundle["swap"]["taker"]
    elif market_kind == "spot" and bundle.get("spot") and bundle["spot"].get("taker") is not None:
        out["taker_applied"] = bundle["spot"]["taker"]
    out["note_fr"] = bundle.get("note_fr")
    return out


async def fetch_funding_rate_8h(exchange_id: str, symbol: str) -> dict[str, Any]:
    """Funding courant sur le perp linéaire (ex. BTC/USDT -> BTC/USDT:USDT)."""
    bundle = await fetch_market_info_bundle(exchange_id, symbol)
    return {
        "exchange_id": exchange_id,
        "symbol_input": symbol,
        "swap_symbol": bundle.get("swap_symbol"),
        "funding_rate_8h": bundle.get("funding_rate_8h"),
        "funding_timestamp_ms": bundle.get("funding_timestamp_ms"),
        "funding_datetime": bundle.get("funding_datetime"),
        "interval": bundle.get("funding_interval"),
        "error": bundle.get("funding_error"),
        "note_fr": bundle.get("note_fr"),
    }


async def fetch_market_info_bundle(exchange_id: str, symbol: str) -> dict[str, Any]:
    """Une connexion CCXT : frais spot/swap + funding swap (pour le dashboard)."""
    cls = getattr(ccxt_async, exchange_id)
    ex = cls({"enableRateLimit": True})
    out: dict[str, Any] = {
        "exchange_id": exchange_id,
        "symbol_input": symbol.strip(),
        "spot": None,
        "swap": None,
        "swap_symbol": None,
        "funding_rate_8h": None,
        "funding_timestamp_ms": None,
        "funding_datetime": None,
        "funding_interval": None,
        "funding_error": None,
        "note_fr": (
            "Frais : fractions CCXT — ex. taker 0.0006 = 0,06 % du notionnel (prix × quantité en USDT). "
            "Sur Bitget, le spot USDT a souvent un taker plus élevé que le perp. "
            "Funding : taux par période (souvent 8 h) sur le perp ; signe selon convention exchange."
        ),
    }
    try:
        await ex.load_markets()
        spot_sym = symbol.strip()
        if spot_sym in ex.markets:
            out["spot"] = _market_fee_side(ex.market(spot_sym))
        swap_sym = _to_linear_swap_symbol(symbol)
        out["swap_symbol"] = swap_sym
        if swap_sym and swap_sym in ex.markets:
            out["swap"] = _market_fee_side(ex.market(swap_sym))
            try:
                raw = await ex.fetch_funding_rate(swap_sym)
                out["funding_rate_8h"] = (
                    float(raw["fundingRate"]) if raw.get("fundingRate") is not None else None
                )
                out["funding_timestamp_ms"] = raw.get("fundingTimestamp")
                out["funding_datetime"] = raw.get("fundingDatetime")
                out["funding_interval"] = raw.get("interval") or "8h"
            except Exception as e:  # noqa: BLE001
                out["funding_error"] = str(e)
        elif swap_sym:
            out["funding_error"] = f"Marché swap inconnu: {swap_sym}"
        else:
            out["funding_error"] = "Paire non mappée vers un perp USDT (ex. BTC/USDT)."
        return out
    finally:
        await ex.close()
