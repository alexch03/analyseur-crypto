"""Univers de symboles à scanner par le moteur d'hypothèses.

Liste statique curated des ~50 paires USDT les plus liquides sur Binance, ordre
approximatif par market cap au moment de l'édition. La liste est volontairement
hardcodée pour rester déterministe (pas de dépendance réseau à l'import).

Override possible via la variable ``symbols`` dans ``Settings`` (config.py) ou via
``ANALYSEUR_SYMBOLS`` (CSV) dans le ``.env``.

Pour rafraîchir dynamiquement depuis l'exchange, utiliser :func:`fetch_top_usdt_pairs`
qui appelle ``ccxt.fetch_markets`` puis filtre par volume 24h.
"""

from __future__ import annotations

DEFAULT_UNIVERSE_50: list[str] = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "SOL/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "TRX/USDT",
    "AVAX/USDT",
    "DOT/USDT",
    "LINK/USDT",
    "MATIC/USDT",
    "TON/USDT",
    "SHIB/USDT",
    "LTC/USDT",
    "BCH/USDT",
    "ATOM/USDT",
    "UNI/USDT",
    "ETC/USDT",
    "XLM/USDT",
    "NEAR/USDT",
    "OP/USDT",
    "ARB/USDT",
    "FIL/USDT",
    "APT/USDT",
    "ICP/USDT",
    "IMX/USDT",
    "INJ/USDT",
    "RNDR/USDT",
    "HBAR/USDT",
    "AAVE/USDT",
    "VET/USDT",
    "GRT/USDT",
    "LDO/USDT",
    "FTM/USDT",
    "ALGO/USDT",
    "MKR/USDT",
    "SAND/USDT",
    "MANA/USDT",
    "AXS/USDT",
    "EOS/USDT",
    "EGLD/USDT",
    "FLOW/USDT",
    "XTZ/USDT",
    "THETA/USDT",
    "KAVA/USDT",
    "RUNE/USDT",
    "SUI/USDT",
    "SEI/USDT",
    "PEPE/USDT",
]

DEFAULT_SCAN_TIMEFRAMES: list[str] = ["15m", "1h", "4h"]


async def fetch_top_usdt_pairs(
    exchange_id: str = "binance",
    *,
    quote: str = "USDT",
    limit: int = 50,
) -> list[str]:
    """Liste les ``limit`` paires ``base/quote`` les plus volumineuses sur 24h.

    Import paresseux de ccxt pour éviter une dépendance dure si non installé.
    """
    import ccxt.async_support as ccxt_async   # type: ignore[import-not-found]

    ex_cls = getattr(ccxt_async, exchange_id)
    ex = ex_cls({"enableRateLimit": True})
    try:
        await ex.load_markets()
        tickers = await ex.fetch_tickers()
    finally:
        await ex.close()

    candidates: list[tuple[str, float]] = []
    for sym, t in tickers.items():
        if "/" not in sym:
            continue
        base, q = sym.split("/", 1)
        if q != quote or not base:
            continue
        info = t or {}
        qv = info.get("quoteVolume") or 0.0
        try:
            qv_f = float(qv)
        except (TypeError, ValueError):
            qv_f = 0.0
        candidates.append((sym, qv_f))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in candidates[:limit]]
