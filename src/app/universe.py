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
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "SOL/USDT",
    "ADA/USDT", "DOGE/USDT", "TRX/USDT", "AVAX/USDT", "DOT/USDT",
    "LINK/USDT", "MATIC/USDT", "TON/USDT", "SHIB/USDT", "LTC/USDT",
    "BCH/USDT", "ATOM/USDT", "UNI/USDT", "ETC/USDT", "XLM/USDT",
    "NEAR/USDT", "OP/USDT", "ARB/USDT", "FIL/USDT", "APT/USDT",
    "ICP/USDT", "IMX/USDT", "INJ/USDT", "RNDR/USDT", "HBAR/USDT",
    "AAVE/USDT", "VET/USDT", "GRT/USDT", "LDO/USDT", "FTM/USDT",
    "ALGO/USDT", "MKR/USDT", "SAND/USDT", "MANA/USDT", "AXS/USDT",
    "EOS/USDT", "EGLD/USDT", "FLOW/USDT", "XTZ/USDT", "THETA/USDT",
    "KAVA/USDT", "RUNE/USDT", "SUI/USDT", "SEI/USDT", "PEPE/USDT",
]

# Univers etendu : top 100 USDT majors + altcoins liquides
DEFAULT_UNIVERSE_100: list[str] = DEFAULT_UNIVERSE_50 + [
    "WLD/USDT", "JUP/USDT", "TIA/USDT", "WIF/USDT", "BONK/USDT",
    "FET/USDT", "GALA/USDT", "FLOKI/USDT", "JTO/USDT", "PYTH/USDT",
    "DYDX/USDT", "STX/USDT", "ENS/USDT", "MINA/USDT", "1INCH/USDT",
    "CRV/USDT", "COMP/USDT", "SNX/USDT", "SUSHI/USDT", "ZIL/USDT",
    "ZRX/USDT", "ANKR/USDT", "BAT/USDT", "CHZ/USDT", "ENJ/USDT",
    "GMX/USDT", "BLUR/USDT", "SSV/USDT", "MAGIC/USDT", "RDNT/USDT",
    "ORDI/USDT", "JASMY/USDT", "CFX/USDT", "MASK/USDT", "WOO/USDT",
    "ROSE/USDT", "ARKM/USDT", "AGIX/USDT", "RLC/USDT", "STORJ/USDT",
    "OCEAN/USDT", "BAND/USDT", "API3/USDT", "ICX/USDT", "QTUM/USDT",
    "WAVES/USDT", "ZEC/USDT", "DASH/USDT", "IOTA/USDT", "OMG/USDT",
]

# Univers ultra-large : top 200 (pour scan exhaustif - exige interval >=120s)
DEFAULT_UNIVERSE_200: list[str] = DEFAULT_UNIVERSE_100 + [
    "PEOPLE/USDT", "C98/USDT", "SXP/USDT", "CELR/USDT", "RVN/USDT",
    "REN/USDT", "BAL/USDT", "YFI/USDT", "KSM/USDT", "ZEN/USDT",
    "DENT/USDT", "HOT/USDT", "ONE/USDT", "HOOK/USDT", "STG/USDT",
    "ID/USDT", "PEOPLE/USDT", "ACH/USDT", "ARPA/USDT", "BNX/USDT",
    "BAKE/USDT", "TLM/USDT", "ALPACA/USDT", "ALICE/USDT", "AUDIO/USDT",
    "CTSI/USDT", "DUSK/USDT", "FXS/USDT", "GLM/USDT", "ILV/USDT",
    "KEY/USDT", "KNC/USDT", "LIT/USDT", "LPT/USDT", "LRC/USDT",
    "MTL/USDT", "NKN/USDT", "PERP/USDT", "POLY/USDT", "POWR/USDT",
    "REI/USDT", "REQ/USDT", "RIF/USDT", "RSR/USDT", "SC/USDT",
    "SFP/USDT", "SKL/USDT", "SLP/USDT", "SPELL/USDT", "STMX/USDT",
    "STPT/USDT", "SUPER/USDT", "TROY/USDT", "TRU/USDT", "TWT/USDT",
    "UMA/USDT", "VITE/USDT", "VTHO/USDT", "WAXP/USDT", "WIN/USDT",
    "WRX/USDT", "XEC/USDT", "XEM/USDT", "XNO/USDT", "XVS/USDT",
    "YGG/USDT", "ZRX/USDT", "FRONT/USDT", "BURGER/USDT", "AGLD/USDT",
    "ASR/USDT", "AVA/USDT", "BAR/USDT", "BICO/USDT", "BTS/USDT",
    "CAKE/USDT", "CITY/USDT", "CKB/USDT", "CLV/USDT", "CTK/USDT",
    "CVC/USDT", "CVP/USDT", "DAR/USDT", "DGB/USDT", "DODO/USDT",
    "ELF/USDT", "EPX/USDT", "FIO/USDT", "FIS/USDT", "FLM/USDT",
    "FORTH/USDT", "FUN/USDT", "GAS/USDT", "GHST/USDT", "GMT/USDT",
    "GNO/USDT", "GTC/USDT", "HARD/USDT", "HFT/USDT", "HIGH/USDT",
]
# Dedup tout en preservant ordre
DEFAULT_UNIVERSE_200 = list(dict.fromkeys(DEFAULT_UNIVERSE_200))

DEFAULT_SCAN_TIMEFRAMES: list[str] = ["15m", "1h", "4h"]


def get_universe(name: str = "50") -> list[str]:
    """Helper pour selectionner par nom : '50', '100', '200', ou liste libre."""
    mapping = {
        "50": DEFAULT_UNIVERSE_50,
        "100": DEFAULT_UNIVERSE_100,
        "200": DEFAULT_UNIVERSE_200,
    }
    return mapping.get(str(name), DEFAULT_UNIVERSE_50)


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
