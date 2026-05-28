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


def _load_bitget_symbols(market: str, key: str) -> list[str] | None:
    """Charge une liste depuis data/bitget_symbols_{market}.json[key]."""
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "data" / f"bitget_symbols_{market}.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get(key) or None
    except Exception:
        return None


def get_universe(name: str = "50") -> list[str]:
    """Selecteur d'univers par nom.

    Statique (hardcoded) :
        "50"  : DEFAULT_UNIVERSE_50  (50 majors)
        "100" : DEFAULT_UNIVERSE_100 (100 majors + alts)
        "200" : DEFAULT_UNIVERSE_200 (198 unique)

    Dynamique (depuis data/bitget_symbols_*.json fetcher) :
        "bitget_spot_top50"      : top 50 spot par volume 24h
        "bitget_spot_top100"     : top 100 spot
        "bitget_spot_top200"     : top 200 spot
        "bitget_spot_tier12"     : Tier 1+2 (volume > 10M$)
        "bitget_spot_all"        : TOUS les USDT spot (~1000+)
        "bitget_futures_top50"   : top 50 futures perpetual
        "bitget_futures_top100"  : top 100 futures
        "bitget_futures_top200"  : top 200 futures
        "bitget_futures_tier12"  : Tier 1+2 futures
        "bitget_futures_all"     : TOUS les USDT futures (~571)
    """
    name_str = str(name).lower()
    # Statiques
    static = {
        "50": DEFAULT_UNIVERSE_50,
        "100": DEFAULT_UNIVERSE_100,
        "200": DEFAULT_UNIVERSE_200,
    }
    if name_str in static:
        return static[name_str]

    # Dynamiques : "bitget_{market}_{tag}"
    if name_str.startswith("bitget_"):
        parts = name_str.split("_", 2)
        if len(parts) == 3:
            _, market, tag = parts  # ex: spot, top100
            key_map = {
                "top50": "symbols_top50",
                "top100": "symbols_top100",
                "top200": "symbols_top200",
                "tier12": "tiered",
                "all": "symbols_all",
            }
            json_key = key_map.get(tag)
            if json_key == "tiered":
                # tier12 = concat des 2 premiers tiers
                from pathlib import Path
                import json
                p = (Path(__file__).resolve().parents[2]
                     / "data" / f"bitget_symbols_{market}.json")
                if p.exists():
                    try:
                        data = json.loads(p.read_text(encoding="utf-8"))
                        tiers = data.get("tiered", {})
                        return (tiers.get("tier1_high_volume", [])
                                + tiers.get("tier2_medium_volume", []))
                    except Exception:
                        pass
                return DEFAULT_UNIVERSE_50
            if json_key:
                loaded = _load_bitget_symbols(market, json_key)
                if loaded:
                    return loaded

    # Fallback
    return DEFAULT_UNIVERSE_50


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
