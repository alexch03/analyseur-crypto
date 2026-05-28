"""Recupere TOUS les symboles USDT actifs sur Bitget via ccxt fetch_tickers.

Sauvegarde dans data/bitget_symbols.json + JSON groupes par categorie de volume.

Usage :
    python scripts/fetch_bitget_symbols.py             # tous USDT
    python scripts/fetch_bitget_symbols.py --top 200   # top 200 par volume 24h
    python scripts/fetch_bitget_symbols.py --futures   # USDT-FUTURES (pour trading)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import ccxt.async_support as ccxt_async  # noqa: E402

OUT_DIR = ROOT / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE_SPOT = OUT_DIR / "bitget_symbols_spot.json"
OUT_FILE_FUTURES = OUT_DIR / "bitget_symbols_futures.json"


async def fetch_bitget_symbols(
    *, top_n: int | None = None, futures: bool = False,
) -> list[dict]:
    """Fetch tous les symboles USDT actifs sur Bitget avec leur volume 24h.

    futures=True : USDT-M perpetuals (BTC/USDT:USDT)
    futures=False : spot (BTC/USDT)
    """
    ex_cls = ccxt_async.bitget
    opts = {"enableRateLimit": True}
    if futures:
        opts["options"] = {"defaultType": "swap", "defaultSubType": "linear"}
    ex = ex_cls(opts)

    print(f"Connexion Bitget ({'futures' if futures else 'spot'})...")
    try:
        await ex.load_markets()
        tickers = await ex.fetch_tickers()
    finally:
        await ex.close()

    print(f"  {len(tickers)} tickers recus, filtrage USDT...")

    out: list[dict] = []
    for sym, t in tickers.items():
        if "/" not in sym:
            continue
        # Pour futures, sym = "BTC/USDT:USDT"
        if futures and ":USDT" not in sym:
            continue
        if not futures and ":" in sym:
            continue
        base = sym.split("/", 1)[0]
        quote = sym.split("/", 1)[1].split(":")[0]
        if quote != "USDT" or not base:
            continue
        info = t or {}
        qv = info.get("quoteVolume") or 0.0
        try:
            qv_f = float(qv)
        except (TypeError, ValueError):
            qv_f = 0.0
        out.append({
            "symbol": sym,
            "base": base,
            "quote_volume_usd_24h": qv_f,
            "last": float(info.get("last") or 0),
            "change_24h_pct": float(info.get("percentage") or 0),
        })

    out.sort(key=lambda x: x["quote_volume_usd_24h"], reverse=True)
    if top_n:
        out = out[:top_n]
    return out


def categorize_by_volume(symbols: list[dict]) -> dict:
    """Groupe par tranche de volume 24h."""
    tiers = {
        "tier1_high_volume": [],     # > 50M$
        "tier2_medium_volume": [],   # 10M-50M$
        "tier3_low_volume": [],      # 1M-10M$
        "tier4_micro": [],           # <1M$
    }
    for s in symbols:
        v = s["quote_volume_usd_24h"]
        if v > 50_000_000:
            tiers["tier1_high_volume"].append(s["symbol"])
        elif v > 10_000_000:
            tiers["tier2_medium_volume"].append(s["symbol"])
        elif v > 1_000_000:
            tiers["tier3_low_volume"].append(s["symbol"])
        else:
            tiers["tier4_micro"].append(s["symbol"])
    return tiers


async def main(args) -> None:
    symbols = await fetch_bitget_symbols(top_n=args.top, futures=args.futures)

    tiers = categorize_by_volume(symbols)
    payload = {
        "exchange": "bitget",
        "market_type": "futures" if args.futures else "spot",
        "total_symbols": len(symbols),
        "tiers": {k: len(v) for k, v in tiers.items()},
        "symbols_all": [s["symbol"] for s in symbols],
        "symbols_top50": [s["symbol"] for s in symbols[:50]],
        "symbols_top100": [s["symbol"] for s in symbols[:100]],
        "symbols_top200": [s["symbol"] for s in symbols[:200]],
        "tiered": tiers,
        "details": symbols[:50],  # top 50 details (volume, prix)
    }

    out_file = OUT_FILE_FUTURES if args.futures else OUT_FILE_SPOT
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print("=" * 60)
    print(f"BITGET {'FUTURES' if args.futures else 'SPOT'} - SYMBOLES USDT")
    print("=" * 60)
    print(f"Total      : {len(symbols)}")
    print(f"Tier 1 (>50M$/24h)  : {len(tiers['tier1_high_volume'])}")
    print(f"Tier 2 (10-50M$/24h): {len(tiers['tier2_medium_volume'])}")
    print(f"Tier 3 (1-10M$/24h) : {len(tiers['tier3_low_volume'])}")
    print(f"Tier 4 (<1M$/24h)   : {len(tiers['tier4_micro'])}")
    print()
    print(f"Top 10 par volume :")
    for s in symbols[:10]:
        v = s["quote_volume_usd_24h"]
        v_str = f"{v/1e6:.1f}M$" if v > 1e6 else f"{v/1e3:.0f}k$"
        print(f"  {s['symbol']:<22} vol24h={v_str:<10} chg={s['change_24h_pct']:+6.2f}%")
    print()
    print(f"-> Sauvegarde : {out_file}")
    print()
    print("Pour utiliser dans .env :")
    prefix = "bitget_futures" if args.futures else "bitget_spot"
    print(f"  SCAN_UNIVERSE={prefix}_tier12  # tous les volume > 10M$")
    print(f"  SCAN_UNIVERSE={prefix}_top100  # top 100 par volume")
    print(f"  SCAN_UNIVERSE={prefix}_top200  # top 200 par volume")


def parse_args():
    p = argparse.ArgumentParser(description="Fetch symboles Bitget")
    p.add_argument("--top", type=int, default=None,
                    help="Limite au top N par volume (defaut: tous)")
    p.add_argument("--futures", action="store_true",
                    help="USDT-FUTURES au lieu de spot")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
