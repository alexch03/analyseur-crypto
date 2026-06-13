"""Replay configuration + fee resolution (shared between control API and paper live)."""

from __future__ import annotations

from typing import Any

from app.config import settings
from app.services.exchange_market_quotes import fetch_funding_rate_8h, fetch_taker_fee_rate

_FEE_MARKET_TYPES = frozenset({"spot", "swap"})


async def resolve_trade_cost_rates(
    state: dict[str, Any],
    symbol: str,
    *,
    exchange_id: str | None = None,
) -> dict[str, Any]:
    """Frais manuels ou auto (taker CCXT) + funding manuel ou live (perp).

    ``exchange_id`` permet d'aligner les quotes sur l'exchange des bougies paper (ex. bitget)
    sans changer ``settings.exchange_id`` utilisé ailleurs.
    """
    ex = (exchange_id or settings.exchange_id).strip().lower() or str(settings.exchange_id).lower()
    entry = float(state.get("entry_fee_rate", 0.0004))
    exit_ = float(state.get("exit_fee_rate", 0.0004))
    funding = float(state.get("funding_rate_8h", state.get("funding_rate_per_bar", 0.0)))
    meta: dict[str, Any] = {
        "entry_fee_rate": entry,
        "exit_fee_rate": exit_,
        "funding_rate_8h": funding,
        "fee_auto": False,
        "funding_auto": False,
    }
    if state.get("auto_fee_from_exchange"):
        kind = str(state.get("fee_market_type", "swap")).lower()
        if kind not in _FEE_MARKET_TYPES:
            kind = "swap"
        try:
            q = await fetch_taker_fee_rate(ex, symbol, market_kind=kind)
            if q.get("taker_applied") is not None:
                entry = exit_ = float(q["taker_applied"])
                meta["fee_auto"] = True
                meta["fee_exchange_detail"] = q
        except Exception as e:  # noqa: BLE001
            meta["fee_auto_error"] = str(e)
    if state.get("auto_funding_from_exchange"):
        try:
            fr = await fetch_funding_rate_8h(ex, symbol)
            if fr.get("funding_rate_8h") is not None:
                funding = float(fr["funding_rate_8h"])
                meta["funding_auto"] = True
                meta["funding_exchange_detail"] = fr
            elif fr.get("error"):
                meta["funding_auto_error"] = fr["error"]
        except Exception as e:  # noqa: BLE001
            meta["funding_auto_error"] = str(e)
    meta["entry_fee_rate"] = entry
    meta["exit_fee_rate"] = exit_
    meta["funding_rate_8h"] = funding
    return meta


def build_replay_bt_config(state: dict[str, Any], costs: dict[str, Any]) -> dict[str, Any]:
    """Configuration du moteur replay + clés filtres pour ``engine_params``."""
    mspb = int(state.get("max_setups_per_bar", 1))
    mspb = max(1, min(10, mspb))
    cfg: dict[str, Any] = {
        "warmup_bars": int(state.get("training_bars", 120)),
        "max_holding_bars": int(state.get("max_holding_bars", 120)),
        "max_setups_per_bar": mspb,
        "unit_size": float(state.get("unit_size", 1.0)),
        "entry_fee_rate": float(costs["entry_fee_rate"]),
        "exit_fee_rate": float(costs["exit_fee_rate"]),
        "funding_rate_8h": float(costs["funding_rate_8h"]),
    }
    for k in (
        "replay_trail_after_r",
        "replay_trail_atr_mult",
        "replay_trail_atr_period",
        "replay_timeout_smart_extend",
        "replay_timeout_grace_bars",
        "replay_timeout_max_extensions",
        "replay_timeout_bb_period",
        "replay_timeout_sma_fast",
        "replay_timeout_sma_slow",
        "require_ifvg_confluence",
        "ifvg_confluence_pct",
        "require_rsi_divergence",
    ):
        if k in state:
            cfg[k] = state[k]
    return cfg
