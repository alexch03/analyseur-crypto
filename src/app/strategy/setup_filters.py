"""Optional filters on setups (IFVG, RSI divergence)."""

from __future__ import annotations

from app.schemas.domain import FVGType, FairValueGap, Side, TradeSetupDTO


def entry_near_ifvg_zone(
    entry: float,
    ifvgs: list[FairValueGap],
    *,
    fvg_type: FVGType,
    proximity_pct: float,
) -> bool:
    """Entrée dans la zone IFVG ou à proximité relative du milieu du gap."""
    if proximity_pct < 0:
        return False
    for z in ifvgs:
        if z.fvg_type != fvg_type:
            continue
        if z.bottom <= entry <= z.top:
            return True
        mid = (z.top + z.bottom) * 0.5
        if mid > 0 and abs(entry - mid) / mid <= proximity_pct:
            return True
    return False


def setup_passes_ifvg_filter(
    setup: TradeSetupDTO,
    ifvgs: list[FairValueGap],
    proximity_pct: float,
) -> bool:
    """Long : IFVG haussier (zone ex-bearish mitigée). Short : IFVG baissier."""
    if setup.side == Side.LONG:
        return entry_near_ifvg_zone(setup.entry, ifvgs, fvg_type=FVGType.BULLISH, proximity_pct=proximity_pct)
    return entry_near_ifvg_zone(setup.entry, ifvgs, fvg_type=FVGType.BEARISH, proximity_pct=proximity_pct)


def setup_passes_rsi_divergence_filter(setup: TradeSetupDTO, indicator_tags: dict[str, bool]) -> bool:
    if setup.side == Side.LONG:
        return bool(indicator_tags.get("rsi_bull_div_recent"))
    return bool(indicator_tags.get("rsi_bear_div_recent"))
