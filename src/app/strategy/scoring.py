"""Rule-based confidence scoring with confluence bonuses (v2).

Base confidence from the setup engine is intentionally low (0.48 – 0.62).
Bonuses are additive but **non-redundant**: a bonus is only awarded if the
confluence is not already baked into the setup type itself.

Maximum confidence is capped at 0.92 to preserve ranking granularity.
"""

from __future__ import annotations

from app.schemas.domain import (
    FVGType,
    MarketContextDTO,
    OBType,
    Side,
    TradeSetupDTO,
)

_CAP = 0.92

_SETUP_TYPES_WITH_FVG = frozenset({"FVG_FILL", "OB_FVG_CONFLUENCE", "IFVG_REACTION"})
_SETUP_TYPES_WITH_OB = frozenset({"OB_REJECTION", "OB_FVG_CONFLUENCE"})


class RuleBasedScorer:
    def score(self, setup: TradeSetupDTO, ctx: MarketContextDTO) -> float:
        conf = setup.confidence

        # +0.10 — aligned with trend (most valuable confluence)
        if setup.side.value == "LONG" and ctx.trend.value == "BULLISH":
            conf += 0.10
        elif setup.side.value == "SHORT" and ctx.trend.value == "BEARISH":
            conf += 0.10

        # +0.06 — entry sits inside an unmitigated FVG (skip if setup already uses FVG)
        if setup.setup_type not in _SETUP_TYPES_WITH_FVG:
            for fvg in ctx.fvgs:
                if fvg.mitigated:
                    continue
                if setup.side == Side.LONG and fvg.fvg_type == FVGType.BULLISH:
                    if fvg.bottom <= setup.entry <= fvg.top:
                        conf += 0.06
                        break
                if setup.side == Side.SHORT and fvg.fvg_type == FVGType.BEARISH:
                    if fvg.bottom <= setup.entry <= fvg.top:
                        conf += 0.06
                        break

        # +0.06 — entry sits inside an unmitigated OB (skip if setup already uses OB)
        if setup.setup_type not in _SETUP_TYPES_WITH_OB:
            for ob in ctx.order_blocks:
                if ob.mitigated:
                    continue
                if setup.side == Side.LONG and ob.ob_type == OBType.BULLISH:
                    if ob.bottom <= setup.entry <= ob.top:
                        conf += 0.06
                        break
                if setup.side == Side.SHORT and ob.ob_type == OBType.BEARISH:
                    if ob.bottom <= setup.entry <= ob.top:
                        conf += 0.06
                        break

        # +0.04 — S/R level with multiple touches near entry
        for lv in ctx.sr_levels:
            if lv.touches >= 3 and abs(lv.price - setup.entry) / max(setup.entry, 1e-9) < 0.005:
                conf += 0.04
                break

        # +0.04 — high R:R
        if setup.risk_reward >= 3.0:
            conf += 0.04

        # +0.04 — RSI divergence aligned with trade direction
        tags = ctx.indicator_tags or {}
        if setup.side.value == "LONG" and tags.get("rsi_bull_div_recent"):
            conf += 0.04
        elif setup.side.value == "SHORT" and tags.get("rsi_bear_div_recent"):
            conf += 0.04

        return min(conf, _CAP)
