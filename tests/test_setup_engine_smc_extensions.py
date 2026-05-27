"""Nouveaux setups SMC : OB+FVG confluence, IFVG."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from app.schemas.domain import (
    FVGType,
    FairValueGap,
    MarketContextDTO,
    OBType,
    OrderBlock,
    Trend,
)
from app.strategy.setup_engine import RuleBasedSetupEngine


def _minimal_ctx(
    *,
    close: float,
    fvgs: list[FairValueGap],
    order_blocks: list[OrderBlock],
    ifvgs: list[FairValueGap] | None = None,
) -> MarketContextDTO:
    ts = datetime(2025, 1, 1, 12, tzinfo=UTC)
    df = pd.DataFrame({"timestamp": [ts], "close": [close]})
    return MarketContextDTO(
        symbol="TEST/USDT",
        timeframe="1h",
        ohlcv=df,
        swings=[],
        sr_levels=[],
        structure_events=[],
        trend=Trend.BULLISH,
        fvgs=fvgs,
        order_blocks=order_blocks,
        ifvgs=ifvgs or [],
    )


def test_ob_fvg_confluence_bullish():
    ts = datetime(2025, 1, 1, 12, tzinfo=UTC)
    fvg = FairValueGap(index=1, timestamp=ts, top=103.0, bottom=99.0, fvg_type=FVGType.BULLISH)
    ob = OrderBlock(index=2, timestamp=ts, top=102.0, bottom=100.0, ob_type=OBType.BULLISH)
    ctx = _minimal_ctx(close=101.0, fvgs=[fvg], order_blocks=[ob])
    eng = RuleBasedSetupEngine(rr_min=2.0, max_setups=10, fvg_proximity_pct=0.05, ob_proximity_pct=0.05)
    out = eng.propose(ctx)
    assert any(s.setup_type == "OB_FVG_CONFLUENCE" and s.side.value == "LONG" for s in out)


def test_ifvg_reaction_bullish():
    ts = datetime(2025, 1, 1, 12, tzinfo=UTC)
    ifvg = FairValueGap(index=0, timestamp=ts, top=102.0, bottom=98.0, fvg_type=FVGType.BULLISH)
    ctx = _minimal_ctx(close=99.0, fvgs=[], order_blocks=[], ifvgs=[ifvg])
    eng = RuleBasedSetupEngine(rr_min=2.0, max_setups=10, fvg_proximity_pct=0.05, ob_proximity_pct=0.05)
    out = eng.propose(ctx)
    assert any(s.setup_type == "IFVG_REACTION" and s.side.value == "LONG" for s in out)
