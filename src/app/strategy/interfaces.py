"""Protocol interfaces for the strategy engine."""

from __future__ import annotations

from typing import Protocol

from app.schemas.domain import MarketContextDTO, TradeSetupDTO


class SetupEngine(Protocol):
    def propose(self, ctx: MarketContextDTO) -> list[TradeSetupDTO]: ...
