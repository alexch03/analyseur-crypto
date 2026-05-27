"""Protocol interfaces for ML ranking (stub)."""

from __future__ import annotations

from typing import Protocol

from app.schemas.domain import RankedSetupDTO, TradeSetupDTO


class SetupRanker(Protocol):
    def rank(self, setups: list[TradeSetupDTO]) -> list[RankedSetupDTO]: ...
