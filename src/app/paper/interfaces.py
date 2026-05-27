"""Protocol interfaces for paper trading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from app.schemas.domain import TradeSetupDTO


@dataclass(frozen=True, slots=True)
class PaperOrderResult:
    order_id: int
    status: str


class PaperTradingEngine(Protocol):
    def on_signal(
        self, setup: TradeSetupDTO, *, mode: Literal["live", "replay"]
    ) -> PaperOrderResult: ...
