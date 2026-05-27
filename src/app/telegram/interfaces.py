"""Protocol interfaces for Telegram notifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.schemas.domain import TradeSetupDTO


@dataclass(frozen=True, slots=True)
class DeliveryReceipt:
    signal_id: int
    chat_id: str
    success: bool
    error: str | None = None


class SignalNotifier(Protocol):
    async def send(self, setup: TradeSetupDTO) -> DeliveryReceipt: ...
