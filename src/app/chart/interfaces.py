"""Protocol interfaces for chart rendering DTOs."""

from __future__ import annotations

from typing import Any, Protocol


class ChartDTOBuilder(Protocol):
    def build(self, symbol: str, timeframe: str) -> dict[str, Any]: ...
