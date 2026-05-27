"""Protocol interfaces for market structure analysis."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from app.schemas.domain import SRLevel, StructureEvent, SwingPoint


class SwingDetector(Protocol):
    def detect(self, ohlcv: pd.DataFrame, *, left: int, right: int) -> list[SwingPoint]: ...


class SupportResistanceDetector(Protocol):
    def levels(
        self, ohlcv: pd.DataFrame, swings: list[SwingPoint], *, atr_mult: float
    ) -> list[SRLevel]: ...


class StructureAnalyzer(Protocol):
    def analyze(
        self, swings: list[SwingPoint], closes: pd.Series
    ) -> list[StructureEvent]: ...
