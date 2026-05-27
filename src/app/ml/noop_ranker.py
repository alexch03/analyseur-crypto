"""No-op ranker: passes setups through unranked until ML is implemented (phase H)."""

from __future__ import annotations

from app.schemas.domain import RankedSetupDTO, TradeSetupDTO


class NoOpRanker:
    def rank(self, setups: list[TradeSetupDTO]) -> list[RankedSetupDTO]:
        return [RankedSetupDTO(setup=s, ml_score=None) for s in setups]
