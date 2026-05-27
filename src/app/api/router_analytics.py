"""API analytics : breakdowns par pattern, tag, score, etc."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.deps import SessionDep
from app.services.analytics import compute_breakdowns, optimize_filters

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/breakdown")
async def breakdown(session: SessionDep) -> dict:
    """Decomposition des trades cloturés par segment.

    Trie chaque section par expectancy decroissante : en tete = ce qui marche,
    en queue = ce qu'il faut filtrer ou debugger.
    """
    return await compute_breakdowns(session)


@router.get("/optimize")
async def optimize(
    session: SessionDep,
    top_n: int = Query(20, ge=5, le=100),
) -> dict:
    """Grid search sur les filtres (min_score, reject_tags, required_tags).

    Simule virtuellement chaque combinaison sur les trades cloturés existants.
    Retourne les meilleures configs par cumul compound.
    """
    return await optimize_filters(session, top_n=top_n)
