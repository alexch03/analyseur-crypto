"""Market regime API: current + history."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.deps import SessionDep
from app.services.market_regime import (
    PATTERN_REGIME_AFFINITY, pattern_regime_score,
)
from app.services.regime_tracker import get_regime_tracker

router = APIRouter(prefix="/regime", tags=["regime"])


@router.get("/current")
async def current_regime(session: SessionDep) -> dict:
    """Regime detecte courant (charge depuis DB si memoire vide)."""
    tracker = get_regime_tracker()
    regime = tracker.current()
    if regime is None:
        regime = await tracker.load_latest_from_db(session)
    if regime is None:
        return {"regime": None, "info": "no snapshot yet, scanner not running?"}

    # Calcule l'affinity de chaque pattern avec le regime courant
    pattern_scores = {
        pat: round(pattern_regime_score(pat, regime), 2)
        for pat in PATTERN_REGIME_AFFINITY.keys()
    }
    favored = sorted(pattern_scores.items(), key=lambda x: -x[1])[:5]
    rejected = sorted(pattern_scores.items(), key=lambda x: x[1])[:5]

    return {
        "regime": regime.as_dict(),
        "patterns_favored": [{"pattern": p, "score": s} for p, s in favored],
        "patterns_rejected": [{"pattern": p, "score": s} for p, s in rejected],
        "all_pattern_scores": pattern_scores,
    }


@router.get("/history")
async def regime_history(
    session: SessionDep,
    limit: int = Query(100, ge=10, le=1000),
) -> dict:
    """Historique des snapshots regime (du plus recent au plus ancien)."""
    tracker = get_regime_tracker()
    items = await tracker.fetch_history(session, limit=limit)
    # Compute regime transitions
    transitions: list[dict] = []
    for i in range(len(items) - 1):
        prev = items[i + 1]
        cur = items[i]
        if cur["trend"] != prev["trend"] or cur["volatility"] != prev["volatility"]:
            transitions.append({
                "ts": cur["ts"],
                "from": f"{prev['trend']}/{prev['volatility']}",
                "to": f"{cur['trend']}/{cur['volatility']}",
            })
    return {
        "items": items,
        "transitions": transitions[:20],
        "count": len(items),
    }


@router.get("/affinity")
async def regime_affinity() -> dict:
    """Retourne la table d'affinity pattern x regime (pour comprehension UI)."""
    return {"affinity": PATTERN_REGIME_AFFINITY}
