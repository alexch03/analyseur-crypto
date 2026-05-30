"""API admin : reset DB, etc."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import delete

from app.api.deps import ApiKeyDep, SessionDep
from app.db.models import Hypothesis, ScanRun, UnitTrade

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/clean-db")
async def clean_db(
    session: SessionDep,
    _auth: ApiKeyDep,
    confirm: str = Query("", description="Doit valoir 'yes' pour proceder."),
    keep_universe: bool = Query(True, description="Garde exchanges/symbols/timeframes/candles"),
) -> dict:
    """Vide les tables d'hypotheses, unit_trades et scan_runs.

    `confirm=yes` requis pour eviter les appels accidentels.
    Les tables 'universe' (exchanges/symbols/timeframes/candles) sont conservees
    par defaut pour ne pas perdre l'historique OHLCV.
    """
    if confirm != "yes":
        raise HTTPException(status_code=400, detail="Pass confirm=yes to proceed.")

    deleted_unit = (await session.execute(delete(UnitTrade))).rowcount or 0
    deleted_hyp = (await session.execute(delete(Hypothesis))).rowcount or 0
    deleted_scan = (await session.execute(delete(ScanRun))).rowcount or 0
    await session.commit()
    return {
        "deleted_unit_trades": deleted_unit,
        "deleted_hypotheses": deleted_hyp,
        "deleted_scan_runs": deleted_scan,
        "kept_universe": keep_universe,
    }
