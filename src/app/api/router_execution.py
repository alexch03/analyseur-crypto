"""API execution : statut, emergency stop, reset killswitch, balance."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.deps import ApiKeyDep
from app.execution.router import get_executor, get_safety

router = APIRouter(prefix="/execution", tags=["execution"])


@router.get("/status")
async def status(_auth: ApiKeyDep) -> dict:
    """Status de l'executor actif + safety guards."""
    safety = get_safety()
    executor = await get_executor()
    try:
        balance = await executor.fetch_balance()
    except Exception as e:
        balance = {"error": str(e)}
    try:
        positions = await executor.fetch_open_positions()
    except Exception as e:
        positions = [{"error": str(e)}]
    return {
        "executor": executor.name,
        "mode": safety.config.mode,
        "killswitch": safety.killswitch_tripped(),
        "killswitch_reason": safety.state.killswitch_reason,
        "daily_pnl_usd": safety.state.daily_pnl_usd,
        "consecutive_losses": safety.state.consecutive_losses,
        "max_position_usd": safety.config.max_position_usd,
        "max_daily_loss_usd": safety.config.max_daily_loss_usd,
        "max_open_positions": safety.config.max_open_positions,
        "balance": balance,
        "open_positions_count": len(positions) if isinstance(positions, list) else 0,
        "open_positions": positions if isinstance(positions, list) else [],
    }


@router.post("/emergency_stop")
async def emergency_stop(_auth: ApiKeyDep) -> dict:
    """Ferme TOUTES les positions ouvertes immediatement (urgence).

    Trigger le killswitch pour empecher toute nouvelle ouverture.
    """
    safety = get_safety()
    executor = await get_executor()
    safety._trip_killswitch("manual emergency stop via API")
    closed = []
    try:
        if hasattr(executor, "_client"):
            results = await executor._client.close_all_positions()
            return {"ok": True, "closed": results, "killswitch": True}
    except Exception as e:
        raise HTTPException(500, f"emergency_stop failed: {e}")
    return {"ok": True, "closed": closed, "killswitch": True,
            "info": "no exchange client, only killswitch tripped"}


@router.post("/reset_killswitch")
async def reset_killswitch(_auth: ApiKeyDep) -> dict:
    """Reset manuel du killswitch apres review humaine."""
    safety = get_safety()
    safety.reset_killswitch()
    return {"ok": True, "status": safety.status_text()}


@router.get("/safety")
async def safety_status(_auth: ApiKeyDep) -> dict:
    safety = get_safety()
    return {
        "mode": safety.config.mode,
        "killswitch_tripped": safety.killswitch_tripped(),
        "killswitch_reason": safety.state.killswitch_reason,
        "daily_pnl_usd": safety.state.daily_pnl_usd,
        "consecutive_losses": safety.state.consecutive_losses,
        "config": {
            "max_position_usd": safety.config.max_position_usd,
            "max_open_positions": safety.config.max_open_positions,
            "max_daily_loss_usd": safety.config.max_daily_loss_usd,
            "max_consecutive_losses": safety.config.max_consecutive_losses,
            "blacklist_symbols": safety.config.blacklist_symbols,
            "allowed_sides": list(safety.config.allowed_sides),
            "min_balance_usd": safety.config.min_balance_usd,
        },
    }
