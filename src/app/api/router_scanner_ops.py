"""Endpoints pour declencher manuellement le scanner (scan immediat + backfill).

Le scanner long-running tourne dans son propre process (worker --scan-daemon).
Ces endpoints lancent des passes a la demande depuis l'API, dans des tasks en
arriere-plan pour ne pas bloquer le client.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, UTC

from fastapi import APIRouter, BackgroundTasks, Query

from app.services.continuous_scanner import ContinuousScanner, ScanPlan
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/scanner", tags=["scanner"])


@dataclass
class _JobStatus:
    job_id: str
    kind: str
    started_at: datetime
    finished_at: datetime | None = None
    in_progress: bool = True
    result: dict | None = None
    error: str | None = None


_jobs: dict[str, _JobStatus] = {}
_lock = asyncio.Lock()


def _new_job(kind: str) -> _JobStatus:
    job_id = f"{kind}-{int(datetime.now(tz=UTC).timestamp())}"
    job = _JobStatus(job_id=job_id, kind=kind, started_at=datetime.now(tz=UTC))
    _jobs[job_id] = job
    return job


def _job_to_dict(j: _JobStatus) -> dict:
    return {
        "job_id": j.job_id,
        "kind": j.kind,
        "started_at": j.started_at.isoformat(),
        "finished_at": j.finished_at.isoformat() if j.finished_at else None,
        "in_progress": j.in_progress,
        "result": j.result,
        "error": j.error,
    }


async def _daemon_is_active() -> tuple[bool, int]:
    """Detecte si un scanner daemon tourne deja, via scan_runs recents.

    Retourne (active, seconds_since_last). Si scan_run < scan_interval, le daemon
    est actif et on ne doit pas en lancer un 2eme en parallele.
    """
    try:
        from app.db.session import async_session_factory
        from app.db.models import ScanRun
        from sqlalchemy import select, func
        async with async_session_factory() as session:
            q = select(func.max(ScanRun.ts_finished))
            last = (await session.execute(q)).scalar_one_or_none()
            if last is None:
                return False, 999999
            # Normalize tz
            from datetime import timezone
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            now = datetime.now(tz=UTC)
            elapsed = int((now - last).total_seconds())
            # Si dernier scan < interval, daemon actif
            return elapsed < int(settings.scan_interval_seconds * 2), elapsed
    except Exception:
        return False, 999999


async def _run_scan_once(job: _JobStatus, symbols: list[str] | None, timeframes: list[str] | None) -> None:
    try:
        # GARDE : si le daemon scanne deja, on n'en lance pas un 2eme (evite doublons)
        active, elapsed = await _daemon_is_active()
        if active:
            interval = int(settings.scan_interval_seconds)
            eta = max(0, interval - elapsed)
            job.result = {
                "skipped": True,
                "reason": "scanner daemon deja actif",
                "last_scan_seconds_ago": elapsed,
                "next_cycle_eta_seconds": eta,
                "message": f"Le scanner daemon tourne en boucle (cycle {interval}s). Prochain scan dans ~{eta}s.",
            }
            return

        # Sinon (daemon non lance), on fait un vrai scan manuel
        plan = ScanPlan(
            symbols=symbols or settings.effective_scan_symbols(),
            timeframes=timeframes or settings.effective_scan_timeframes(),
            interval_seconds=int(settings.scan_interval_seconds),
        )
        scanner = ContinuousScanner(plan=plan)
        try:
            result = await scanner.scan_once()
        finally:
            await scanner.stop()
        job.result = result
    except Exception as e:
        logger.exception("scan-once job failed")
        job.error = str(e)
    finally:
        job.in_progress = False
        job.finished_at = datetime.now(tz=UTC)


async def _run_backfill(
    job: _JobStatus,
    symbols: list[str] | None,
    timeframes: list[str] | None,
    history_bars: int,
    bars_per_step: int,
) -> None:
    try:
        plan = ScanPlan(
            symbols=symbols or settings.effective_scan_symbols(),
            timeframes=timeframes or settings.effective_scan_timeframes(),
            interval_seconds=int(settings.scan_interval_seconds),
        )
        scanner = ContinuousScanner(plan=plan)
        try:
            result = await scanner.backfill(
                bars_per_step=bars_per_step,
                history_bars=history_bars,
                symbols=symbols,
                timeframes=timeframes,
            )
        finally:
            await scanner.stop()
        job.result = result
    except Exception as e:
        logger.exception("backfill job failed")
        job.error = str(e)
    finally:
        job.in_progress = False
        job.finished_at = datetime.now(tz=UTC)


def _parse_csv(s: str | None) -> list[str] | None:
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


@router.post("/scan-now")
async def scan_now(
    background: BackgroundTasks,
    symbols: str | None = Query(None, description="CSV ex: BTC/USDT,ETH/USDT (vide = tous)"),
    timeframes: str | None = Query(None, description="CSV ex: 15m,1h"),
) -> dict:
    """Declenche une passe immediate sur tous les symbols/TF (ou subset)."""
    job = _new_job("scan_once")
    background.add_task(_run_scan_once, job, _parse_csv(symbols), _parse_csv(timeframes))
    return _job_to_dict(job)


@router.post("/backfill")
async def backfill(
    background: BackgroundTasks,
    history_bars: int = Query(250, ge=60, le=1000, description="Nombre de bougies a rejouer"),
    bars_per_step: int = Query(1, ge=1, le=5, description=">1 = saute des bougies pour aller plus vite"),
    symbols: str | None = Query(None, description="CSV de symbols (vide = tous)"),
    timeframes: str | None = Query(None, description="CSV de TFs (vide = tous : 15m,1h,4h)"),
) -> dict:
    """Rejoue l'historique pour reconstruire un cumul % et trades cloturés.

    Astuce : commence avec un subset (`symbols=BTC/USDT,ETH/USDT&timeframes=1h`) pour tester
    rapidement, puis lance le plein backfill une fois rassuré.
    """
    job = _new_job("backfill")
    background.add_task(
        _run_backfill,
        job,
        _parse_csv(symbols),
        _parse_csv(timeframes),
        history_bars,
        bars_per_step,
    )
    return _job_to_dict(job)


@router.get("/jobs")
async def list_jobs(limit: int = Query(20, ge=1, le=100)) -> list[dict]:
    items = sorted(_jobs.values(), key=lambda j: j.started_at, reverse=True)[:limit]
    return [_job_to_dict(j) for j in items]


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    j = _jobs.get(job_id)
    if j is None:
        return {"error": "not_found", "job_id": job_id}
    return _job_to_dict(j)
