"""Local web dashboards (SMC control + patterns chartistes)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["dashboard"])

_WEB_DIR = Path(__file__).resolve().parents[1] / "web"


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    page = _WEB_DIR / "dashboard.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Dashboard page not found")
    return page.read_text(encoding="utf-8")


@router.get("/patterns", response_class=HTMLResponse)
async def patterns_dashboard():
    """Dashboard dédié au scanner de patterns chartistes (hypothèses + unit_paper)."""
    page = _WEB_DIR / "patterns_dashboard.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Patterns dashboard not found")
    return page.read_text(encoding="utf-8")
