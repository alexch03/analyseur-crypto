"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response

from app.db.session import engine
from app.logging_setup import setup_logging
from app.services import paper_live_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    await paper_live_service.reconcile_paper_live_task_with_state()
    yield
    await paper_live_service.stop_paper_live()
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Analyseur Crypto",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from app.api.router_backtest import router as backtest_router
    from app.api.router_candles import router as candles_router
    from app.api.router_chart import router as chart_router
    from app.api.router_control import router as control_router
    from app.api.router_dashboard import router as dashboard_router
    from app.api.router_health import router as health_router
    from app.api.router_hypotheses import router as hypotheses_router
    from app.api.router_admin import router as admin_router
    from app.api.router_analytics import router as analytics_router
    from app.api.router_execution import router as execution_router
    from app.api.router_ingestion import router as ingestion_router
    from app.api.router_scan import router as scan_router
    from app.api.router_scanner_ops import router as scanner_ops_router
    from app.api.router_signals import router as signals_router
    from app.api.router_unit_paper import router as unit_paper_router

    app.include_router(health_router, prefix="/api/v1")
    app.include_router(candles_router, prefix="/api/v1")
    app.include_router(signals_router, prefix="/api/v1")
    app.include_router(chart_router, prefix="/api/v1")
    app.include_router(ingestion_router, prefix="/api/v1")
    app.include_router(scan_router, prefix="/api/v1")
    app.include_router(backtest_router, prefix="/api/v1")
    app.include_router(control_router, prefix="/api/v1")
    app.include_router(hypotheses_router, prefix="/api/v1")
    app.include_router(unit_paper_router, prefix="/api/v1")
    app.include_router(scanner_ops_router, prefix="/api/v1")
    app.include_router(analytics_router, prefix="/api/v1")
    app.include_router(execution_router, prefix="/api/v1")
    app.include_router(admin_router, prefix="/api/v1")
    app.include_router(dashboard_router)

    @app.get("/", include_in_schema=False)
    async def root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/dashboard", status_code=307)

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

    return app


app = create_app()
