"""FastAPI dependency injection utilities."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.session import get_session

logger = logging.getLogger(__name__)

SessionDep = Annotated[AsyncSession, Depends(get_session)]

_INSECURE_DEFAULTS = {"", "changeme"}
_warned_insecure = False


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> str:
    """Validate the X-API-Key header.

    Behavior:
      - If ``settings.api_key`` is unset or left at the default "changeme",
        requests are allowed but a startup-style warning is logged once
        (so the local dashboard keeps working out of the box).
      - If a real key is configured, the header MUST match exactly.
    """
    global _warned_insecure
    configured = (settings.api_key or "").strip()
    if configured in _INSECURE_DEFAULTS:
        if not _warned_insecure:
            logger.warning(
                "API_KEY is unset or 'changeme' — protected endpoints are unauthenticated. "
                "Set a strong API_KEY in .env to enforce authentication."
            )
            _warned_insecure = True
        return configured
    if x_api_key is None or x_api_key != configured:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing X-API-Key header",
        )
    return x_api_key


ApiKeyDep = Annotated[str, Depends(verify_api_key)]
