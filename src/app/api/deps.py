"""FastAPI dependency injection utilities."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.session import get_session

SessionDep = Annotated[AsyncSession, Depends(get_session)]


async def verify_api_key(x_api_key: str = Header(...)) -> str:
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key")
    return x_api_key


ApiKeyDep = Annotated[str, Depends(verify_api_key)]
