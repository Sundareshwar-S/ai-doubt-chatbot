"""Response schema for `GET /health` (Phase 5)."""
from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    detail: str | None = None
