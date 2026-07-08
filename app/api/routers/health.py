"""`GET /health` -- Ollama reachability + model-presence check (Phase 5, T5.7)."""
from __future__ import annotations

from fastapi import APIRouter, Response

from app.api.schemas.health import HealthResponse
from app.api.services.ollama_health import check_ollama_health

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def get_health(response: Response) -> HealthResponse:
    status = await check_ollama_health()
    if not status.ok:
        response.status_code = 503
    return HealthResponse(status="ok" if status.ok else "degraded", detail=status.detail)
