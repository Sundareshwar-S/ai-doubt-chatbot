"""Ollama reachability + model-presence check for `GET /health` (Phase 5).

Ollama's `/api/tags` returns each model's *full* tag, appending `:latest`
when a model was pulled without an explicit tag (e.g. `nomic-embed-text` pulled
bare shows up as `"nomic-embed-text:latest"`) -- `_model_present` accounts for
that so a config entry with no `:tag` suffix still matches.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx

import config


@dataclass(frozen=True)
class HealthStatus:
    ok: bool
    detail: str | None = None


def _model_present(configured_model: str, served_names: set[str]) -> bool:
    if configured_model in served_names:
        return True
    if ":" not in configured_model:
        return f"{configured_model}:latest" in served_names
    return False


async def check_ollama_health() -> HealthStatus:
    """Ping Ollama and confirm the configured LLM + embedding models are pulled.

    Distinguishes "Ollama isn't running" from "Ollama is up but a configured
    model hasn't been pulled" -- both need a different, actionable `detail`
    (T5.7's DoD, hardened further in T6.3).
    """
    try:
        async with httpx.AsyncClient(timeout=config.HEALTH_CHECK_TIMEOUT_SECONDS) as client:
            response = await client.get(f"{config.OLLAMA_BASE_URL}/api/tags")
    except httpx.RequestError:
        return HealthStatus(
            ok=False,
            detail=f"Cannot reach Ollama at {config.OLLAMA_BASE_URL}. Is it running?",
        )

    if response.status_code != 200:
        return HealthStatus(ok=False, detail=f"Ollama returned HTTP {response.status_code}.")

    served_names = {m.get("name", "") for m in response.json().get("models", [])}
    missing = [
        model
        for model in (config.LLM_MODEL, config.EMBED_MODEL)
        if not _model_present(model, served_names)
    ]
    if missing:
        return HealthStatus(
            ok=False,
            detail=(
                f"Model(s) not pulled: {', '.join(missing)}. "
                f"Run `ollama pull {missing[0]}`."
            ),
        )

    return HealthStatus(ok=True)
