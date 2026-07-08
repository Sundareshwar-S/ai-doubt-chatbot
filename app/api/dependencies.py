"""FastAPI `Depends()` getters reading from `request.app.state` (Phase 5).

Unlike `app/index`'s and `app/qa`'s "build a default if missing" DI pattern
(meant for standalone, independently testable functions), these getters are
pure reads off state assembled once in `main.py`'s lifespan -- that is what
makes "built once at startup, not per request" structurally guaranteed rather
than incidentally true. Tests override `get_state` via
`app.dependency_overrides` to inject a `tmp_path`-backed `AppState`.
"""
from __future__ import annotations

from fastapi import Request

from app.api.state import AppState


def get_state(request: Request) -> AppState:
    return request.app.state.app_state
