"""FastAPI app factory + lifespan (Phase 5).

`lifespan` builds the `VectorStoreHandle`/embed model/index/chat engine exactly
once at startup -- the FastAPI equivalent of Streamlit's `@st.cache_resource`.
Ingestion (`app/index/pipeline.ingest_file`) writes straight into the same
persisted Chroma collection this state's `index`/`chat_engine` query, so no
rebuild step is needed after an upload; only `data/manifest.json` and the
Chroma collection are shared mutable state, not the process's Python objects.

Routers are included last-before-the-static-mount so the built frontend
(mounted at `/` once `frontend/dist` exists) never shadows an API route.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

import config
from app.api.exceptions import register_exception_handlers
from app.api.routers import chat, documents, health
from app.api.state import AppState
from app.index.pipeline import default_embed_model, load_index
from app.index.store import get_vector_store_handle
from app.qa.chat import build_chat_engine


def _build_state() -> AppState:
    """Blocking setup (Chroma disk I/O, index reload, chat engine construction)
    -- run via `run_in_threadpool` so `lifespan` never blocks the event loop,
    consistent with how routers/services handle identical blocking calls.
    """
    handle = get_vector_store_handle()
    embed_model = default_embed_model()
    index = load_index(handle, embed_model=embed_model)
    chat_engine = build_chat_engine(index)
    return AppState(
        handle=handle,
        embed_model=embed_model,
        index=index,
        chat_engine=chat_engine,
        chat_lock=asyncio.Lock(),
        library_lock=asyncio.Lock(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    app.state.app_state = await run_in_threadpool(_build_state)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="AI Doubt Solver", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(chat.router)

    # Serve the built frontend last so it never shadows an API route above.
    if config.FRONTEND_DIST_DIR.exists():
        app.mount(
            "/", StaticFiles(directory=config.FRONTEND_DIST_DIR, html=True), name="frontend"
        )

    return app


app = create_app()
