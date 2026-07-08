"""Process-wide application state, assembled once at FastAPI startup (Phase 5).

A single `AppState` instance lives at `request.app.state.app_state`, built by
`main.py`'s lifespan and never rebuilt per request -- the FastAPI equivalent of
Streamlit's `@st.cache_resource`. `chat_engine` is a single shared conversation
(this is a personal, single-user, no-auth tool, not a multi-tenant one); the
paired `chat_lock` serializes access to its mutable `ChatMemoryBuffer` so two
concurrent requests (e.g. two browser tabs) can't interleave writes to it.
`library_lock` serializes access to the other piece of shared mutable state
the API exposes to concurrent requests: `data/manifest.json` (a
read-all/mutate-list/write-all cycle with no other concurrency guard) and the
Chroma collection's add/delete calls -- without it, two simultaneous uploads
(or an upload racing a delete) can lose a manifest entry.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.embeddings.ollama import OllamaEmbedding

from app.index.store import VectorStoreHandle


@dataclass(frozen=True)
class AppState:
    handle: VectorStoreHandle
    embed_model: OllamaEmbedding
    index: VectorStoreIndex
    chat_engine: BaseChatEngine
    chat_lock: asyncio.Lock
    library_lock: asyncio.Lock
