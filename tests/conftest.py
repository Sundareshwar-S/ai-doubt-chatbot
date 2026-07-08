"""Shared pytest fixtures."""
import asyncio
import urllib.request

import httpx
import pytest

import config
from app.api.dependencies import get_state
from app.api.main import create_app
from app.api.state import AppState
from app.index.pipeline import default_embed_model, load_index
from app.index.store import get_vector_store_handle
from app.qa.chat import build_chat_engine


@pytest.fixture(scope="session")
def require_ollama():
    """Skip the test if the local Ollama server isn't reachable."""
    try:
        urllib.request.urlopen(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=2.0)
    except Exception:
        pytest.skip("Ollama not reachable at OLLAMA_BASE_URL; start it to run this test")


@pytest.fixture
def app_state(tmp_path):
    """A fully-built `AppState` against an isolated tmp_path Chroma collection.

    Building `index`/`chat_engine` makes real Ollama calls (embeddings, LLM
    construction) -- tests using this fixture should also depend on
    `require_ollama`.
    """
    handle = get_vector_store_handle(persist_dir=tmp_path / "chroma", collection_name="test")
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


@pytest.fixture
def api_app(app_state, tmp_path, monkeypatch):
    """A `create_app()` instance with `get_state` overridden to the isolated
    `app_state`, and `config.MANIFEST_PATH`/`config.UPLOADS_DIR` pointed at
    tmp_path so document routes never touch the real data/ directory.

    `httpx.ASGITransport` never sends the ASGI lifespan protocol, so
    `create_app()`'s own `lifespan` (which would build a *second*, real
    `AppState` against the real Ollama/Chroma) simply never runs here.
    """
    monkeypatch.setattr(config, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(config, "UPLOADS_DIR", tmp_path / "uploads")

    app = create_app()
    app.dependency_overrides[get_state] = lambda: app_state
    yield app
    app.dependency_overrides.clear()


@pytest.fixture
async def client(api_app):
    """An async test client bound to `api_app` via `httpx.ASGITransport`."""
    transport = httpx.ASGITransport(app=api_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
