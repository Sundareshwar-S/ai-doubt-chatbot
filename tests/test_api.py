"""Tests for the FastAPI backend (Phase 5).

Each test builds its own minimal FastAPI app rather than depending on router
files that later tasks add, except where noted -- this file grows as
app/api/routers/* fill in.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llama_index.core import Document

import config
from app.api.exceptions import (
    ApiError,
    DocumentNotFoundError,
    OllamaUnavailableError,
    UnsupportedFileTypeError,
    UploadTooLargeError,
    register_exception_handlers,
)
from app.index.pipeline import add_documents
from app.ingest.extract import ExtractionError

_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_BORN_DIGITAL_PDF = _FIXTURES_DIR / "born_digital.pdf"

_BIO_DOCS = [
    Document(
        text="Mitochondria are the powerhouse of the cell.",
        metadata={"source": "bio.pdf", "page": 1},
    ),
    Document(
        text="The mitochondrion generates ATP through cellular respiration.",
        metadata={"source": "bio.pdf", "page": 2},
    ),
]


def _error_test_app() -> FastAPI:
    """A throwaway app with only the exception handlers wired up, plus routes
    that deliberately raise each domain error -- isolates T5.2 from routers
    that don't exist yet.
    """
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/raise/upload-too-large")
    def _raise_upload_too_large():
        raise UploadTooLargeError("file exceeds the configured size limit")

    @app.get("/raise/unsupported-file-type")
    def _raise_unsupported_file_type():
        raise UnsupportedFileTypeError("unsupported file suffix")

    @app.get("/raise/document-not-found")
    def _raise_document_not_found():
        raise DocumentNotFoundError("no such document in the manifest")

    @app.get("/raise/ollama-unavailable")
    def _raise_ollama_unavailable():
        raise OllamaUnavailableError("Ollama is not reachable")

    @app.get("/raise/extraction-error")
    def _raise_extraction_error():
        raise ExtractionError("could not parse this file")

    return app


@pytest.mark.parametrize(
    ("path", "expected_status", "expected_code"),
    [
        ("/raise/upload-too-large", 413, "upload_too_large"),
        ("/raise/unsupported-file-type", 400, "unsupported_file_type"),
        ("/raise/document-not-found", 404, "document_not_found"),
        ("/raise/ollama-unavailable", 503, "ollama_unavailable"),
        ("/raise/extraction-error", 422, "extraction_failed"),
    ],
)
def test_exception_handlers_render_error_envelope(path, expected_status, expected_code):
    """T5.2 -- each domain exception (and ExtractionError) renders as
    `{"error": {"code", "message"}}` with the right HTTP status.
    """
    # Arrange
    app = _error_test_app()
    client = TestClient(app, raise_server_exceptions=False)

    # Act
    response = client.get(path)

    # Assert
    assert response.status_code == expected_status
    body = response.json()
    assert body["error"]["code"] == expected_code
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]


def test_api_error_base_class_defaults():
    """ApiError itself defaults to 400/bad_request when a subclass doesn't override."""
    exc = ApiError("something went wrong")
    assert exc.status_code == 400
    assert exc.code == "bad_request"
    assert exc.message == "something went wrong"


@pytest.mark.integration
async def test_health_ok_when_ollama_reachable(client, require_ollama):
    """T5.7 -- with Ollama running and the configured models pulled, 200 "ok"."""
    # Act
    response = await client.get("/health")

    # Assert
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_health_degraded_when_ollama_unreachable(client, monkeypatch):
    """T5.7 -- pointing OLLAMA_BASE_URL at a dead port -> 503 "degraded" with a
    readable detail, deterministic (no live Ollama needed).
    """
    # Arrange
    import config

    monkeypatch.setattr(config, "OLLAMA_BASE_URL", "http://localhost:1")

    # Act
    response = await client.get("/health")

    # Assert
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert "Ollama" in body["detail"]


@pytest.mark.integration
async def test_health_degraded_with_distinct_detail_when_model_not_pulled(
    client, require_ollama, monkeypatch
):
    """T6.3 -- Ollama reachable but a configured model isn't pulled -> 503
    "degraded" with a message distinct from the "can't reach Ollama" case
    (asserts on a model tag that's real-Ollama-reachable but never pulled).
    """
    # Arrange
    monkeypatch.setattr(config, "LLM_MODEL", "definitely-not-a-pulled-model:1b")

    # Act
    response = await client.get("/health")

    # Assert
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert "not pulled" in body["detail"]
    assert "ollama pull" in body["detail"]
    assert "Cannot reach Ollama" not in body["detail"]


@pytest.mark.integration
async def test_upload_pdf_ingests_and_returns_201(client, require_ollama):
    """T5.3 -- a new PDF upload is ingested and returns 201 with ingested: true."""
    # Act
    with _BORN_DIGITAL_PDF.open("rb") as f:
        response = await client.post(
            "/api/documents", files={"file": ("born_digital.pdf", f, "application/pdf")}
        )

    # Assert
    assert response.status_code == 201
    body = response.json()
    assert body["source"] == "born_digital.pdf"
    assert body["ingested"] is True
    assert body["pages"] >= 1


@pytest.mark.integration
async def test_reupload_same_bytes_returns_200_ingested_false(client, require_ollama):
    """T5.3/T2.5 -- re-uploading identical bytes is a dedup no-op: 200, ingested: false."""
    # Arrange
    with _BORN_DIGITAL_PDF.open("rb") as f:
        first = await client.post(
            "/api/documents", files={"file": ("born_digital.pdf", f, "application/pdf")}
        )
    assert first.status_code == 201

    # Act
    with _BORN_DIGITAL_PDF.open("rb") as f:
        second = await client.post(
            "/api/documents", files={"file": ("born_digital.pdf", f, "application/pdf")}
        )

    # Assert
    assert second.status_code == 200
    assert second.json()["ingested"] is False


async def test_upload_oversized_file_returns_413_with_no_partial_file(
    client, monkeypatch, tmp_path
):
    """T6.2 -- a file over MAX_UPLOAD_SIZE_BYTES is rejected with 413, and no
    partial file is left in UPLOADS_DIR. Deterministic: no Ollama call happens
    (size is checked before ingestion).
    """
    # Arrange: shrink the limit so a tiny payload already exceeds it.
    monkeypatch.setattr(config, "MAX_UPLOAD_SIZE_BYTES", 10)

    # Act
    response = await client.post(
        "/api/documents",
        files={"file": ("too_big.pdf", b"%PDF-1.4 way more than 10 bytes", "application/pdf")},
    )

    # Assert
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "upload_too_large"
    assert not (tmp_path / "uploads" / "too_big.pdf").exists()


async def test_upload_unsupported_suffix_returns_400(client):
    """T6.2 -- an unsupported file suffix is rejected with 400, deterministic."""
    # Act
    response = await client.post(
        "/api/documents", files={"file": ("notes.txt", b"plain text", "text/plain")}
    )

    # Assert
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_file_type"


@pytest.mark.integration
async def test_upload_path_traversal_filename_is_sanitized(client, tmp_path, require_ollama):
    """T6.2 -- a filename containing `../` is stripped to its basename before
    it's ever written to disk or recorded in the manifest.
    """
    # Act
    with _BORN_DIGITAL_PDF.open("rb") as f:
        response = await client.post(
            "/api/documents",
            files={"file": ("../../etc/passwd.pdf", f, "application/pdf")},
        )

    # Assert
    assert response.status_code == 201
    assert response.json()["source"] == "passwd.pdf"
    assert (tmp_path / "uploads" / "passwd.pdf").exists()
    assert not (tmp_path / "etc").exists()


@pytest.mark.integration
async def test_list_documents_shows_uploaded_files(client, require_ollama):
    """T5.4 -- GET /api/documents lists what's been uploaded."""
    # Arrange
    with _BORN_DIGITAL_PDF.open("rb") as f:
        await client.post(
            "/api/documents", files={"file": ("born_digital.pdf", f, "application/pdf")}
        )

    # Act
    response = await client.get("/api/documents")

    # Assert
    assert response.status_code == 200
    sources = [d["source"] for d in response.json()["documents"]]
    assert sources == ["born_digital.pdf"]


@pytest.mark.integration
async def test_delete_document_removes_it_and_404s_on_repeat(client, require_ollama):
    """T5.4 -- deleting a document removes it from the list; deleting again 404s,
    since remove_document() is otherwise a silent no-op on a missing source.
    """
    # Arrange
    with _BORN_DIGITAL_PDF.open("rb") as f:
        await client.post(
            "/api/documents", files={"file": ("born_digital.pdf", f, "application/pdf")}
        )

    # Act
    first_delete = await client.delete("/api/documents/born_digital.pdf")
    second_delete = await client.delete("/api/documents/born_digital.pdf")

    # Assert
    assert first_delete.status_code == 204
    assert second_delete.status_code == 404
    assert second_delete.json()["error"]["code"] == "document_not_found"

    listing = await client.get("/api/documents")
    assert listing.json()["documents"] == []


@pytest.mark.integration
async def test_chat_ask_returns_cited_answer_for_indoc_question(client, app_state, require_ollama):
    """T5.5 -- an in-doc question returns an answer with >=1 citation."""
    # Arrange
    add_documents(_BIO_DOCS, handle=app_state.handle, embed_model=app_state.embed_model)

    # Act
    response = await client.post(
        "/api/chat/ask", json={"question": "What is the powerhouse of the cell?"}
    )

    # Assert
    assert response.status_code == 200
    body = response.json()
    assert len(body["answer"]) > 0
    assert len(body["citations"]) >= 1
    assert body["citations"][0]["source"] == "bio.pdf"


@pytest.mark.integration
async def test_chat_ask_refuses_out_of_doc_question_with_200(client, app_state, require_ollama):
    """T5.5/C3 -- an out-of-doc question is a normal 200 refusal, not an error."""
    # Arrange
    add_documents(_BIO_DOCS, handle=app_state.handle, embed_model=app_state.embed_model)

    # Act
    response = await client.post(
        "/api/chat/ask",
        json={"question": "What was the exchange rate of the Ottoman lira in 1875?"},
    )

    # Assert
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "I can't find this in your documents."
    assert body["citations"] == []


async def _collect_stream(client, question):
    """Act helper: POST /api/chat/stream and return the parsed NDJSON events."""
    async with client.stream("POST", "/api/chat/stream", json={"question": question}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        return [json.loads(line) async for line in response.aiter_lines() if line.strip()]


@pytest.mark.integration
async def test_chat_stream_returns_tokens_then_done_for_indoc_question(
    client, app_state, require_ollama
):
    """Streaming: an in-doc question emits `token` events whose concatenation is a
    non-empty answer, then a terminal `done` event carrying citations.
    """
    # Arrange
    add_documents(_BIO_DOCS, handle=app_state.handle, embed_model=app_state.embed_model)

    # Act
    events = await _collect_stream(client, "What is the powerhouse of the cell?")

    # Assert
    answer = "".join(e["text"] for e in events if e["type"] == "token")
    assert len(answer) > 0
    assert events[-1]["type"] == "done"
    citations = events[-1]["citations"]
    assert len(citations) >= 1
    assert citations[0]["source"] == "bio.pdf"


@pytest.mark.integration
async def test_chat_stream_refuses_out_of_doc_question(client, app_state, require_ollama):
    """Streaming refusal: an out-of-doc question streams exactly the fixed refusal
    text and a `done` event with no citations (mirrors the /ask refusal contract).
    """
    # Arrange
    add_documents(_BIO_DOCS, handle=app_state.handle, embed_model=app_state.embed_model)

    # Act
    events = await _collect_stream(
        client, "What was the exchange rate of the Ottoman lira in 1875?"
    )

    # Assert
    answer = "".join(e["text"] for e in events if e["type"] == "token")
    assert answer == "I can't find this in your documents."
    assert events[-1] == {"type": "done", "citations": []}


@pytest.mark.integration
async def test_chat_reset_clears_conversation(client, app_state, require_ollama):
    """T5.6 -- after reset, a follow-up no longer resolves against the prior turn."""
    # Arrange
    add_documents(_BIO_DOCS, handle=app_state.handle, embed_model=app_state.embed_model)
    await client.post("/api/chat/ask", json={"question": "What is the powerhouse of the cell?"})
    assert len(app_state.chat_engine.chat_history) > 0

    # Act
    reset_response = await client.post("/api/chat/reset")

    # Assert
    assert reset_response.status_code == 204
    assert app_state.chat_engine.chat_history == []
