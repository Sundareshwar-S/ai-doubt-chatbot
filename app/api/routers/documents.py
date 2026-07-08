"""`/api/documents` -- upload, list, remove (Phase 5, T5.3/T5.4).

Upload and remove both go through `state.library_lock`: they mutate the same
shared `data/manifest.json` (read-all/mutate-list/write-all, no other
concurrency guard) and the shared Chroma collection, so two concurrent
requests (two tabs uploading at once, or an upload racing a delete) could
otherwise lose a manifest entry -- the same class of race `chat_lock` guards
against for the chat engine.
"""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, File, Request, Response, UploadFile, status
from starlette.concurrency import run_in_threadpool

import config
from app.api.dependencies import get_state
from app.api.exceptions import DocumentNotFoundError, UploadTooLargeError
from app.api.schemas.documents import DocumentEntry, DocumentListResponse, DocumentUploadResponse
from app.api.services import ingestion_service
from app.api.state import AppState
from app.index import manifest as manifest_store
from app.index.pipeline import list_documents, remove_document

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("", response_model=DocumentUploadResponse)
async def upload_document(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    state: AppState = Depends(get_state),
) -> DocumentUploadResponse:
    # Cheap pre-check on the request's Content-Length so an obviously-oversized
    # upload is rejected before its body is read into memory. This is a proxy
    # (it covers the whole multipart body, not just this field, and is absent
    # for chunked transfer encoding) -- ingestion_service.ingest()'s exact
    # per-file check after read() remains the source of truth.
    content_length = request.headers.get("content-length")
    if content_length is not None and int(content_length) > config.MAX_UPLOAD_SIZE_BYTES:
        limit_mb = config.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        raise UploadTooLargeError(f"File exceeds the {limit_mb} MB limit.")

    content = await file.read()
    async with state.library_lock:
        result = await run_in_threadpool(
            ingestion_service.ingest, file.filename, content, state.handle, state.embed_model
        )
    response.status_code = status.HTTP_201_CREATED if result.ingested else status.HTTP_200_OK
    return DocumentUploadResponse(**asdict(result.entry), ingested=result.ingested)


@router.get("", response_model=DocumentListResponse)
async def get_documents() -> DocumentListResponse:
    entries = await run_in_threadpool(list_documents)
    return DocumentListResponse(documents=[DocumentEntry(**asdict(e)) for e in entries])


@router.delete("/{source}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(source: str, state: AppState = Depends(get_state)) -> None:
    async with state.library_lock:
        entries = await run_in_threadpool(list_documents)
        if manifest_store.find_by_source(entries, source) is None:
            raise DocumentNotFoundError(f"No document named '{source}' in the library.")
        await run_in_threadpool(remove_document, source, state.handle)
