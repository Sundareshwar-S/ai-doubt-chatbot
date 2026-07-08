"""Request/response schemas for `/api/documents` (Phase 5)."""
from __future__ import annotations

from pydantic import BaseModel


class DocumentEntry(BaseModel):
    source: str
    sha256: str
    pages: int
    added_at: str


class DocumentUploadResponse(DocumentEntry):
    ingested: bool


class DocumentListResponse(BaseModel):
    documents: list[DocumentEntry]
