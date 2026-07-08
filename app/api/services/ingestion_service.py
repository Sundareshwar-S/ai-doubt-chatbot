"""Upload validation + ingestion orchestration for `POST /api/documents` (Phase 5, T5.3/T6.2).

Entirely synchronous/blocking (file save, then `pipeline.ingest_file`'s PyMuPDF/
RapidOCR/Ollama-embedding calls) -- the router awaits this via
`starlette.concurrency.run_in_threadpool`, never calling it directly from an
`async def` route, per this project's FastAPI rule against blocking calls in
async routes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llama_index.embeddings.ollama import OllamaEmbedding

import config
from app.api.exceptions import UnsupportedFileTypeError, UploadTooLargeError
from app.index import manifest as manifest_store
from app.index.pipeline import ingest_file
from app.index.store import VectorStoreHandle
from app.ingest.extract import IMAGE_SUFFIXES, PDF_SUFFIXES

_ALLOWED_SUFFIXES = PDF_SUFFIXES | IMAGE_SUFFIXES


@dataclass(frozen=True)
class IngestResult:
    entry: manifest_store.ManifestEntry
    ingested: bool


def _safe_filename(filename: str | None) -> str:
    """Basename-only (strips any directory component, including `../`), and
    rejects an empty/`.`/`..` name -- T6.2's filename-sanitization guard.
    """
    name = Path(filename or "").name
    if not name or name in {".", ".."}:
        raise UnsupportedFileTypeError("Missing or invalid filename.")
    return name


def ingest(
    filename: str | None,
    content: bytes,
    handle: VectorStoreHandle,
    embed_model: OllamaEmbedding,
) -> IngestResult:
    """Validate suffix/size, sanitize the filename, save, and ingest.

    Suffix and size are checked *before* anything is written to disk, so a
    rejected upload leaves no partial file behind (T6.2).
    """
    name = _safe_filename(filename)
    suffix = Path(name).suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise UnsupportedFileTypeError(f"Unsupported file type: {suffix or '(none)'}")

    if len(content) > config.MAX_UPLOAD_SIZE_BYTES:
        limit_mb = config.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        raise UploadTooLargeError(f"File exceeds the {limit_mb} MB limit.")

    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.UPLOADS_DIR / name
    dest.write_bytes(content)

    was_ingested = ingest_file(dest, handle=handle, embed_model=embed_model)
    entries = manifest_store.load_entries()
    entry = manifest_store.find_by_source(entries, name)
    if entry is None:
        # ingest_file() always adds/updates an entry for `name` -- reaching this
        # means the manifest write silently failed or lost a race.
        raise RuntimeError(f"Ingested '{name}' but no manifest entry was found for it.")
    return IngestResult(entry=entry, ingested=was_ingested)


__all__ = ["IngestResult", "ingest"]
