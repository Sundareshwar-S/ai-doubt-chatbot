"""Chunk -> embed -> index; ingest/add/list/remove backed by data/manifest.json.

Chroma persists the vectors directly, so there is no separate "build" vs.
"insert" path to maintain: loading always reloads from the persisted
collection (`VectorStoreIndex.from_vector_store`), and adding documents
always goes through `insert_nodes` on that freshly loaded index. This is
also what makes reload-after-restart correct -- nothing here trusts
in-memory index state.
"""
from __future__ import annotations

from pathlib import Path

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding

import config
from app.index import manifest as manifest_store
from app.index.store import VectorStoreHandle, get_vector_store_handle
from app.ingest.extract import extract_documents


def default_embed_model() -> OllamaEmbedding:
    return OllamaEmbedding(model_name=config.EMBED_MODEL, base_url=config.OLLAMA_BASE_URL)


def load_index(
    handle: VectorStoreHandle | None = None,
    embed_model: OllamaEmbedding | None = None,
) -> VectorStoreIndex:
    """Reload an index bound to an existing (possibly empty) collection.

    Safe to call from a brand-new process (T2.4) -- reads only from the
    Chroma-persisted vectors, never from in-memory state.
    """
    handle = handle or get_vector_store_handle()
    return VectorStoreIndex.from_vector_store(
        handle.vector_store,
        storage_context=handle.storage_context,
        embed_model=embed_model or default_embed_model(),
    )


def add_documents(
    documents: list[Document],
    handle: VectorStoreHandle | None = None,
    embed_model: OllamaEmbedding | None = None,
) -> VectorStoreIndex:
    """Chunk (SentenceSplitter) + embed + insert Documents.

    Covers both the very first document and later additions.
    """
    handle = handle or get_vector_store_handle()
    embed_model = embed_model or default_embed_model()
    splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)
    index = load_index(handle, embed_model=embed_model)
    index.insert_nodes(nodes)
    return index


def ingest_file(
    path: str | Path,
    handle: VectorStoreHandle | None = None,
    embed_model: OllamaEmbedding | None = None,
    manifest_path: Path | None = None,
) -> bool:
    """Extract -> add_documents -> manifest write.

    Returns False (no-op) if this file's content sha256 is already in the
    manifest (T2.5); True if it was ingested. If a *different* file was
    previously ingested under the same filename, its old vectors are
    removed first so the two versions don't coexist under one `source`.

    Extraction runs before any destructive step, so a corrupt/unsupported
    file (ExtractionError) leaves both the manifest and the vector store
    untouched -- the stale-source delete only happens once the new
    content has already been successfully extracted.
    """
    path = Path(path)
    file_hash = manifest_store.sha256_file(path)
    entries = manifest_store.load_entries(manifest_path)
    if any(e.sha256 == file_hash for e in entries):
        return False

    documents = extract_documents(path)

    handle = handle or get_vector_store_handle()
    if manifest_store.find_by_source(entries, path.name) is not None:
        handle.collection.delete(where={"source": path.name})

    add_documents(documents, handle=handle, embed_model=embed_model)

    manifest_store.add_entry(
        manifest_store.ManifestEntry(
            source=path.name,
            sha256=file_hash,
            pages=len(documents),
            added_at=manifest_store.now_iso(),
        ),
        manifest_path,
    )
    return True


def list_documents(manifest_path: Path | None = None) -> list[manifest_store.ManifestEntry]:
    return manifest_store.load_entries(manifest_path)


def remove_document(
    source: str,
    handle: VectorStoreHandle | None = None,
    manifest_path: Path | None = None,
) -> None:
    """Delete a document's vectors (by {source} metadata filter) and its manifest entry.

    Uses the Chroma collection's delete(where=...) directly rather than
    ChromaVectorStore.delete_nodes(filters=...): the installed
    llama-index-vector-stores-chroma (0.5.5) passes ids=[] alongside the
    filter, which chromadb 1.5.9 now rejects ("Expected IDs to be a
    non-empty list"). `where={"source": source}` is a strict equality
    match on the metadata key, so `source` can't be used to inject
    filter operators.
    """
    handle = handle or get_vector_store_handle()
    handle.collection.delete(where={"source": source})
    manifest_store.remove_entry(source, manifest_path)
