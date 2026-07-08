"""Chroma-backed vector store + storage context (persistent, on-disk)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

import config


@dataclass(frozen=True)
class VectorStoreHandle:
    client: chromadb.ClientAPI
    collection: Collection
    vector_store: ChromaVectorStore
    storage_context: StorageContext


def get_vector_store_handle(
    persist_dir: str | Path | None = None,
    collection_name: str | None = None,
) -> VectorStoreHandle:
    """Open (creating if needed) the persistent Chroma collection + storage context.

    persist_dir/collection_name default to config but are overridable so tests can
    point at an isolated tmp_path collection instead of the real data/chroma/.
    """
    persist_dir = str(persist_dir or config.PERSIST_DIR)
    collection_name = collection_name or config.COLLECTION_NAME
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreHandle(client, collection, vector_store, storage_context)
