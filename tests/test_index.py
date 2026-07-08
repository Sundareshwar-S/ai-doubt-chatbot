"""Tests for app/index/store.py, app/index/manifest.py, app/index/pipeline.py (T2.2-T2.7)."""
import subprocess
import sys
from pathlib import Path

import pytest
from llama_index.core import Document

import config
from app.index import manifest as manifest_store
from app.index import pipeline
from app.index.store import get_vector_store_handle

FIXTURES = Path(__file__).parent / "fixtures"


# --- store.py -----------------------------------------------------------

def test_get_vector_store_handle_creates_persist_dir(tmp_path):
    # Arrange
    persist_dir = tmp_path / "chroma"

    # Act
    handle = get_vector_store_handle(persist_dir=persist_dir, collection_name="test")

    # Assert
    assert persist_dir.exists()
    assert handle.collection.count() == 0


# --- manifest.py (pure Python, no Ollama needed) -------------------------

def test_manifest_round_trips_entries(tmp_path):
    # Arrange
    manifest_path = tmp_path / "manifest.json"
    entry = manifest_store.ManifestEntry(
        source="a.pdf", sha256="abc123", pages=2, added_at=manifest_store.now_iso()
    )

    # Act
    manifest_store.add_entry(entry, manifest_path)

    # Assert
    assert manifest_store.has_hash("abc123", manifest_path)
    assert manifest_store.load_entries(manifest_path) == [entry]


def test_manifest_add_entry_replaces_same_source(tmp_path):
    # Arrange
    manifest_path = tmp_path / "manifest.json"
    first = manifest_store.ManifestEntry(source="a.pdf", sha256="hash1", pages=1, added_at="t1")
    second = manifest_store.ManifestEntry(source="a.pdf", sha256="hash2", pages=3, added_at="t2")

    # Act
    manifest_store.add_entry(first, manifest_path)
    manifest_store.add_entry(second, manifest_path)

    # Assert
    entries = manifest_store.load_entries(manifest_path)
    assert len(entries) == 1
    assert entries[0].sha256 == "hash2"


def test_manifest_remove_entry(tmp_path):
    # Arrange
    manifest_path = tmp_path / "manifest.json"
    entry = manifest_store.ManifestEntry(source="a.pdf", sha256="abc", pages=1, added_at="t")
    manifest_store.add_entry(entry, manifest_path)

    # Act
    manifest_store.remove_entry("a.pdf", manifest_path)

    # Assert
    assert manifest_store.load_entries(manifest_path) == []


def test_sha256_file_is_deterministic(tmp_path):
    # Arrange
    path = tmp_path / "sample.txt"
    path.write_text("hello world")

    # Act
    hash1 = manifest_store.sha256_file(path)
    hash2 = manifest_store.sha256_file(path)

    # Assert
    assert hash1 == hash2
    assert len(hash1) == 64


# --- pipeline.py (integration: real local Ollama + Chroma) ---------------

@pytest.mark.integration
def test_add_documents_and_retrieve_top_k_with_metadata(tmp_path, require_ollama):
    # Arrange
    handle = get_vector_store_handle(persist_dir=tmp_path / "chroma", collection_name="test")
    docs = [
        Document(text="Mitochondria are the powerhouse of the cell.",
                 metadata={"source": "bio.pdf", "page": 1}),
        Document(text="The mitochondrion generates ATP through respiration.",
                 metadata={"source": "bio.pdf", "page": 2}),
        Document(text="Photosynthesis converts light into chemical energy in plants.",
                 metadata={"source": "botany.pdf", "page": 1}),
    ]

    # Act
    pipeline.add_documents(docs, handle=handle)
    index = pipeline.load_index(handle)
    nodes = index.as_retriever(similarity_top_k=config.TOP_K).retrieve(
        "What is the powerhouse of the cell?"
    )

    # Assert
    assert len(nodes) > 0
    assert all("source" in n.metadata and "page" in n.metadata for n in nodes)
    assert nodes[0].metadata["source"] == "bio.pdf"


@pytest.mark.integration
def test_ingest_file_end_to_end_with_real_fixture(tmp_path, require_ollama):
    # Arrange
    handle = get_vector_store_handle(persist_dir=tmp_path / "chroma", collection_name="test")
    manifest_path = tmp_path / "manifest.json"
    fixture = FIXTURES / "born_digital.pdf"

    # Act
    ingested = pipeline.ingest_file(fixture, handle=handle, manifest_path=manifest_path)

    # Assert
    assert ingested is True
    assert handle.collection.count() > 0
    entries = pipeline.list_documents(manifest_path)
    assert len(entries) == 1
    assert entries[0].source == "born_digital.pdf"
    assert entries[0].pages == 2


@pytest.mark.integration
def test_ingest_file_skips_reingest_of_same_content(tmp_path, require_ollama):
    # Arrange
    handle = get_vector_store_handle(persist_dir=tmp_path / "chroma", collection_name="test")
    manifest_path = tmp_path / "manifest.json"
    fixture = FIXTURES / "born_digital.pdf"
    pipeline.ingest_file(fixture, handle=handle, manifest_path=manifest_path)
    count_after_first = handle.collection.count()

    # Act
    ingested_again = pipeline.ingest_file(fixture, handle=handle, manifest_path=manifest_path)

    # Assert
    assert ingested_again is False
    assert handle.collection.count() == count_after_first
    assert len(pipeline.list_documents(manifest_path)) == 1


@pytest.mark.integration
def test_ingest_file_replaces_stale_vectors_on_same_filename_new_content(tmp_path, require_ollama):
    """Re-uploading a *changed* file under the same name must not leave the old
    version's vectors alongside the new ones under the same `source`."""
    # Arrange
    handle = get_vector_store_handle(persist_dir=tmp_path / "chroma", collection_name="test")
    manifest_path = tmp_path / "manifest.json"
    original = tmp_path / "notes.pdf"
    original.write_bytes((FIXTURES / "born_digital.pdf").read_bytes())  # mentions mitochondria
    pipeline.ingest_file(original, handle=handle, manifest_path=manifest_path)

    changed = tmp_path / "notes.pdf"  # same filename
    changed.write_bytes((FIXTURES / "scanned.pdf").read_bytes())  # different content, "second law"

    # Act
    ingested_again = pipeline.ingest_file(changed, handle=handle, manifest_path=manifest_path)

    # Assert
    assert ingested_again is True
    entries = pipeline.list_documents(manifest_path)
    assert len(entries) == 1

    index = pipeline.load_index(handle)
    new_hits = index.as_retriever(similarity_top_k=5).retrieve("second law of thermodynamics")
    assert any(n.metadata["source"] == "notes.pdf" for n in new_hits)

    stale_hits = index.as_retriever(similarity_top_k=5).retrieve("mitochondria powerhouse of the cell")
    assert all(n.score < 0.5 for n in stale_hits), "old version's vectors were not removed"


@pytest.mark.integration
def test_list_and_remove_document(tmp_path, require_ollama):
    # Arrange
    handle = get_vector_store_handle(persist_dir=tmp_path / "chroma", collection_name="test")
    manifest_path = tmp_path / "manifest.json"
    doc_a = [Document(text="Content about apples.", metadata={"source": "a.pdf", "page": 1})]
    doc_b = [Document(text="Content about bananas.", metadata={"source": "b.pdf", "page": 1})]
    pipeline.add_documents(doc_a, handle=handle)
    manifest_store.add_entry(
        manifest_store.ManifestEntry("a.pdf", "hashA", 1, manifest_store.now_iso()), manifest_path
    )
    pipeline.add_documents(doc_b, handle=handle)
    manifest_store.add_entry(
        manifest_store.ManifestEntry("b.pdf", "hashB", 1, manifest_store.now_iso()), manifest_path
    )

    # Act
    listed_before = pipeline.list_documents(manifest_path)
    pipeline.remove_document("a.pdf", handle=handle, manifest_path=manifest_path)
    listed_after = pipeline.list_documents(manifest_path)

    # Assert
    assert {e.source for e in listed_before} == {"a.pdf", "b.pdf"}
    assert {e.source for e in listed_after} == {"b.pdf"}
    remaining_nodes = pipeline.load_index(handle).as_retriever(
        similarity_top_k=10
    ).retrieve("apples bananas")
    assert all(n.metadata["source"] != "a.pdf" for n in remaining_nodes)


@pytest.mark.integration
def test_reload_after_fresh_process_returns_relevant_chunk(tmp_path, require_ollama):
    """T2.4 (B5): a document ingested in this process must be retrievable from a
    brand-new OS process pointed at the same persist_dir -- proves persistence,
    not just in-memory state."""
    # Arrange: this process ingests
    persist_dir = tmp_path / "chroma"
    collection = "test"
    handle = get_vector_store_handle(persist_dir=persist_dir, collection_name=collection)
    pipeline.add_documents(
        [Document(text="Mitochondria are the powerhouse of the cell.",
                  metadata={"source": "bio.pdf", "page": 1})],
        handle=handle,
    )

    # Act: a fresh OS process loads the same persist_dir and queries it
    script = (
        f"import sys; sys.path.insert(0, {str(config.PROJECT_ROOT)!r})\n"
        "from app.index.store import get_vector_store_handle\n"
        "from app.index.pipeline import load_index\n"
        f"h = get_vector_store_handle({str(persist_dir)!r}, {collection!r})\n"
        "idx = load_index(h)\n"
        "n = idx.as_retriever(similarity_top_k=1).retrieve('What is the powerhouse of the cell?')\n"
        "print(n[0].metadata['source'] if n else 'NO_RESULTS')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )

    # Assert
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "bio.pdf"
