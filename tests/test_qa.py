"""Tests for app/qa/engine.py, app/qa/prompts.py (T3.1-T3.3)."""
import pytest
from llama_index.core import Document

from app.index.pipeline import add_documents, load_index
from app.index.store import get_vector_store_handle
from app.qa import engine
from app.qa.prompts import REFUSAL_MESSAGE


@pytest.mark.integration
def test_in_doc_question_returns_cited_answer(tmp_path, require_ollama):
    # Arrange
    handle = get_vector_store_handle(persist_dir=tmp_path / "chroma", collection_name="test")
    docs = [
        Document(text="Mitochondria are the powerhouse of the cell.",
                 metadata={"source": "bio.pdf", "page": 1}),
        Document(text="The mitochondrion generates ATP through cellular respiration.",
                 metadata={"source": "bio.pdf", "page": 2}),
    ]
    add_documents(docs, handle=handle)
    index = load_index(handle)

    # Act
    result = engine.answer_question("What is the powerhouse of the cell?", index=index)

    # Assert
    assert result.answer != REFUSAL_MESSAGE
    assert len(result.answer) > 0
    assert len(result.citations) >= 1
    assert all(c.source == "bio.pdf" for c in result.citations)
    assert all(c.page in (1, 2) for c in result.citations)


@pytest.mark.integration
def test_out_of_doc_question_refuses(tmp_path, require_ollama):
    # Arrange
    handle = get_vector_store_handle(persist_dir=tmp_path / "chroma", collection_name="test")
    docs = [
        Document(text="Mitochondria are the powerhouse of the cell.",
                 metadata={"source": "bio.pdf", "page": 1}),
    ]
    add_documents(docs, handle=handle)
    index = load_index(handle)

    # Act
    result = engine.answer_question(
        "What was the exchange rate of the Ottoman lira in 1875?", index=index
    )

    # Assert: exact-string check -- REFUSAL_MESSAGE is a fixed constant
    # returned by our own short-circuit, never LLM-generated text, so this
    # cannot be flaky regardless of what the local model would say.
    assert result.answer == REFUSAL_MESSAGE
    assert result.citations == []


@pytest.mark.integration
def test_citations_deduplicate_by_source_and_page(tmp_path, require_ollama):
    # Arrange: two chunks that both land on the same (source, page) --
    # simulates what happens when one page splits into >1 node.
    handle = get_vector_store_handle(persist_dir=tmp_path / "chroma", collection_name="test")
    docs = [
        Document(text="Mitochondria are the powerhouse of the cell.",
                 metadata={"source": "bio.pdf", "page": 1}),
        Document(text="Mitochondria generate ATP via oxidative phosphorylation.",
                 metadata={"source": "bio.pdf", "page": 1}),
    ]
    add_documents(docs, handle=handle)
    index = load_index(handle)

    # Act
    result = engine.answer_question("What is the powerhouse of the cell?", index=index)

    # Assert: both chunks are on page 1 of bio.pdf -> exactly one citation for it
    assert len([c for c in result.citations if (c.source, c.page) == ("bio.pdf", 1)]) == 1
