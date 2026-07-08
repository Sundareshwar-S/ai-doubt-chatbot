"""Tests for app/qa/chat.py -- multi-turn chat memory (T4.1-T4.3).

Mirrors tests/test_qa.py: integration tests hit a live local Ollama (auto-skipped
via require_ollama), each with an isolated tmp_path Chroma handle.

Where the small (3B) model's *answer wording* is unreliable, assertions target
robust signals instead: the follow-up test asserts on *retrieval* (which document
the condensed query pulled), and the reset/token-cap tests are fully deterministic.
"""
import pytest
from llama_index.core import Document
from llama_index.core.llms import MessageRole

import config
from app.index.pipeline import add_documents, load_index
from app.index.store import get_vector_store_handle
from app.qa import chat
from app.qa.chat import default_memory
from app.qa.prompts import REFUSAL_MESSAGE

_BIO_DOCS = [
    Document(text="Mitochondria are the powerhouse of the cell.",
             metadata={"source": "bio.pdf", "page": 1}),
    Document(text="The mitochondrion generates ATP through cellular respiration.",
             metadata={"source": "bio.pdf", "page": 2}),
]

# Extra off-topic docs so the follow-up test's retrieval has to *discriminate*:
# a topic-neutral pronoun follow-up can only rank bio.pdf top if the condense
# step pulled "mitochondria" out of the prior turn.
_MULTI_TOPIC_DOCS = _BIO_DOCS + [
    Document(text="The French Revolution began in 1789 and ended the monarchy in France.",
             metadata={"source": "history.pdf", "page": 1}),
    Document(text="Photosynthesis converts sunlight into chemical energy in chloroplasts.",
             metadata={"source": "botany.pdf", "page": 1}),
]


def _index_with(docs, tmp_path):
    """Arrange helper: fresh isolated Chroma index seeded with `docs`."""
    handle = get_vector_store_handle(persist_dir=tmp_path / "chroma", collection_name="test")
    add_documents(docs, handle=handle)
    return load_index(handle)


@pytest.mark.integration
def test_chat_returns_response_with_source_nodes(tmp_path, require_ollama):
    """T4.1 -- a single in-doc question returns a cited, non-refusal answer."""
    # Arrange
    index = _index_with(_BIO_DOCS, tmp_path)
    chat_engine = chat.build_chat_engine(index=index)

    # Act
    result = chat.ask(chat_engine, "What is the powerhouse of the cell?")

    # Assert
    assert result.answer != REFUSAL_MESSAGE
    assert len(result.answer) > 0
    assert len(result.citations) >= 1
    assert all(c.source == "bio.pdf" for c in result.citations)


@pytest.mark.integration
def test_followup_resolves_via_history(tmp_path, require_ollama):
    """T4.2 -- a pronoun follow-up resolves against the prior turn.

    "Explain that more simply." has no topical words of its own, so in a
    multi-topic corpus its top-scoring evidence can only be the bio doc if
    condense_plus_context rewrote "that" into "...mitochondria..." using the
    previous turn. (We assert on retrieval, not the 3B model's answer wording,
    which sometimes emits a refusal even with context in hand.)
    """
    # Arrange
    index = _index_with(_MULTI_TOPIC_DOCS, tmp_path)
    chat_engine = chat.build_chat_engine(index=index)
    chat.ask(chat_engine, "What is the powerhouse of the cell?")

    # Act
    followup = chat.ask(chat_engine, "Explain that more simply.")

    # Assert: the condense step pulled the topic from history, so the best
    # evidence for the topic-neutral follow-up is the previous turn's doc.
    assert len(followup.citations) >= 1
    assert followup.citations[0].source == "bio.pdf"


@pytest.mark.integration
def test_out_of_doc_question_refuses_without_poisoning_memory(tmp_path, require_ollama):
    """T4.2/T3.3 (chat) -- an out-of-doc question returns REFUSAL_MESSAGE, and the
    engine's memory records that same refusal, not LlamaIndex's raw "Empty Response"
    fallback (which would otherwise leak into the next turn's condense step).
    """
    # Arrange
    index = _index_with(_BIO_DOCS, tmp_path)
    memory = default_memory()
    chat_engine = chat.build_chat_engine(index=index, memory=memory)

    # Act
    result = chat.ask(chat_engine, "What was the exchange rate of the Ottoman lira in 1875?")

    # Assert: refusal contract holds...
    assert result.answer == REFUSAL_MESSAGE
    assert result.citations == []
    # ...and memory was repaired to match what the user saw (deterministic).
    last = memory.get_all()[-1]
    assert last.role == MessageRole.ASSISTANT
    assert last.content == REFUSAL_MESSAGE


@pytest.mark.integration
def test_start_stream_yields_tokens_and_citations(tmp_path, require_ollama):
    """Streaming path: an in-doc question streams a non-empty answer and exposes
    deduped citations up front (mirrors chat.ask's non-refusal contract).
    """
    # Arrange
    index = _index_with(_BIO_DOCS, tmp_path)
    chat_engine = chat.build_chat_engine(index=index)

    # Act
    handle = chat.start_stream(chat_engine, "What is the powerhouse of the cell?")
    streamed = "".join(handle.tokens)  # drain -> joins the history-writer thread

    # Assert
    assert handle.refused is False
    assert len(streamed) > 0
    assert streamed != REFUSAL_MESSAGE
    assert len(handle.citations) >= 1
    assert all(c.source == "bio.pdf" for c in handle.citations)


@pytest.mark.integration
def test_start_stream_refuses_without_poisoning_memory(tmp_path, require_ollama):
    """Streaming refusal: out-of-doc question yields refused=True with no tokens and
    no citations, and the engine's memory is repaired to REFUSAL_MESSAGE (not the raw
    "Empty Response") so the next turn's condense step isn't poisoned.
    """
    # Arrange
    index = _index_with(_BIO_DOCS, tmp_path)
    memory = default_memory()
    chat_engine = chat.build_chat_engine(index=index, memory=memory)

    # Act (same reliably out-of-doc question as the non-streaming refusal test, so it
    # stays comfortably below SIMILARITY_CUTOFF against the bio corpus)
    handle = chat.start_stream(chat_engine, "What was the exchange rate of the Ottoman lira in 1875?")

    # Assert: streaming refusal contract holds...
    assert handle.refused is True
    assert list(handle.tokens) == []
    assert handle.citations == []
    # ...and memory was repaired (deterministic), same as the non-streaming path.
    last = memory.get_all()[-1]
    assert last.role == MessageRole.ASSISTANT
    assert last.content == REFUSAL_MESSAGE


@pytest.mark.integration
def test_reset_clears_memory(tmp_path, require_ollama):
    """T4.3 -- reset() empties the memory buffer (deterministic, no LLM fuzziness)."""
    # Arrange: hold our own memory buffer so we assert on it directly.
    index = _index_with(_BIO_DOCS, tmp_path)
    memory = default_memory()
    chat_engine = chat.build_chat_engine(index=index, memory=memory)
    chat.ask(chat_engine, "What is the powerhouse of the cell?")

    # Act / Assert: a turn populated memory (user + assistant messages)...
    assert len(memory.get()) > 0

    # ...and reset clears it.
    chat_engine.reset()
    assert memory.get() == []


def test_memory_is_token_capped():
    """T4.3 -- the memory buffer is token-capped from config, not left at the
    library default (deterministic; constructs no network call).
    """
    # Act
    memory = default_memory()

    # Assert
    assert memory.token_limit == config.CHAT_MEMORY_TOKEN_LIMIT
