"""Query engine + grounded citations.

Short-circuits to a fixed refusal string when SimilarityPostprocessor drops
every retrieved node -- BaseSynthesizer.synthesize() already skips the LLM
call in that case (see its `if len(nodes) == 0` branch), so this only swaps
LlamaIndex's generic "Empty Response" for our own grounded refusal text.
"""
from __future__ import annotations

from dataclasses import dataclass

from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.llms.ollama import Ollama

import config
from app.index.pipeline import load_index
from app.qa.prompts import GROUNDED_QA_PROMPT, REFUSAL_MESSAGE


@dataclass(frozen=True)
class Citation:
    source: str
    page: int
    score: float


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    citations: list[Citation]


def default_llm() -> Ollama:
    """Build the real Ollama LLM client from config (injectable for tests)."""
    return Ollama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        request_timeout=config.REQUEST_TIMEOUT,
        context_window=config.CONTEXT_WINDOW,
    )


def build_query_engine(index: VectorStoreIndex, llm: Ollama | None = None) -> BaseQueryEngine:
    """Query engine with the grounding prompt and similarity-cutoff postprocessor applied."""
    return index.as_query_engine(
        llm=llm or default_llm(),
        similarity_top_k=config.TOP_K,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=config.SIMILARITY_CUTOFF)],
        text_qa_template=GROUNDED_QA_PROMPT,
    )


def _dedup_citations(source_nodes: list[NodeWithScore]) -> list[Citation]:
    """Dedup by (source, page): the same page can surface via >1 retrieved chunk
    (e.g. a long page split across multiple nodes); keep the highest score seen
    for that page, ordered best-evidence-first.
    """
    best: dict[tuple[str, int], Citation] = {}
    for node_with_score in source_nodes:
        meta = node_with_score.node.metadata
        key = (meta.get("source", "unknown"), meta.get("page", 0))
        score = node_with_score.score or 0.0
        if key not in best or score > best[key].score:
            best[key] = Citation(source=key[0], page=key[1], score=score)
    return sorted(best.values(), key=lambda c: c.score, reverse=True)


def answer_question(
    question: str,
    index: VectorStoreIndex | None = None,
    llm: Ollama | None = None,
) -> AnswerResult:
    """Answer a question from the indexed documents, or return REFUSAL_MESSAGE
    if the similarity cutoff drops every retrieved node (see module docstring).
    """
    index = index or load_index()
    response = build_query_engine(index, llm=llm).query(question)

    if not response.source_nodes:
        return AnswerResult(answer=REFUSAL_MESSAGE, citations=[])

    return AnswerResult(
        answer=str(response),
        citations=_dedup_citations(response.source_nodes),
    )
