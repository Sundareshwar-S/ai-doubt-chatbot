"""Multi-turn chat engine (Phase 4).

Wraps LlamaIndex's `condense_plus_context` chat engine: each turn condenses the
follow-up + prior history into a standalone retrieval query *before* retrieval,
so a pronoun follow-up ("explain that more simply") resolves against earlier
turns. Memory is token-capped (`config.CHAT_MEMORY_TOKEN_LIMIT`) to keep RAM and
the Ollama context window bounded on CPU inference.

Reuses `app/qa/engine.py`'s citation types + helpers rather than duplicating them:
the (source, page) dedup and the grounded-refusal short-circuit are identical to
the single-shot query path.

Refusal handling differs from the query engine, though. When the similarity
cutoff drops every node, LlamaIndex's synthesizer returns the literal string
"Empty Response" *and the chat engine writes that into its memory* -- unlike
`engine.answer_question`, which is stateless. `ask()` therefore both (a) returns
our fixed REFUSAL_MESSAGE and (b) rewrites that poisoned message in memory, so
the next turn's condense step never sees LlamaIndex's internal artifact.

`start_stream()` doesn't go through `chat_engine.stream_chat()` at all -- see its
docstring for why (LlamaIndex's `Refine` synthesizer buffers the whole answer
before yielding once, defeating real token streaming) -- so it never produces
that poisoned artifact in the first place; refused turns write REFUSAL_MESSAGE
to memory directly.

The chat engine is stateful (it owns the memory buffer); a caller holds one
instance and its memory accumulates across turns. Phase 5 will build an instance
in the FastAPI lifespan; because a single `ChatMemoryBuffer` is one linear
conversation, memory should be scoped per conversation/session there (not shared
across concurrent requests) -- that scoping is a Phase 5 design decision, out of
scope here.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from llama_index.core import VectorStoreIndex
from llama_index.core.base.llms.generic_utils import messages_to_history_str
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import MetadataMode, QueryBundle
from llama_index.llms.ollama import Ollama

import config
from app.index.pipeline import load_index
from app.qa.engine import AnswerResult, Citation, dedup_citations, default_llm
from app.qa.prompts import GROUNDED_CHAT_CONTEXT_PROMPT, REFUSAL_MESSAGE

__all__ = [
    "AnswerResult",
    "Citation",
    "StreamHandle",
    "default_memory",
    "build_chat_engine",
    "ask",
    "start_stream",
]


def default_memory(llm: Ollama | None = None) -> ChatMemoryBuffer:
    """Token-capped chat memory (injectable for tests).

    The token limit bounds retained history so it fits `config.CONTEXT_WINDOW`
    alongside the retrieved chunks + prompt.
    """
    return ChatMemoryBuffer.from_defaults(
        token_limit=config.CHAT_MEMORY_TOKEN_LIMIT,
        llm=llm or default_llm(),
    )


def build_chat_engine(
    index: VectorStoreIndex | None = None,
    llm: Ollama | None = None,
    memory: ChatMemoryBuffer | None = None,
) -> BaseChatEngine:
    """Stateful condense_plus_context chat engine with grounding + similarity cutoff.

    Returns the engine itself (not a per-call result): the caller holds it across
    turns so its memory accumulates. Same retrieval knobs as the query engine
    (`TOP_K`, `SIMILARITY_CUTOFF`) plus a grounded context prompt.
    """
    index = index or load_index()
    llm = llm or default_llm()
    return index.as_chat_engine(
        chat_mode="condense_plus_context",
        llm=llm,
        memory=memory or default_memory(llm),
        similarity_top_k=config.TOP_K,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=config.SIMILARITY_CUTOFF)],
        context_prompt=GROUNDED_CHAT_CONTEXT_PROMPT,
    )


def _repair_refused_turn(chat_engine: BaseChatEngine) -> None:
    """Replace the just-written "Empty Response" assistant message with our
    REFUSAL_MESSAGE, so the transcript and the next turn's condense input match
    what the user was actually shown. No-op if the memory isn't reachable or the
    last message isn't the assistant turn we just added.
    """
    memory = getattr(chat_engine, "_memory", None)
    if memory is None:
        return
    history = memory.get_all()
    if history and history[-1].role == MessageRole.ASSISTANT:
        history[-1] = ChatMessage(role=MessageRole.ASSISTANT, content=REFUSAL_MESSAGE)
        memory.set(history)


def ask(chat_engine: BaseChatEngine, question: str) -> AnswerResult:
    """Send one turn through the chat engine, returning a grounded AnswerResult.

    Mirrors `engine.answer_question`'s refusal contract: when the similarity
    cutoff drops every retrieved node, substitute the fixed REFUSAL_MESSAGE
    (deterministic) instead of whatever the LLM would say with no context, and
    repair the poisoned "Empty Response" that the chat engine wrote to memory.
    """
    response = chat_engine.chat(question)

    if not response.source_nodes:
        _repair_refused_turn(chat_engine)
        return AnswerResult(answer=REFUSAL_MESSAGE, citations=[])

    return AnswerResult(
        answer=str(response),
        citations=dedup_citations(response.source_nodes),
    )


@dataclass(frozen=True)
class StreamHandle:
    """The start of a streaming chat turn, in one of two mutually exclusive states.

    `refused=True`: the similarity cutoff dropped every node. `REFUSAL_MESSAGE`
    has already been written to memory as the assistant turn (no LlamaIndex
    "Empty Response" artifact to repair -- unlike `ask()`, `start_stream` never
    calls the LLM for a refused turn, so nothing poisoned is ever written),
    `tokens` is empty, and the caller should emit `REFUSAL_MESSAGE` -- the
    streaming mirror of `ask()`'s refusal contract.

    `refused=False`: `tokens` yields the live answer deltas one at a time
    (drive it to completion so the full answer gets written to memory) and
    `citations` are the deduped source citations, available up front.
    """

    refused: bool
    tokens: Iterator[str]
    citations: list[Citation]


def start_stream(chat_engine: BaseChatEngine, question: str) -> StreamHandle:
    """Begin a streaming turn. Blocking (runs condense + retrieval); the caller
    drives `tokens` to trigger generation and the memory write.

    Deliberately bypasses `chat_engine.stream_chat()`. LlamaIndex's
    `condense_plus_context` synthesizes the answer via `Refine`/`CompactAndRefine`,
    whose `DefaultRefineProgram.stream_call()` fully drains the LLM's token stream
    into one string before yielding it *once* ("we want to mimic [structured
    output] behavior", per that method's own comment) -- so `response_gen` there
    delivers the whole answer as a single chunk, not real per-token deltas
    (confirmed empirically: `start_stream` blocked for the full generation time,
    then yielded exactly one item). Ollama itself streams token-by-token fine, so
    this function replicates condense_plus_context's condense-question -> retrieve
    -> grounded-prompt steps by hand, then drives `llm.stream_chat()` directly for
    genuine incremental delivery. Reaches into the engine's private attributes
    (`_retriever`, `_llm`, `_memory`, ...) since `build_chat_engine()` only exposes
    the assembled `BaseChatEngine`, not its components separately.
    """
    retriever = chat_engine._retriever
    llm = chat_engine._llm
    memory = chat_engine._memory
    node_postprocessors = chat_engine._node_postprocessors
    condense_prompt = chat_engine._condense_prompt_template
    skip_condense = chat_engine._skip_condense

    chat_history = memory.get(input=question)

    condensed_question = question
    if not skip_condense and chat_history:
        llm_input = condense_prompt.format(
            chat_history=messages_to_history_str(chat_history), question=question
        )
        condensed_question = str(llm.complete(llm_input))

    nodes = retriever.retrieve(condensed_question)
    for postprocessor in node_postprocessors:
        nodes = postprocessor.postprocess_nodes(
            nodes, query_bundle=QueryBundle(condensed_question)
        )

    memory.put(ChatMessage(role=MessageRole.USER, content=question))

    if not nodes:
        memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=REFUSAL_MESSAGE))
        return StreamHandle(refused=True, tokens=iter(()), citations=[])

    context_str = "\n\n".join(
        n.node.get_content(metadata_mode=MetadataMode.LLM) for n in nodes
    )
    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=GROUNDED_CHAT_CONTEXT_PROMPT.format(context_str=context_str),
        ),
        *chat_history,
        ChatMessage(role=MessageRole.USER, content=question),
    ]

    def token_gen() -> Iterator[str]:
        full_response = ""
        for chunk in llm.stream_chat(messages):
            delta = chunk.delta or ""
            if delta:
                full_response += delta
                yield delta
        memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=full_response.strip()))

    return StreamHandle(
        refused=False,
        tokens=token_gen(),
        citations=dedup_citations(nodes),
    )
