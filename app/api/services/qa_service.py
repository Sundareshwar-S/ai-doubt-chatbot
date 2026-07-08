"""Chat orchestration for `/api/chat/*` (Phase 5, T5.5/T5.6).

Both `ask_question` and `reset_chat` serialize access to the single shared,
process-wide `chat_engine` via `state.chat_lock`: `chat.ask()` mutates the
engine's `ChatMemoryBuffer` (including its refusal-repair write), so two
concurrent requests against the same engine instance could otherwise
interleave and corrupt the transcript. A refusal is a normal `AnswerResult`,
not an exception -- it isn't routed through the T5.2 exception machinery.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from starlette.concurrency import run_in_threadpool

from app.api.state import AppState
from app.qa.chat import ask as chat_ask
from app.qa.chat import start_stream
from app.qa.engine import AnswerResult
from app.qa.prompts import REFUSAL_MESSAGE

# Distinguishes "generator exhausted" from a legitimately yielded token when
# pulling the sync token stream one item at a time via run_in_threadpool.
_STREAM_DONE = object()


async def ask_question(state: AppState, question: str) -> AnswerResult:
    async with state.chat_lock:
        return await run_in_threadpool(chat_ask, state.chat_engine, question)


async def ask_question_stream(state: AppState, question: str) -> AsyncIterator[dict]:
    """Stream one chat turn as structured NDJSON-ready events.

    Holds `chat_lock` for the *entire* stream (retrieval + generation + the
    engine's background memory write), since the shared `ChatMemoryBuffer` must
    not interleave with a concurrent request. The chat engine's token generator
    is synchronous, so each token is pulled in the threadpool -- never iterated
    directly from this async generator. Yields `{"type": "token", ...}` events
    followed by a final `{"type": "done", "citations": [...]}`.
    """
    async with state.chat_lock:
        handle = await run_in_threadpool(start_stream, state.chat_engine, question)

        if handle.refused:
            yield {"type": "token", "text": REFUSAL_MESSAGE}
            yield {"type": "done", "citations": []}
            return

        tokens = handle.tokens
        while True:
            token = await run_in_threadpool(next, tokens, _STREAM_DONE)
            if token is _STREAM_DONE:
                break
            if token:
                yield {"type": "token", "text": token}

        yield {
            "type": "done",
            "citations": [
                {"source": c.source, "page": c.page, "score": c.score}
                for c in handle.citations
            ],
        }


async def reset_chat(state: AppState) -> None:
    async with state.chat_lock:
        await run_in_threadpool(state.chat_engine.reset)
