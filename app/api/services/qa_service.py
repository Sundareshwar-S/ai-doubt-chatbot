"""Chat orchestration for `/api/chat/*` (Phase 5, T5.5/T5.6).

Both `ask_question` and `reset_chat` serialize access to the single shared,
process-wide `chat_engine` via `state.chat_lock`: `chat.ask()` mutates the
engine's `ChatMemoryBuffer` (including its refusal-repair write), so two
concurrent requests against the same engine instance could otherwise
interleave and corrupt the transcript. A refusal is a normal `AnswerResult`,
not an exception -- it isn't routed through the T5.2 exception machinery.
"""
from __future__ import annotations

from starlette.concurrency import run_in_threadpool

from app.api.state import AppState
from app.qa.chat import ask as chat_ask
from app.qa.engine import AnswerResult


async def ask_question(state: AppState, question: str) -> AnswerResult:
    async with state.chat_lock:
        return await run_in_threadpool(chat_ask, state.chat_engine, question)


async def reset_chat(state: AppState) -> None:
    async with state.chat_lock:
        await run_in_threadpool(state.chat_engine.reset)
