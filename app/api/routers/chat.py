"""`/api/chat` -- ask + reset (Phase 5, T5.5/T5.6).

There is exactly one shared conversation (`state.chat_engine`), not one per
session -- this is a personal, single-user, no-auth tool, so the simplest
viable model wins. Two browser tabs share one conversation; "Clear chat"
(`POST /api/chat/reset`) is the escape hatch.
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, status
from starlette.responses import StreamingResponse

from app.api.dependencies import get_state
from app.api.schemas.chat import ChatAskRequest, ChatAskResponse, ChatCitation
from app.api.services import qa_service
from app.api.state import AppState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/ask", response_model=ChatAskResponse)
async def ask(body: ChatAskRequest, state: AppState = Depends(get_state)) -> ChatAskResponse:
    result = await qa_service.ask_question(state, body.question)
    return ChatAskResponse(
        answer=result.answer,
        citations=[
            ChatCitation(source=c.source, page=c.page, score=c.score) for c in result.citations
        ],
    )


@router.post("/stream")
async def stream(body: ChatAskRequest, state: AppState = Depends(get_state)) -> StreamingResponse:
    """Same turn as `/ask`, streamed as newline-delimited JSON so the UI can render
    the answer token-by-token instead of waiting for the whole CPU generation.

    Emits one JSON object per line: `{"type":"token","text":...}` deltas, then a
    final `{"type":"done","citations":[...]}`. Because streaming responds 200 up
    front, a mid-stream failure can't use the T5.2 error envelope, so it surfaces
    as a terminal `{"type":"error","message":...}` event instead.
    """

    async def ndjson():
        try:
            async for event in qa_service.ask_question_stream(state, body.question):
                yield json.dumps(event) + "\n"
        except Exception:  # noqa: BLE001 -- headers already sent; report in-band + log
            logger.exception("Chat stream failed")
            yield json.dumps({"type": "error", "message": "Could not generate an answer."}) + "\n"

    return StreamingResponse(ndjson(), media_type="application/x-ndjson")


@router.post("/reset", status_code=status.HTTP_204_NO_CONTENT)
async def reset(state: AppState = Depends(get_state)) -> None:
    await qa_service.reset_chat(state)
