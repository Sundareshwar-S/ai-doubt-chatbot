"""`/api/chat` -- ask + reset (Phase 5, T5.5/T5.6).

There is exactly one shared conversation (`state.chat_engine`), not one per
session -- this is a personal, single-user, no-auth tool, so the simplest
viable model wins. Two browser tabs share one conversation; "Clear chat"
(`POST /api/chat/reset`) is the escape hatch.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.api.dependencies import get_state
from app.api.schemas.chat import ChatAskRequest, ChatAskResponse, ChatCitation
from app.api.services import qa_service
from app.api.state import AppState

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


@router.post("/reset", status_code=status.HTTP_204_NO_CONTENT)
async def reset(state: AppState = Depends(get_state)) -> None:
    await qa_service.reset_chat(state)
