"""Request/response schemas for `/api/chat` (Phase 5)."""
from __future__ import annotations

from pydantic import BaseModel


class ChatAskRequest(BaseModel):
    question: str


class ChatCitation(BaseModel):
    source: str
    page: int
    score: float


class ChatAskResponse(BaseModel):
    answer: str
    citations: list[ChatCitation]
