from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Chat Session
# ---------------------------------------------------------------------------

class SessionCreate(BaseModel):
    title: str | None = Field(None, description="Optional title; auto-set from first message if omitted")


class SessionResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Chat Message
# ---------------------------------------------------------------------------

class MessageCreate(BaseModel):
    content: str = Field(..., min_length=1, description="User's message text")


class MessageResponse(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    meta: dict[str, Any] | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatTurnResponse(BaseModel):
    user_message: MessageResponse
    assistant_message: MessageResponse


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

class SessionSummary(BaseModel):
    """Lightweight session info for the sidebar recent-chats list."""
    id: str
    title: str
    last_message_preview: str | None = None
    message_count: int
    updated_at: datetime

    model_config = {"from_attributes": True}


class HistoryResponse(BaseModel):
    sessions: list[SessionSummary]
    total: int


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

class IngestedFileResponse(BaseModel):
    id: str
    original_filename: str
    doc_type: str = "document"
    section: str = "general"
    context: str = ""
    document_id: str | None
    status: str
    error: str | None
    created_at: datetime

    model_config = {"from_attributes": True}



class IngestListResponse(BaseModel):
    files: list[IngestedFileResponse]
    total: int


# ---------------------------------------------------------------------------
# RAG (single-turn, backward compat)
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    query: str


class RetrievedChunkResponse(BaseModel):
    id: str
    document_id: str
    page_number: int
    modality: str
    rank: int
    score: float
    citation: str
    metadata: dict[str, Any]


class AnswerResponse(BaseModel):
    text: str
    is_fallback: bool
    verifier_report: dict[str, Any]
    retrieved: list[RetrievedChunkResponse]
