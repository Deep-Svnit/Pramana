"""
Chat v2 Router
--------------
Same session / message model as the v1 chat router, but delegates
to the LangGraph + Pinecone RAG pipeline (rag_v2_conversation.py).

POST   /chat/v2/sessions                        → create session
GET    /chat/v2/sessions/{id}                   → get session
DELETE /chat/v2/sessions/{id}                   → delete session
GET    /chat/v2/sessions/{id}/messages          → list messages
POST   /chat/v2/sessions/{id}/messages          → send message + get reply
"""
from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.models import ChatMessage, ChatSession
from app.schemas import (
    ChatTurnResponse,
    MessageCreate,
    MessageResponse,
    SessionCreate,
    SessionResponse,
)
from app.services import chat_service
from app.services.rag_v2_conversation import rag_chat_v2

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat/v2", tags=["chat-v2"])


# ---------------------------------------------------------------------------
# Session endpoints (delegate to existing chat_service helpers)
# ---------------------------------------------------------------------------

@router.post("/sessions", response_model=SessionResponse, status_code=201)
def create_session(body: SessionCreate, db: Session = Depends(get_db)):
    """Create a new chat session."""
    return chat_service.create_session(db, title=body.title)


@router.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str, db: Session = Depends(get_db)):
    session = chat_service.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse.model_validate(session)


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str, db: Session = Depends(get_db)):
    ok = chat_service.delete_session(db, session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")


@router.get("/sessions/{session_id}/messages", response_model=list[MessageResponse])
def list_messages(session_id: str, db: Session = Depends(get_db)):
    if not chat_service.get_session(db, session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return chat_service.get_messages(db, session_id)


# ---------------------------------------------------------------------------
# Message endpoint — uses LangGraph v2
# ---------------------------------------------------------------------------

@router.post(
    "/sessions/{session_id}/messages",
    response_model=ChatTurnResponse,
    status_code=201,
)
async def send_message_v2(
    session_id: str,
    body: MessageCreate,
    db: Session = Depends(get_db),
):
    """
    Send a user message and receive a reply from the LangGraph RAG pipeline.

    - Full conversation history is passed to the graph for context-aware QA.
    - Sources from Pinecone are stored in the assistant message metadata.
    """
    session = chat_service.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # 1. Persist user message
    from sqlalchemy import select  # noqa: PLC0415
    user_msg = ChatMessage(session_id=session_id, role="user", content=body.content)
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    # 2. Build prior history (everything before the current message)
    prior_msgs = (
        db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .where(ChatMessage.id != user_msg.id)
            .order_by(ChatMessage.created_at)
        )
        .scalars()
        .all()
    )
    history = [{"role": m.role, "content": m.content} for m in prior_msgs]

    # 3. Call LangGraph v2
    try:
        response_text, meta = await rag_chat_v2(messages=history, query=body.content)
    except Exception as exc:
        logger.exception("rag_chat_v2 raised an unexpected error: %s", exc)
        response_text = "An unexpected error occurred. Please try again."
        meta = {"error": str(exc), "is_fallback": True}

    # 4. Persist assistant message
    assistant_msg = ChatMessage(
        session_id=session_id,
        role="assistant",
        content=response_text,
        meta=json.dumps(meta) if meta else None,
    )
    db.add(assistant_msg)

    # 5. Update session title from first user message
    if session.title == "New Chat":
        session.title = body.content[:60] + ("…" if len(body.content) > 60 else "")

    db.commit()
    db.refresh(assistant_msg)
    db.refresh(session)

    def _to_schema(m: ChatMessage) -> MessageResponse:
        raw_meta: dict[str, Any] | None = None
        if m.meta:
            try:
                raw_meta = json.loads(m.meta)
            except Exception:
                raw_meta = None
        return MessageResponse(
            id=m.id,
            session_id=m.session_id,
            role=m.role,
            content=m.content,
            meta=raw_meta,
            created_at=m.created_at,
        )

    return ChatTurnResponse(
        user_message=_to_schema(user_msg),
        assistant_message=_to_schema(assistant_msg),
    )
