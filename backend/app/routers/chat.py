"""
Chat Router
-----------
Endpoints for managing chat sessions and multi-turn messages.

POST   /chat/sessions                        → create a new session
GET    /chat/sessions/{session_id}           → get session details
DELETE /chat/sessions/{session_id}           → delete a session
GET    /chat/sessions/{session_id}/messages  → list all messages in a session
POST   /chat/sessions/{session_id}/messages  → send a user message + get reply
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.dependencies import get_db, get_pipeline
from app.schemas import (
    ChatTurnResponse,
    MessageCreate,
    MessageResponse,
    SessionCreate,
    SessionResponse,
)
from app.services import chat_service

router = APIRouter(prefix="/chat", tags=["chat"])


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------

@router.post("/sessions", response_model=SessionResponse, status_code=201)
def create_session(
    body: SessionCreate,
    db: Session = Depends(get_db),
):
    """Create a new chat session. Optionally provide an initial title."""
    return chat_service.create_session(db, title=body.title)


@router.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str, db: Session = Depends(get_db)):
    """Get details for a specific chat session."""
    session = chat_service.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse.model_validate(session)


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str, db: Session = Depends(get_db)):
    """Delete a chat session and all its messages."""
    ok = chat_service.delete_session(db, session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")


# ---------------------------------------------------------------------------
# Message endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/sessions/{session_id}/messages",
    response_model=list[MessageResponse],
)
def list_messages(session_id: str, db: Session = Depends(get_db)):
    """Return all messages in a session ordered by creation time."""
    session = chat_service.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return chat_service.get_messages(db, session_id)


@router.post(
    "/sessions/{session_id}/messages",
    response_model=ChatTurnResponse,
    status_code=201,
)
async def send_message(
    session_id: str,
    body: MessageCreate,
    db: Session = Depends(get_db),
):
    """
    Send a user message to a session.

    The full conversation history is passed to the RAG conversation service
    to produce a context-aware, multi-turn assistant reply.
    Returns both the persisted user message and the assistant's reply.
    """
    session = chat_service.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Try to get pipeline — may fail if env not configured; that's OK
    try:
        pipeline = get_pipeline()
    except Exception:  # noqa: BLE001
        pipeline = None

    try:
        user_msg, assistant_msg = await chat_service.handle_user_message(
            db=db,
            session_id=session_id,
            content=body.content,
            pipeline=pipeline,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return ChatTurnResponse(user_message=user_msg, assistant_message=assistant_msg)
