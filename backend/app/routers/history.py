"""
History Router
--------------
Endpoints for the sidebar "recent chats" list.

GET /history/sessions          → paginated list of recent sessions
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.schemas import HistoryResponse
from app.services import chat_service

router = APIRouter(prefix="/history", tags=["history"])


@router.get("/sessions", response_model=HistoryResponse)
def recent_sessions(
    limit: int = Query(30, ge=1, le=100, description="Max sessions to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db),
):
    """
    Return a paginated list of chat sessions ordered by last activity (newest first).

    Designed for the sidebar recent-chats panel — each entry includes:
    - session id + title
    - last message preview (≤80 chars)
    - total message count
    - last-updated timestamp
    """
    sessions, total = chat_service.list_sessions(db, limit=limit, offset=offset)
    return HistoryResponse(sessions=sessions, total=total)
