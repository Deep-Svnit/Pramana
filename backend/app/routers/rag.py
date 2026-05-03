"""
RAG Router
----------
Preserves the original single-turn RAG endpoints for backward compatibility.

POST /rag/ask     → single-turn query (no session / history)
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.dependencies import get_pipeline, get_pipeline_lock
from app.schemas import AnswerResponse, AskRequest, RetrievedChunkResponse

router = APIRouter(prefix="/rag", tags=["rag"])


def _serialize_answer(answer: Any) -> AnswerResponse:
    return AnswerResponse(
        text=answer.text,
        is_fallback=answer.is_fallback,
        verifier_report=answer.verifier_report,
        retrieved=[
            RetrievedChunkResponse(
                id=chunk.id,
                document_id=chunk.document_id,
                page_number=chunk.page_number,
                modality=chunk.modality,
                rank=chunk.rank,
                score=chunk.score,
                citation=chunk.citation,
                metadata=chunk.metadata,
            )
            for chunk in answer.retrieved
        ],
    )


@router.post("/ask", response_model=AnswerResponse)
async def ask(request: AskRequest) -> AnswerResponse:
    """
    Single-turn RAG query — no session or history.
    Preserved for backward compatibility with existing integrations.
    """
    pipeline = get_pipeline()
    lock = get_pipeline_lock()
    try:
        async with lock:
            answer = await run_in_threadpool(pipeline.answer, request.query)
    except RuntimeError as exc:
        if "No ingested records" in str(exc):
            raise HTTPException(status_code=400, detail=str(exc))
        raise
    return _serialize_answer(answer)
