"""
RAG v2 Conversation Service
----------------------------
Replaces the stub in rag_conversation.py with a real LangGraph-based
multi-turn RAG pipeline backed by Pinecone.

The compiled graph is cached as a module-level singleton to avoid
rebuilding the LangGraph and reconnecting to Pinecone on every request.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph singleton
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_graph():  # type: ignore[return]
    """Build and cache the LangGraph RAG graph (once per process)."""
    from app.rag_v2.config import RagV2Config  # noqa: PLC0415
    from app.rag_v2.graph import build_graph  # noqa: PLC0415
    cfg = RagV2Config.from_env()
    logger.info(
        "Building RAG v2 graph — LLM: groq/%s | Embedding: %s | Pinecone index: %s",
        cfg.groq_chat_model,
        cfg.embedding_model,
        cfg.pinecone_index_name,
    )
    return build_graph(cfg)


# ---------------------------------------------------------------------------
# Public API — called by chat_service.handle_user_message
# ---------------------------------------------------------------------------

async def rag_chat_v2(
    messages: list[dict[str, str]],
    query: str,
) -> tuple[str, dict[str, Any]]:
    """
    Multi-turn RAG conversation via LangGraph + Pinecone.

    Parameters
    ----------
    messages : Prior conversation turns (role/content dicts), newest last.
               Does NOT include the current `query`.
    query    : Latest user message.

    Returns
    -------
    (answer_text, metadata_dict)
        metadata_dict contains: sources, is_fallback, standalone_q
    """
    try:
        graph = _get_graph()
    except Exception as exc:
        logger.error("Failed to build RAG v2 graph: %s", exc)
        return (
            "The RAG service is not available right now. "
            "Please check your PINECONE_API_KEY and LLM API keys.",
            {"error": str(exc), "is_fallback": True},
        )

    initial_state = {
        "query": query,
        "history": messages,
        "chunks": [],
        "answer": "",
        "sources": [],
        "crag_report": {},
        "is_fallback": False,
    }

    try:
        # LangGraph .invoke() is synchronous; run in thread pool to keep async path
        from fastapi.concurrency import run_in_threadpool  # noqa: PLC0415
        result = await run_in_threadpool(graph.invoke, initial_state)
    except Exception as exc:
        logger.exception("LangGraph invocation error: %s", exc)
        return (
            "An error occurred while processing your request. Please try again.",
            {"error": str(exc), "is_fallback": True},
        )

    answer: str = result.get("answer", "")
    meta: dict[str, Any] = {
        "is_fallback": result.get("is_fallback", False),
        "sources": result.get("sources", []),
        "chunks_found": len(result.get("chunks", [])),
        "crag_report": result.get("crag_report", {}),
    }
    return answer, meta
