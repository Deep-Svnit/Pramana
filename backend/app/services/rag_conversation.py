"""
RAG Conversation Stub
---------------------
PLACEHOLDER — replace the body of `rag_chat` with real multi-turn RAG
conversation logic when it's ready.

The contract:
    messages  — list of {"role": "user"|"assistant", "content": str}
                representing the full chat history BEFORE the current query.
    query     — the latest user message string.
    pipeline  — the MultimodalRAGPipeline singleton (or None if not yet loaded).

Returns a plain-text assistant response string.
"""
from __future__ import annotations

from typing import Any


async def rag_chat(
    messages: list[dict[str, str]],
    query: str,
    pipeline: Any | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Multi-turn RAG conversation — STUB implementation.

    Currently delegates to the pipeline's single-turn `answer()` method,
    ignoring prior history. Replace this entire function body when the real
    RAG conversation is implemented.

    Returns
    -------
    (response_text, metadata_dict)
        metadata_dict may contain keys like `sources`, `is_fallback`, etc.
        and will be stored as JSON in ChatMessage.meta.
    """
    if pipeline is None:
        # No pipeline loaded yet — return a friendly placeholder
        return (
            "I'm ready to help! Please upload some documents first so I can "
            "answer questions about them.",
            {},
        )

    # --- STUB: single-turn fallback via existing pipeline.answer() ---
    # TODO: Replace with real multi-turn RAG conversation implementation.
    from fastapi.concurrency import run_in_threadpool  # noqa: PLC0415

    try:
        answer = await run_in_threadpool(pipeline.answer, query)
        meta: dict[str, Any] = {
            "is_fallback": answer.is_fallback,
            "verifier_report": answer.verifier_report,
            "sources": [
                {
                    "id": c.id,
                    "document_id": c.document_id,
                    "citation": c.citation,
                    "score": c.score,
                }
                for c in answer.retrieved
            ],
        }
        return answer.text, meta
    except RuntimeError as exc:
        if "No ingested records" in str(exc):
            return (
                "No documents have been ingested yet. Please upload a document first.",
                {"error": str(exc)},
            )
        raise
