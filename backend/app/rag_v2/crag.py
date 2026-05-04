"""CRAG verification using lettuce-detect and page-level VLM fallback."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.database import SessionLocal
from app.models import IngestedFile
from app.rag_v2.config import RagV2Config
from app.rag_v2.vision import build_vision_client, pdf_page_to_png_bytes

logger = logging.getLogger(__name__)

NOT_FOUND = "Not found in the Documents"


@lru_cache(maxsize=2)
def _get_detector(model_path: str, max_length: int):
    from lettucedetect import HallucinationDetector

    return HallucinationDetector(
        method="transformer",
        model_path=model_path,
        max_length=max_length,
    )


@lru_cache(maxsize=1)
def _get_page_client(
    provider: str,
    model: str,
    device: str,
    max_new_tokens: int,
    min_pixels: int,
    max_pixels: int,
    api_keys: tuple[str, ...],
):
    return build_vision_client(
        provider=provider,
        model=model,
        api_keys=api_keys,
        device=device,
        max_new_tokens=max_new_tokens,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )


def _answer_claims_not_found(answer: str) -> bool:
    lowered = answer.lower()
    not_found_markers = [
        "not found in the documents",
        "not present in the document",
        "not present in the documents",
        "doesn't contain enough information",
        "does not contain enough information",
        "couldn't find",
        "could not find",
    ]
    return any(marker in lowered for marker in not_found_markers)


def verify_answer(
    *,
    config: RagV2Config,
    query: str,
    answer: str,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run lettuce-detect over the generated answer and retrieved chunk text."""
    if not config.crag_enabled:
        return {"status": "disabled", "needs_page_fallback": False, "spans": []}

    context = [
        str(chunk.get("metadata", {}).get("text", "")).strip()
        for chunk in chunks
        if str(chunk.get("metadata", {}).get("text", "")).strip()
    ]
    if not context or not answer.strip():
        return {"status": "no_context", "needs_page_fallback": True, "spans": []}

    try:
        detector = _get_detector(config.lettuce_model_path, config.lettuce_max_length)
        spans = detector.predict(
            context=context,
            answer=answer,
            question=query,
            output_format="spans",
        )
    except Exception as exc:
        logger.warning("lettuce-detect verification failed: %s", exc)
        return {
            "status": "error",
            "error": str(exc),
            "needs_page_fallback": True,
            "spans": [],
        }

    high_conf_spans = [
        span for span in spans
        if float(span.get("confidence", 1.0)) >= config.lettuce_threshold
    ]
    needs_page_fallback = bool(high_conf_spans) or _answer_claims_not_found(answer)
    return {
        "status": "checked",
        "needs_page_fallback": needs_page_fallback,
        "spans": high_conf_spans,
        "raw_span_count": len(spans),
    }


def _candidate_pages(chunks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    seen: set[tuple[str, int]] = set()
    pages: list[dict[str, Any]] = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        document_id = str(meta.get("document_id", ""))
        try:
            page_number = int(meta.get("page_number"))
        except (TypeError, ValueError):
            continue
        key = (document_id, page_number)
        if not document_id or key in seen:
            continue
        seen.add(key)
        pages.append(
            {
                "document_id": document_id,
                "filename": meta.get("filename", ""),
                "page_number": page_number,
                "score": float(chunk.get("score", 0.0)),
            }
        )
        if len(pages) >= limit:
            break
    return pages


def _stored_path_for_document(document_id: str) -> Path | None:
    db = SessionLocal()
    try:
        record = (
            db.query(IngestedFile)
            .filter(IngestedFile.document_id == document_id)
            .first()
        )
        if not record:
            return None
        path = Path(record.stored_path)
        return path if path.exists() else None
    finally:
        db.close()


def page_level_fallback(
    *,
    config: RagV2Config,
    query: str,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Re-check likely source pages with Qwen VLM.

    Returns an answer only if Qwen can find it directly on the rendered page.
    """
    client = _get_page_client(
        config.vision_provider,
        config.vision_model,
        config.qwen_device,
        config.qwen_max_new_tokens,
        config.qwen_min_pixels,
        config.qwen_max_pixels,
        config.gemini_api_keys,
    )
    checked_pages: list[dict[str, Any]] = []

    for page in _candidate_pages(chunks, config.crag_page_top_k):
        stored_path = _stored_path_for_document(page["document_id"])
        if stored_path is None or stored_path.suffix.lower() != ".pdf":
            checked_pages.append({**page, "status": "source_pdf_not_found"})
            continue

        image_bytes = pdf_page_to_png_bytes(stored_path, page["page_number"], config.vision_dpi)
        if not image_bytes:
            checked_pages.append({**page, "status": "render_failed"})
            continue

        try:
            page_answer = client.answer_page(
                image_bytes=image_bytes,
                query=query,
                page_number=page["page_number"],
            ).strip()
        except Exception as exc:
            logger.warning("Qwen page verification failed for %s p%s: %s", stored_path.name, page["page_number"], exc)
            checked_pages.append({**page, "status": "vlm_error", "error": str(exc)})
            continue

        checked_pages.append({**page, "status": "checked", "answer": page_answer})
        if page_answer and page_answer != NOT_FOUND and NOT_FOUND.lower() not in page_answer.lower():
            return {
                "answer": page_answer,
                "found": True,
                "checked_pages": checked_pages,
            }

    return {
        "answer": NOT_FOUND,
        "found": False,
        "checked_pages": checked_pages,
    }
