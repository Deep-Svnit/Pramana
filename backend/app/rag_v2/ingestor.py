"""Document ingestor — parse → chunk → embed → upsert to Pinecone."""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from time import perf_counter
from typing import Any

from app.rag_v2.config import RagV2Config
from app.rag_v2.embedder import build_embedder
from app.rag_v2.vector_store import PineconeStore
from app.rag_v2.vision import build_vision_client, extract_page_with_vision

logger = logging.getLogger(__name__)


def _progress(message: str) -> None:
    logger.info(message)
    print(f"[ingest-v2] {message}", flush=True)


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _extract_text_pages(path: Path) -> list[tuple[int, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            import pdfplumber
        except ImportError as exc:
            raise ImportError("Run: pip install pdfplumber") from exc
        pages = []
        with pdfplumber.open(path) as pdf:
            _progress(f"{path.name}: reading normal text from {len(pdf.pages)} PDF pages")
            for i, page in enumerate(pdf.pages, start=1):
                page_start = perf_counter()
                _progress(f"{path.name}: text page {i}/{len(pdf.pages)} - extracting")
                text = page.extract_text() or ""
                if text.strip():
                    pages.append((i, text))
                    _progress(
                        f"{path.name}: text page {i}/{len(pdf.pages)} - "
                        f"extracted {len(text)} chars in {perf_counter() - page_start:.1f}s"
                    )
                else:
                    _progress(
                        f"{path.name}: text page {i}/{len(pdf.pages)} - no text found "
                        f"in {perf_counter() - page_start:.1f}s"
                    )
        return pages
    if suffix == ".txt":
        _progress(f"{path.name}: reading TXT file")
        text = path.read_text(encoding="utf-8", errors="ignore")
        _progress(f"{path.name}: read {len(text)} chars from TXT")
        return [(1, text)]
    if suffix == ".docx":
        try:
            import docx
        except ImportError as exc:
            raise ImportError("Run: pip install python-docx") from exc
        doc = docx.Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs)
        _progress(f"{path.name}: read {len(text)} chars from DOCX")
        return [(1, text)]
    raise ValueError(f"Unsupported file type: {suffix}")


def _extract_vlm_pages(path: Path, vision_client, dpi: int) -> tuple[list[tuple[int, str]], list[dict[str, Any]]]:
    if vision_client is None or path.suffix.lower() != ".pdf":
        return [], []
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError("Run: pip install pdfplumber") from exc

    pages: list[tuple[int, str]] = []
    failures: list[dict[str, Any]] = []
    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
    _progress(f"{path.name}: starting VLM feature extraction for {page_count} PDF pages")
    for page_number in range(1, page_count + 1):
        page_start = perf_counter()
        _progress(f"{path.name}: VLM page {page_number}/{page_count} - rendering and sending to vision model")
        vision_text = extract_page_with_vision(vision_client, path, page_number, dpi=dpi)
        if vision_text:
            pages.append((page_number, vision_text))
            _progress(
                f"{path.name}: VLM page {page_number}/{page_count} - "
                f"extracted {len(vision_text)} chars in {perf_counter() - page_start:.1f}s"
            )
        else:
            failures.append({
                "filename": path.name,
                "page_number": page_number,
                "reason": "empty_or_failed_visual_extraction",
            })
            _progress(
                f"{path.name}: VLM page {page_number}/{page_count} - no content returned "
                f"in {perf_counter() - page_start:.1f}s"
            )
    return pages, failures


def _chunk(text: str, size: int, overlap: int) -> list[str]:
    """
    Split text into chunks of at most `size` characters, never cutting mid-word.
    Tries to break at sentence boundaries first; falls back to word boundaries.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # Split into sentences
    sentences = re.split(r'(?<=[.!?•\n])\s+', text)

    chunks: list[str] = []
    current = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) > size:
            words = sent.split()
            piece = ""
            for word in words:
                if not piece or len(piece) + 1 + len(word) <= size:
                    piece = (piece + " " + word).strip() if piece else word
                else:
                    if current and len(current) >= 50:
                        chunks.append(current)
                        current = ""
                    if len(piece) >= 50:
                        chunks.append(piece)
                    piece = word
            if piece:
                if not current or len(current) + 1 + len(piece) <= size:
                    current = (current + " " + piece).strip() if current else piece
                else:
                    if len(current) >= 50:
                        chunks.append(current)
                    current = piece
            continue
        # Sentence fits in current chunk
        if not current or len(current) + 1 + len(sent) <= size:
            current = (current + " " + sent).strip() if current else sent
        else:
            if len(current) >= 50:
                chunks.append(current)
            # Overlap: carry last `overlap` chars (word-aligned) into next chunk
            if overlap and current:
                carry = current[-overlap:]
                boundary = carry.find(" ")
                current = carry[boundary + 1:].strip() if boundary != -1 else ""
            else:
                current = ""
            current = (current + " " + sent).strip() if current else sent

    if len(current) >= 50:
        chunks.append(current)

    return chunks


def _doc_id(path: Path) -> str:
    return "doc_" + hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _batched(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _local_raptor_summary(child_texts: list[str], max_chars: int) -> str:
    """
    Build an extractive RAPTOR parent without any LLM/API calls.

    Keeps the most information-dense lines first, especially lines containing
    numbers, dates, currencies, percentages, or financial/table language.
    """
    important_patterns = re.compile(
        r"(\d|%|₹|\$|rs\.?|inr|crore|million|billion|ebitda|revenue|pat|cagr|"
        r"margin|growth|fy\d|q[1-4]|table|total|profit|loss|cash|debt)",
        re.IGNORECASE,
    )
    candidates: list[tuple[int, int, str]] = []
    order = 0
    for text in child_texts:
        parts = re.split(r"(?<=[.!?])\s+|\n+", text)
        for part in parts:
            cleaned = re.sub(r"\s+", " ", part).strip()
            if len(cleaned) < 30:
                continue
            score = 2 if important_patterns.search(cleaned) else 1
            score += min(3, len(re.findall(r"\d", cleaned)) // 3)
            candidates.append((-score, order, cleaned))
            order += 1

    if not candidates:
        return " ".join(child_texts)[:max_chars]

    selected: list[str] = []
    total = 0
    for _, _, sentence in sorted(candidates):
        bullet = f"- {sentence}"
        if total + len(bullet) + 1 > max_chars:
            continue
        selected.append(bullet)
        total += len(bullet) + 1
        if total >= max_chars:
            break
    return "\n".join(selected)[:max_chars]


# ---------------------------------------------------------------------------
# Ingestor
# ---------------------------------------------------------------------------

class DocumentIngestor:
    def __init__(self, config: RagV2Config) -> None:
        self._config = config
        _progress(
            "initializing sentence-transformer embedder "
            f"model={config.embedding_model} device={config.embedding_device}"
        )
        self._embedder = build_embedder(config)
        _progress(
            "sentence-transformer embedder ready; "
            f"Pinecone index={config.pinecone_index_name} namespace={config.pinecone_namespace}"
        )
        self._store = PineconeStore(config)

        # Vision client — None means disabled (uses pdfplumber text only)
        _progress(
            "initializing vision client "
            f"provider={config.vision_provider} model={config.vision_model}"
        )
        self._vision = build_vision_client(
            provider=config.vision_provider,
            api_key=config.gemini_api_key,
            api_keys=config.gemini_api_keys,
            model=config.vision_model,
            rpm_per_key=config.gemini_rpm_per_key,
            rpd_per_key=config.gemini_rpd_per_key,
            base_url=config.ollama_base_url,
            device=config.qwen_device,
            max_new_tokens=config.qwen_max_new_tokens,
            min_pixels=config.qwen_min_pixels,
            max_pixels=config.qwen_max_pixels,
        )
        if self._vision is not None:
            _progress(
                "vision extraction enabled: "
                f"provider={config.vision_provider} model={config.vision_model}"
            )
        else:
            _progress("vision extraction disabled (POWERMIND_V2_VISION_PROVIDER=none)")

    def ingest(self, path: Path, metadata: dict[str, Any] | None = None) -> str:
        return self.ingest_with_report(path, metadata)["document_id"]

    def ingest_with_report(self, path: Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        total_start = perf_counter()
        document_id = _doc_id(path)
        _progress(f"{path.name}: ingest started document_id={document_id}")
        text_pages = _extract_text_pages(path)
        _progress(f"{path.name}: normal text extraction complete pages_with_text={len(text_pages)}")
        vlm_pages, vlm_failures = _extract_vlm_pages(path, self._vision, self._config.vision_dpi)
        _progress(f"{path.name}: VLM extraction complete pages_with_vlm_text={len(vlm_pages)}")
        if vlm_failures:
            _progress(f"{path.name}: VLM failures tracked count={len(vlm_failures)} details={vlm_failures}")
        page_sets = [
            {
                "channel": "text",
                "embedding_source": "sentence_transformer",
                "pages": text_pages,
            },
            {
                "channel": "vlm",
                "embedding_source": f"{self._config.vision_provider}_vlm_page_extraction_sentence_transformer",
                "pages": vlm_pages,
            },
        ]
        if not text_pages and not vlm_pages:
            raise RuntimeError(f"No text extracted from {path}")

        vectors: list[dict[str, Any]] = []
        channel_counts = {"text": 0, "vlm": 0}
        chunk_index = 0
        raptor_seed_nodes: list[dict[str, Any]] = []
        for page_set in page_sets:
            channel = page_set["channel"]
            embedding_source = page_set["embedding_source"]
            _progress(
                f"{path.name}: channel={channel} chunking {len(page_set['pages'])} pages "
                f"source={embedding_source}"
            )
            for page_number, page_text in page_set["pages"]:
                chunks = _chunk(page_text, self._config.chunk_size, self._config.chunk_overlap)
                if not chunks:
                    _progress(f"{path.name}: channel={channel} page={page_number} produced 0 chunks")
                    continue
                embed_start = perf_counter()
                _progress(
                    f"{path.name}: channel={channel} page={page_number} "
                    f"embedding {len(chunks)} chunks"
                )
                embeddings = self._embedder.embed(chunks)
                _progress(
                    f"{path.name}: channel={channel} page={page_number} "
                    f"embedded {len(chunks)} chunks in {perf_counter() - embed_start:.1f}s"
                )
                for channel_chunk_index, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
                    vector_id = (
                        f"{document_id}:p{page_number}:"
                        f"{channel}:c{channel_chunk_index}"
                    )
                    vectors.append({
                        "id": vector_id,
                        "values": emb.tolist(),
                        "metadata": {
                            "document_id": document_id,
                            "filename": path.name,
                            "index_level": 0,
                            "page_number": page_number,
                            "chunk_index": chunk_index,
                            "channel_chunk_index": channel_chunk_index,
                            "embedding_channel": channel,
                            "embedding_source": embedding_source,
                            "text": chunk_text,
                            **(metadata or {}),
                        },
                    })
                    raptor_seed_nodes.append({
                        "id": vector_id,
                        "text": chunk_text,
                        "page_number": page_number,
                        "embedding_channel": channel,
                    })
                    channel_counts[channel] += 1
                    chunk_index += 1

        if not vectors:
            raise RuntimeError(f"Chunking produced 0 chunks for {path}")

        raptor_counts: dict[int, int] = {}
        if self._config.raptor_enabled:
            raptor_vectors, raptor_counts = self._build_raptor_vectors(
                document_id=document_id,
                path=path,
                seed_nodes=raptor_seed_nodes,
                metadata=metadata or {},
            )
            vectors.extend(raptor_vectors)
            _progress(
                f"{path.name}: RAPTOR complete summary_vectors={sum(raptor_counts.values())} "
                f"levels={raptor_counts}"
            )
        else:
            _progress(f"{path.name}: RAPTOR disabled")

        _progress(
            f"{path.name}: upserting {len(vectors)} vectors to Pinecone "
            f"index={self._config.pinecone_index_name} namespace={self._config.pinecone_namespace}"
        )
        upsert_start = perf_counter()
        self._store.upsert(vectors)
        _progress(f"{path.name}: Pinecone upsert complete in {perf_counter() - upsert_start:.1f}s")
        logger.info(
            "Ingested %d Pinecone vectors from '%s' (doc_id=%s, text=%d, vlm=%d)",
            len(vectors), path.name, document_id, channel_counts["text"], channel_counts["vlm"],
        )
        _progress(
            f"{path.name}: ingest finished in {perf_counter() - total_start:.1f}s "
            f"text_vectors={channel_counts['text']} vlm_vectors={channel_counts['vlm']} "
            f"total_vectors={len(vectors)}"
        )
        return {
            "document_id": document_id,
            "filename": path.name,
            "source_path": str(path),
            "pinecone_index": self._config.pinecone_index_name,
            "pinecone_namespace": self._config.pinecone_namespace,
            "embedding_model": self._config.embedding_model,
            "vision_model": self._config.vision_model if self._vision is not None else None,
            "vectors_upserted": len(vectors),
            "text_vectors": channel_counts["text"],
            "vlm_vectors": channel_counts["vlm"],
            "vlm_pages_succeeded": len(vlm_pages),
            "vlm_pages_failed": len(vlm_failures),
            "vlm_failures": vlm_failures,
            "gemini_report": self.visual_report(),
            "raptor_vectors": sum(raptor_counts.values()),
            "raptor_level_counts": raptor_counts,
            "vector_id_prefix": document_id,
            "metadata_filters": {
                "document_id": document_id,
                "embedding_channel": ["text", "vlm"],
                "index_level": [0, 1, 2, 3, 4],
            },
        }

    def visual_report(self) -> dict[str, Any]:
        if self._vision is not None and hasattr(self._vision, "report"):
            return self._vision.report()
        return {}

    def _build_raptor_vectors(
        self,
        *,
        document_id: str,
        path: Path,
        seed_nodes: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[int, int]]:
        vectors: list[dict[str, Any]] = []
        level_counts: dict[int, int] = {}
        current_nodes = seed_nodes
        max_level = min(4, self._config.raptor_max_level)

        for level in range(1, max_level + 1):
            if len(current_nodes) <= 1:
                break
            batches = _batched(current_nodes, max(2, self._config.raptor_batch_size))
            next_nodes: list[dict[str, Any]] = []
            _progress(f"{path.name}: RAPTOR level={level} summarizing {len(batches)} batches")
            for batch_index, batch in enumerate(batches):
                child_texts = [node["text"] for node in batch if node.get("text")]
                if not child_texts:
                    continue
                summary = _local_raptor_summary(
                    child_texts,
                    self._config.raptor_summary_max_chars,
                ).strip()
                if not summary:
                    summary = " ".join(child_texts)[: self._config.raptor_summary_max_chars]

                vector_id = f"{document_id}:raptor:l{level}:n{batch_index}"
                emb = self._embedder.embed([summary])[0]
                child_ids = [str(node["id"]) for node in batch]
                page_numbers = sorted({
                    int(node["page_number"])
                    for node in batch
                    if node.get("page_number") is not None
                })
                vectors.append({
                    "id": vector_id,
                    "values": emb.tolist(),
                    "metadata": {
                        "document_id": document_id,
                        "filename": path.name,
                        "index_level": level,
                        "embedding_channel": "raptor_summary",
                        "embedding_source": "raptor_local_extractive_e5_small",
                        "summary_level": level,
                        "summary_batch_index": batch_index,
                        "child_count": len(batch),
                        "child_node_ids": child_ids[:50],
                        "page_numbers": [str(page) for page in page_numbers],
                        "text": summary,
                        **metadata,
                    },
                })
                next_nodes.append({
                    "id": vector_id,
                    "text": summary,
                    "page_number": page_numbers[0] if page_numbers else None,
                    "embedding_channel": "raptor_summary",
                })
                _progress(
                    f"{path.name}: RAPTOR level={level} batch={batch_index + 1}/{len(batches)} "
                    f"children={len(batch)} summary_chars={len(summary)}"
                )
            level_counts[level] = len(next_nodes)
            current_nodes = next_nodes
            if len(current_nodes) <= 1:
                break
        return vectors, level_counts

    def delete(self, document_id: str) -> None:
        self._store.delete_document(document_id)
