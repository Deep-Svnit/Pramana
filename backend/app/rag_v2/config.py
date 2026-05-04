"""
Configuration for RAG v2 (local vector cache or Pinecone + LangGraph + Groq).
Reads from backend/.env — path resolved relative to this file.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _load_env() -> None:
    """
    Load backend/.env — the single source of truth for all backend config.
    config.py is at: backend/app/rag_v2/config.py
    parents[2]     = backend/
    """
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)


def _pinecone_index_name() -> str:
    raw = os.getenv("PINECONE_INDEX_NAME", "powermind-rag")
    name = re.sub(r"[^a-z0-9-]+", "-", raw.strip().lower()).strip("-")
    return name or "powermind-rag"


@dataclass(frozen=True)
class RagV2Config:
    # ── Pinecone ─────────────────────────────────────────────────────────────
    pinecone_api_key: str = ""
    pinecone_index_name: str = "powermind-rag"
    pinecone_namespace: str = "default"
    embedding_dimension: int = 384   # matches all-MiniLM-L6-v2
    vector_store_backend: str = "auto"  # auto | local | pinecone
    local_vector_store_path: str = ""

    # ── Embeddings (local sentence-transformers) ──────────────────────────────
    # Uses its OWN env var (POWERMIND_V2_EMBEDDING_MODEL) so it is NOT
    # affected by POWERMIND_DENSE_MODEL="E5_Small" set for v1.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # ── LLM — Groq ────────────────────────────────────────────────────────────
    generation_provider: str = "groq"
    groq_api_key: str = ""
    groq_chat_model: str = "llama-3.3-70b-versatile"
    nvidia_api_key: str = ""
    nvidia_chat_model: str = "meta/llama-3.2-90b-vision-instruct"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    gemini_chat_model: str = "gemini-2.5-flash"
    openrouter_api_key: str = ""
    openrouter_chat_model: str = "anthropic/claude-sonnet-4"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = 8
    score_threshold: float = 0.35

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Vision extraction (multimodal PDF) ───────────────────────────────────
    # provider: "gemini" | "qwen" | "ollama" | "none"
    vision_provider: str = "none"
    vision_model: str = "gemini-2.5-flash"
    gemini_api_key: str = ""              # kept for backward-compatible envs
    gemini_api_keys: tuple[str, ...] = ()
    gemini_rpm_per_key: int = 5
    gemini_rpd_per_key: int = 20
    ollama_base_url: str = "http://localhost:11434"
    vision_dpi: int = 150  # page render resolution (higher = better quality, slower)
    qwen_device: str = "auto"
    qwen_max_new_tokens: int = 2048
    qwen_min_pixels: int = 256 * 28 * 28
    qwen_max_pixels: int = 1536 * 28 * 28

    # CRAG / hallucination detection
    crag_enabled: bool = True
    lettuce_model_path: str = "KRLabsOrg/lettucedect-base-modernbert-en-v1"
    lettuce_max_length: int = 4096
    lettuce_threshold: float = 0.50
    crag_page_top_k: int = 5

    # RAPTOR / hierarchical indexing
    raptor_enabled: bool = True
    raptor_batch_size: int = 8
    raptor_max_level: int = 4
    raptor_summary_max_chars: int = 2400

    @classmethod
    def from_env(cls) -> "RagV2Config":
        _load_env()
        return cls(
            pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
            pinecone_index_name=_pinecone_index_name(),
            pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "default"),
            embedding_dimension=int(os.getenv("PINECONE_EMBEDDING_DIM", "384")),
            vector_store_backend=os.getenv("POWERMIND_V2_VECTOR_STORE", "auto").strip().lower(),
            local_vector_store_path=os.getenv(
                "POWERMIND_V2_LOCAL_VECTOR_PATH",
                str(Path(__file__).resolve().parents[2] / "pinecone_exports" / "vectors.jsonl"),
            ),
            # Separate env var for v2 so it doesn't inherit E5_Small from v1
            embedding_model=os.getenv(
                "POWERMIND_V2_EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            embedding_device=os.getenv("POWERMIND_DEVICE", "cpu"),
            generation_provider=os.getenv("POWERMIND_GENERATION_PROVIDER", "groq").strip().lower(),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            groq_chat_model=os.getenv("GROQ_GENERATION_MODEL", "llama-3.3-70b-versatile"),
            nvidia_api_key=os.getenv("NVIDIA_KEY", ""),
            nvidia_chat_model=os.getenv(
                "NVIDIA_GENERATION_MODEL",
                "meta/llama-3.2-90b-vision-instruct",
            ),
            nvidia_base_url=os.getenv("NVIDIA_VLM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            gemini_chat_model=os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            openrouter_chat_model=os.getenv(
                "OPENROUTER_GENERATION_MODEL",
                "anthropic/claude-sonnet-4",
            ),
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            top_k=int(os.getenv("POWERMIND_V2_TOP_K", "8")),
            score_threshold=float(os.getenv("POWERMIND_V2_SCORE_THRESHOLD", "0.35")),
            chunk_size=int(os.getenv("POWERMIND_V2_CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("POWERMIND_V2_CHUNK_OVERLAP", "64")),
            vision_provider=os.getenv("POWERMIND_V2_VISION_PROVIDER", "gemini"),
            vision_model=os.getenv("POWERMIND_V2_VISION_MODEL", "gemini-2.5-flash"),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_api_keys=tuple(
                key for key in [
                    os.getenv("GEMINI_1", "").strip().strip('"'),
                    os.getenv("GEMINI_2", "").strip().strip('"'),
                    os.getenv("GEMINI_3", "").strip().strip('"'),
                    os.getenv("GEMINI_4", "").strip().strip('"'),
                    os.getenv("GEMINI_5", "").strip().strip('"'),
                    os.getenv("GEMINI_6", "").strip().strip('"'),
                ]
                if key
            )
            or ((os.getenv("GEMINI_API_KEY", "").strip().strip('"'),)
                if os.getenv("GEMINI_API_KEY", "").strip().strip('"') else ()),
            gemini_rpm_per_key=int(os.getenv("GEMINI_RPM_PER_KEY", "5")),
            gemini_rpd_per_key=int(os.getenv("GEMINI_RPD_PER_KEY", "20")),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            vision_dpi=int(os.getenv("POWERMIND_V2_VISION_DPI", "150")),
            qwen_device=os.getenv("POWERMIND_V2_QWEN_DEVICE", "auto"),
            qwen_max_new_tokens=int(os.getenv("POWERMIND_V2_QWEN_MAX_NEW_TOKENS", "2048")),
            qwen_min_pixels=int(os.getenv("POWERMIND_V2_QWEN_MIN_PIXELS", str(256 * 28 * 28))),
            qwen_max_pixels=int(os.getenv("POWERMIND_V2_QWEN_MAX_PIXELS", str(1536 * 28 * 28))),
            crag_enabled=os.getenv("POWERMIND_V2_CRAG_ENABLED", "true").lower()
            not in {"0", "false", "no"},
            lettuce_model_path=os.getenv(
                "LETTUCE_MODEL_PATH",
                "KRLabsOrg/lettucedect-base-modernbert-en-v1",
            ),
            lettuce_max_length=int(os.getenv("LETTUCE_MAX_LENGTH", "4096")),
            lettuce_threshold=float(os.getenv("LETTUCE_THRESHOLD", "0.50")),
            crag_page_top_k=int(os.getenv("POWERMIND_V2_CRAG_PAGE_TOP_K", "5")),
            raptor_enabled=os.getenv("POWERMIND_V2_RAPTOR_ENABLED", "true").lower()
            not in {"0", "false", "no"},
            raptor_batch_size=int(os.getenv("POWERMIND_V2_RAPTOR_BATCH_SIZE", "8")),
            raptor_max_level=min(4, int(os.getenv("POWERMIND_V2_RAPTOR_MAX_LEVEL", "4"))),
            raptor_summary_max_chars=int(os.getenv("POWERMIND_V2_RAPTOR_SUMMARY_MAX_CHARS", "2400")),
        )
