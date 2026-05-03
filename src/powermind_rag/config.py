from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

@dataclass(frozen=True)
class RAGConfig:
    storage_dir: Path = Path("storage")
    qwen_model_path: Path = Path("Qwen")
    dense_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    colpali_model_name: str = "vidore/colpali-v1.2"
    device: str = "cuda"
    cuda_arch: str = "sm_120"
    visual_top_k: int = 5
    text_top_k: int = 12
    final_top_k: int = 16
    relevance_threshold: float = 0.4
    rrf_k: int = 60
    mistral_api_key: str | None = None
    mistral_ocr_model: str = "mistral-ocr-latest"
    groq_api_key: str | None = None
    groq_relevance_model: str = "llama-3.1-8b-instant"
    local_only: bool = False

    @classmethod
    def from_env(cls) -> "RAGConfig":
        load_dotenv()
        local_only = os.getenv("POWERMIND_LOCAL_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}
        return cls(
            storage_dir=Path(os.getenv("POWERMIND_STORAGE_DIR", "storage")),
            qwen_model_path=Path(os.getenv("QWEN_MODEL_PATH", "Qwen")),
            dense_embedding_model=os.getenv(
                "POWERMIND_DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            colpali_model_name=os.getenv("POWERMIND_COLPALI_MODEL", "vidore/colpali-v1.2"),
            device=os.getenv("POWERMIND_DEVICE", "cuda"),
            cuda_arch=os.getenv("POWERMIND_CUDA_ARCH", "sm_120"),
            mistral_api_key=os.getenv("MISTRAL_API_KEY"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            groq_relevance_model=os.getenv("GROQ_RELEVANCE_MODEL", "llama-3.1-8b-instant"),
            local_only=local_only,
        )
