from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy.orm import Session

from app.database import SessionLocal


# ---------------------------------------------------------------------------
# DB session dependency
# ---------------------------------------------------------------------------

def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a SQLAlchemy DB session and closes it after."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# RAG pipeline dependency (singleton, same pattern as original main.py)
# ---------------------------------------------------------------------------

_pipeline_lock = asyncio.Lock()


@lru_cache
def _service_root() -> Path:
    return Path(__file__).resolve().parents[2] / "service"


def _load_service_env() -> None:
    service_root = _service_root()
    service_env = service_root / ".env"
    if service_env.exists():
        load_dotenv(service_env, override=False)
    os.environ.setdefault("POWERMIND_STORAGE_DIR", str(service_root / "storage"))


@lru_cache
def get_pipeline():  # type: ignore[return]
    """Return a singleton MultimodalRAGPipeline. Loaded lazily on first use."""
    from powermind_rag import MultimodalRAGPipeline  # noqa: PLC0415
    from powermind_rag.config import RAGConfig  # noqa: PLC0415

    _load_service_env()
    return MultimodalRAGPipeline(RAGConfig.from_env())


def get_pipeline_lock() -> asyncio.Lock:
    return _pipeline_lock


def get_upload_dir() -> Path:
    """Directory where uploaded files are saved before ingestion."""
    service_root = _service_root()
    upload_dir = service_root / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir
