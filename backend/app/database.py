from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# ---------------------------------------------------------------------------
# DB path — mirrors service/storage dir used by the RAG pipeline
# ---------------------------------------------------------------------------

def _db_url() -> str:
    # DB lives in the backend/ directory (parent of app/)
    backend_dir = Path(__file__).resolve().parent.parent
    db_path = backend_dir / "powermind.db"
    return f"sqlite:///{db_path}"


engine = create_engine(
    _db_url(),
    connect_args={"check_same_thread": False},  # required for SQLite + FastAPI
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """Create all tables if they don't exist yet."""
    # Import models so Base knows about them before create_all
    from app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
