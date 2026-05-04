"""
PowerMind Backend — App Factory
--------------------------------
Mounts all routers and handles lifespan events (DB init).
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app.routers import chat, chat_v2, history, ingest, ingest_v2, rag


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run startup / shutdown tasks."""
    # Initialise SQLite tables on startup
    init_db()
    yield
    # (add cleanup here if needed)


app = FastAPI(
    title="PowerMind API",
    description=(
        "Multimodal RAG chatbot backend.\n\n"
        "- **/rag** — original FAISS-based pipeline (v1)\n"
        "- **/chat/v2** — LangGraph + Pinecone multi-turn chat (v2)\n"
        "- **/ingest/v2** — Pinecone-backed document ingestion (v2)\n"
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow all origins in development; tighten in production
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
# v1 routers (FAISS-based, backward-compatible)
app.include_router(chat.router)
app.include_router(history.router)
app.include_router(ingest.router)
app.include_router(rag.router)

# v2 routers (LangGraph + Pinecone)
app.include_router(chat_v2.router)
app.include_router(ingest_v2.router)


@app.get("/", tags=["health"])
async def root():
    return {"status": "ok", "service": "PowerMind API", "version": "0.2.0"}


@app.get("/health", tags=["health"])
async def health():
    return {"status": "healthy"}
