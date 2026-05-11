"""
Pramana Backend - App Factory
-----------------------------
Mounts compatibility API routers and handles startup database init.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app.routers import chat, history, ingest, rag


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_db()
    yield


app = FastAPI(
    title="Pramana API",
    description=(
        "Pramana document QA backend.\n\n"
        "- **/rag** - original compatibility route\n"
        "- Production RAG lives in service/src/powermind_rag."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(history.router)
app.include_router(ingest.router)
app.include_router(rag.router)


@app.get("/", tags=["health"])
async def root():
    return {"status": "ok", "service": "Pramana API", "version": "1.0.0"}


@app.get("/health", tags=["health"])
async def health():
    return {"status": "healthy"}
