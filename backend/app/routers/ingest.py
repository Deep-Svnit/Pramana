"""
Ingest Router
-------------
Endpoints for file upload and ingestion status.

POST /ingest/file    → upload a file (multipart/form-data) and trigger RAG ingest
GET  /ingest/files   → list all ingested files and their status
"""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from app.dependencies import get_db, get_pipeline, get_upload_dir
from app.models import IngestedFile
from app.schemas import IngestedFileResponse, IngestListResponse

router = APIRouter(prefix="/ingest", tags=["ingest"])

# Allowed MIME types / extensions
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


@router.post("/file", response_model=IngestedFileResponse, status_code=202)
async def ingest_file(
    file: UploadFile,
    db: Session = Depends(get_db),
):
    """
    Upload a document file and ingest it into the RAG knowledge base.

    - File is saved to `service/data/uploads/` on disk.
    - A DB record is created immediately with `status=pending`.
    - Ingestion runs in the background; status updated to `completed` or `failed`.
    - Returns the file record immediately (202 Accepted).
    """
    # Validate extension
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Read content with size guard
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 50 MB limit")

    # Save to upload dir with unique name to avoid collisions
    upload_dir: Path = get_upload_dir()
    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    stored_path = upload_dir / unique_name
    stored_path.write_bytes(content)

    # Create DB record
    record = IngestedFile(
        original_filename=file.filename or unique_name,
        stored_path=str(stored_path),
        status="pending",
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    # Trigger ingestion in background (non-blocking)
    asyncio.create_task(
        _run_ingest(record.id, stored_path)
    )

    return IngestedFileResponse.model_validate(record)


async def _run_ingest(
    record_id: str,
    stored_path: Path,
) -> None:
    """Background task: run RAG ingest and update DB record status."""
    # We need a fresh DB session (the request session is already closed)
    from app.database import SessionLocal  # noqa: PLC0415
    from powermind_rag.document import document_id_for  # noqa: PLC0415

    db = SessionLocal()
    try:
        record = db.get(IngestedFile, record_id)
        if not record:
            return

        try:
            pipeline = get_pipeline()
            lock: asyncio.Lock = asyncio.Lock()
            async with lock:
                await run_in_threadpool(
                    pipeline.ingest_pdf,
                    stored_path,
                    {},
                )
            record.document_id = document_id_for(stored_path)
            record.status = "completed"
        except Exception as exc:  # noqa: BLE001
            record.status = "failed"
            record.error = str(exc)[:1024]

        db.commit()
    finally:
        db.close()


@router.get("/files", response_model=IngestListResponse)
def list_files(db: Session = Depends(get_db)):
    """List all ingested files and their ingestion status."""
    files = db.query(IngestedFile).order_by(IngestedFile.created_at.desc()).all()
    return IngestListResponse(
        files=[IngestedFileResponse.model_validate(f) for f in files],
        total=len(files),
    )


@router.get("/files/{file_id}", response_model=IngestedFileResponse)
def get_file(file_id: str, db: Session = Depends(get_db)):
    """Get status of a specific ingested file."""
    record = db.get(IngestedFile, file_id)
    if not record:
        raise HTTPException(status_code=404, detail="File record not found")
    return IngestedFileResponse.model_validate(record)
