"""
Ingest Router v2
----------------
Handles file upload and ingestion into the Pinecone-backed v2 pipeline.

POST   /ingest/v2/file            → upload + ingest file (202 Accepted)
GET    /ingest/v2/files           → list ingested files
GET    /ingest/v2/files/{id}      → get status of a specific file
POST   /ingest/v2/files/{id}/retry → re-run ingest for a pending/failed file
DELETE /ingest/v2/files/{id}      → delete file vectors from Pinecone + DB record
GET    /ingest/v2/store/stats     → Pinecone index stats
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.dependencies import get_db, get_upload_dir
from app.models import IngestedFile
from app.schemas import IngestedFileResponse, IngestListResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest/v2", tags=["ingest-v2"])

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


def _progress(message: str) -> None:
    logger.info(message)
    print(f"[ingest-v2] {message}", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_ingestor():
    from app.rag_v2.config import RagV2Config    # noqa: PLC0415
    from app.rag_v2.ingestor import DocumentIngestor  # noqa: PLC0415
    return DocumentIngestor(RagV2Config.from_env())


def _get_data_dir() -> Path:
    data_dir = Path(__file__).resolve().parents[2] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _supported_data_files(data_dir: Path) -> list[Path]:
    return sorted(
        path for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    )


def _run_ingest_sync(record_id: str, stored_path: Path) -> None:
    """
    Synchronous ingest worker — runs in a BackgroundTasks thread.
    Opens its own DB session so it's independent of the request session.
    """
    db: Session = SessionLocal()
    try:
        record = db.get(IngestedFile, record_id)
        if not record:
            logger.error("Ingest: record %s not found in DB", record_id)
            return

        try:
            logger.info("v2 ingest starting: %s", stored_path.name)
            ingestor = _get_ingestor()
            report = ingestor.ingest_with_report(
                stored_path,
                {"original_filename": record.original_filename},
            )
            record.document_id = report["document_id"]
            record.status = "completed"
            record.error = None
            logger.info("v2 ingest complete: %s -> %s", stored_path.name, report["document_id"])

        except Exception as exc:
            logger.exception("v2 ingest failed for %s: %s", stored_path.name, exc)
            record.status = "failed"
            record.error = str(exc)[:1024]

        db.commit()

    except Exception as exc:
        logger.exception("v2 ingest DB error for record %s: %s", record_id, exc)
    finally:
        db.close()


def _ingest_data_folder_sync() -> dict:
    """
    Ingest every supported document in backend/data into Pinecone only.

    PDF documents produce two vector channels in the same Pinecone index:
    - embedding_channel=text for normal text extraction
    - embedding_channel=vlm for Qwen VLM page extraction
    """
    from app.rag_v2.config import RagV2Config  # noqa: PLC0415
    from app.rag_v2.vector_store import PineconeStore  # noqa: PLC0415

    config = RagV2Config.from_env()
    data_dir = _get_data_dir()
    files = _supported_data_files(data_dir)
    _progress(f"data-folder ingest requested data_dir={data_dir}")
    _progress(f"found {len(files)} supported documents: {[path.name for path in files]}")
    if not files:
        return {
            "status": "empty",
            "message": f"No supported documents found in {data_dir}",
            "data_dir": str(data_dir),
            "supported_extensions": sorted(ALLOWED_EXTENSIONS),
            "results": [],
        }

    _progress(
        "Pinecone target "
        f"index={config.pinecone_index_name} namespace={config.pinecone_namespace} "
        f"dimension={config.embedding_dimension}"
    )
    _progress("creating ingestor and loading local models")
    ingestor = _get_ingestor()
    _progress("ingestor ready")
    db: Session = SessionLocal()
    results: list[dict] = []
    try:
        for file_index, path in enumerate(files, start=1):
            _progress(f"file {file_index}/{len(files)} started: {path.name}")
            record = (
                db.query(IngestedFile)
                .filter(IngestedFile.stored_path == str(path))
                .first()
            )
            if record is None:
                record = IngestedFile(
                    original_filename=path.name,
                    stored_path=str(path),
                    status="pending",
                )
                db.add(record)
                db.flush()
            else:
                record.original_filename = path.name
                record.status = "pending"
                record.error = None

            try:
                report = ingestor.ingest_with_report(
                    path,
                    {
                        "original_filename": path.name,
                        "source_folder": str(data_dir),
                    },
                )
                record.document_id = report["document_id"]
                record.status = "completed"
                record.error = None
                results.append({"status": "completed", "db_record_id": record.id, **report})
                _progress(
                    f"file {file_index}/{len(files)} completed: {path.name} "
                    f"text_vectors={report['text_vectors']} vlm_vectors={report['vlm_vectors']} "
                    f"vlm_pages_succeeded={report.get('vlm_pages_succeeded', 0)} "
                    f"vlm_pages_failed={report.get('vlm_pages_failed', 0)} "
                    f"total_vectors={report['vectors_upserted']}"
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("data-folder ingest failed for %s: %s", path.name, exc)
                _progress(f"file {file_index}/{len(files)} failed: {path.name} error={exc}")
                record.status = "failed"
                record.error = str(exc)[:1024]
                results.append({
                    "status": "failed",
                    "db_record_id": record.id,
                    "filename": path.name,
                    "source_path": str(path),
                    "error": str(exc),
                })
            db.commit()
            _progress(f"database status saved for {path.name}: {record.status}")
    finally:
        db.close()

    _progress("fetching Pinecone index stats")
    try:
        store_stats = PineconeStore(config).describe()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not fetch Pinecone stats after ingest: %s", exc)
        _progress(f"could not fetch Pinecone stats after ingest: {exc}")
        store_stats = {"error": str(exc)}
    _progress(
        "data-folder ingest finished "
        f"completed={sum(1 for r in results if r['status'] == 'completed')}/{len(files)}"
    )
    completed = [r for r in results if r["status"] == "completed"]
    final_visual_report = completed[-1].get("gemini_report", {}) if completed else {}
    final_summary = {
        "documents_completed": sum(1 for r in results if r["status"] == "completed"),
        "documents_failed": sum(1 for r in results if r["status"] == "failed"),
        "text_vectors": sum(int(r.get("text_vectors", 0)) for r in completed),
        "vlm_vectors": sum(int(r.get("vlm_vectors", 0)) for r in completed),
        "raptor_vectors": sum(int(r.get("raptor_vectors", 0)) for r in completed),
        "vlm_pages_succeeded": sum(int(r.get("vlm_pages_succeeded", 0)) for r in completed),
        "vlm_pages_failed": sum(int(r.get("vlm_pages_failed", 0)) for r in completed),
        "gemini": final_visual_report,
    }
    _progress(f"FINAL INGEST RESULT: {final_summary}")
    return {
        "status": "completed" if all(r["status"] == "completed" for r in results) else "partial",
        "data_dir": str(data_dir),
        "documents_found": len(files),
        "documents_completed": sum(1 for r in results if r["status"] == "completed"),
        "final_summary": final_summary,
        "pinecone_storage": {
            "index": config.pinecone_index_name,
            "namespace": config.pinecone_namespace,
            "dimension": config.embedding_dimension,
            "stats": store_stats,
            "inspect_by_metadata": {
                "embedding_channel": ["text", "vlm"],
                "source_folder": str(data_dir),
            },
        },
        "results": results,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/file", response_model=IngestedFileResponse, status_code=202)
async def ingest_file_v2(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Upload a document and ingest it into Pinecone (v2 pipeline).
    Returns 202 immediately; track status via GET /ingest/v2/files/{id}.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 50 MB limit")

    upload_dir: Path = get_upload_dir()
    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    stored_path = upload_dir / unique_name
    stored_path.write_bytes(content)

    record = IngestedFile(
        original_filename=file.filename or unique_name,
        stored_path=str(stored_path),
        status="pending",
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    # BackgroundTasks runs after the response is sent — reliable, lifecycle-managed
    background_tasks.add_task(_run_ingest_sync, record.id, stored_path)

    return IngestedFileResponse.model_validate(record)


@router.post("/data-folder")
async def ingest_data_folder_v2():
    """
    Ingest every supported document currently present in backend/data.

    Stores only Pinecone vectors. PDF documents get both normal sentence
    transformer text vectors and Qwen VLM-extracted page-content vectors.
    """
    try:
        return await run_in_threadpool(_ingest_data_folder_sync)
    except Exception as exc:
        logger.exception("data-folder ingest crashed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))


@router.post("/files/{file_id}/retry", response_model=IngestedFileResponse)
def retry_ingest_v2(
    file_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Re-run ingestion for a pending or failed file."""
    record = db.get(IngestedFile, file_id)
    if not record:
        raise HTTPException(status_code=404, detail="File record not found")
    if record.status == "completed":
        raise HTTPException(status_code=409, detail="File already ingested successfully")

    stored_path = Path(record.stored_path)
    if not stored_path.exists():
        raise HTTPException(status_code=410, detail="Uploaded file no longer on disk")

    record.status = "pending"
    record.error = None
    db.commit()
    db.refresh(record)

    background_tasks.add_task(_run_ingest_sync, record.id, stored_path)
    return IngestedFileResponse.model_validate(record)


@router.get("/files", response_model=IngestListResponse)
def list_files_v2(db: Session = Depends(get_db)):
    """List all files ingested via the v2 pipeline."""
    files = (
        db.query(IngestedFile)
        .filter(IngestedFile.stored_path.isnot(None))
        .order_by(IngestedFile.created_at.desc())
        .all()
    )
    return IngestListResponse(
        files=[IngestedFileResponse.model_validate(f) for f in files],
        total=len(files),
    )


@router.get("/files/{file_id}", response_model=IngestedFileResponse)
def get_file_v2(file_id: str, db: Session = Depends(get_db)):
    """Get the ingestion status of a specific file."""
    record = db.get(IngestedFile, file_id)
    if not record:
        raise HTTPException(status_code=404, detail="File record not found")
    return IngestedFileResponse.model_validate(record)


@router.delete("/files/{file_id}", status_code=204)
def delete_file_v2(file_id: str, db: Session = Depends(get_db)):
    """Remove file vectors from Pinecone and delete the DB record."""
    record = db.get(IngestedFile, file_id)
    if not record:
        raise HTTPException(status_code=404, detail="File record not found")

    if record.document_id:
        try:
            _get_ingestor().delete(record.document_id)
        except Exception as exc:
            logger.warning("Could not delete Pinecone vectors for %s: %s", file_id, exc)

    db.delete(record)
    db.commit()


@router.get("/store/stats")
def store_stats():
    """Return Pinecone index stats."""
    try:
        from app.rag_v2.config import RagV2Config    # noqa: PLC0415
        from app.rag_v2.vector_store import PineconeStore  # noqa: PLC0415
        return PineconeStore(RagV2Config.from_env()).describe()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

