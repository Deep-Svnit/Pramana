from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from dotenv import load_dotenv

from app.rag_v2.config import RagV2Config
from app.rag_v2.ingestor import DocumentIngestor
from app.rag_v2.vector_store import PineconeStore


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def progress(message: str) -> None:
    print(f"[manual-ingest] {message}", flush=True)


def main() -> int:
    chat_id = sys.argv[1] if len(sys.argv) > 1 else ""
    env_path = BACKEND_ROOT / ".env"
    load_dotenv(env_path, override=True)

    config = RagV2Config.from_env()
    data_dir = BACKEND_ROOT / "data"
    files = sorted(
        path for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    progress(f"chat_id={chat_id or '[none]'}")
    progress(f"data_dir={data_dir}")
    progress(f"files_found={len(files)} names={[path.name for path in files]}")
    progress(f"embedding_device={config.embedding_device}")
    progress(f"qwen_device={config.qwen_device}")
    progress(f"pinecone_index={config.pinecone_index_name} namespace={config.pinecone_namespace}")

    if not files:
        progress("no supported files found; exiting")
        return 0

    start = perf_counter()
    ingestor = DocumentIngestor(config)
    reports = []
    for index, path in enumerate(files, start=1):
        progress(f"file {index}/{len(files)} starting {path.name}")
        report = ingestor.ingest_with_report(
            path,
            {
                "chat_id": chat_id,
                "source_folder": str(data_dir),
                "original_filename": path.name,
                "manual_restart": True,
            },
        )
        reports.append(report)
        progress(
            f"file {index}/{len(files)} done {path.name} "
            f"text_vectors={report['text_vectors']} "
            f"vlm_vectors={report['vlm_vectors']} "
            f"total_vectors={report['vectors_upserted']}"
        )

    progress("fetching pinecone stats")
    stats = PineconeStore(config).describe()
    progress(f"pinecone_stats={stats}")
    progress(f"all done elapsed={perf_counter() - start:.1f}s files={len(reports)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
