from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag_v2.config import RagV2Config


def progress(message: str) -> None:
    print(f"[pinecone-export] {message}", flush=True)


def response_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def vector_to_record(vector: Any) -> dict[str, Any]:
    if isinstance(vector, dict):
        return {
            "id": vector.get("id", ""),
            "values": list(vector.get("values") or []),
            "metadata": dict(vector.get("metadata") or {}),
            "sparse_values": vector.get("sparse_values"),
        }
    return {
        "id": getattr(vector, "id", ""),
        "values": list(getattr(vector, "values", None) or []),
        "metadata": dict(getattr(vector, "metadata", None) or {}),
        "sparse_values": getattr(vector, "sparse_values", None),
    }


def iter_vector_ids(index: Any, namespace: str, page_size: int) -> list[str]:
    ids: list[str] = []
    token: str | None = None
    page = 0
    while True:
        page += 1
        resp = index.list_paginated(
            namespace=namespace,
            limit=page_size,
            pagination_token=token,
        )
        vectors = response_get(resp, "vectors", []) or []
        page_ids = [
            response_get(item, "id", "")
            for item in vectors
            if response_get(item, "id", "")
        ]
        ids.extend(page_ids)
        progress(f"listed page={page} ids={len(page_ids)} total_ids={len(ids)}")

        pagination = response_get(resp, "pagination", None)
        token = response_get(pagination, "next", None) if pagination else None
        if not token:
            break
    return ids


def fetch_batch(index: Any, namespace: str, ids: list[str]) -> list[dict[str, Any]]:
    resp = index.fetch(ids=ids, namespace=namespace)
    vectors = response_get(resp, "vectors", {}) or {}
    if isinstance(vectors, dict):
        return [vector_to_record(vector) for vector in vectors.values()]
    return [vector_to_record(vector) for vector in vectors]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export Pinecone vectors with original values and metadata to local JSONL.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(BACKEND_ROOT / "pinecone_exports"),
        help="Directory where vectors.jsonl and manifest.json will be written.",
    )
    parser.add_argument(
        "--list-page-size",
        type=int,
        default=100,
        help="Number of vector IDs to request per Pinecone list page.",
    )
    parser.add_argument(
        "--fetch-batch-size",
        type=int,
        default=100,
        help="Number of vectors to fetch per Pinecone fetch call.",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="Override namespace. Defaults to PINECONE_NAMESPACE from .env.",
    )
    parser.add_argument(
        "--index-host",
        default=None,
        help="Explicit Pinecone index host. If set, skips api.pinecone.io lookup.",
    )
    args = parser.parse_args()

    cfg = RagV2Config.from_env()
    namespace = args.namespace or cfg.pinecone_namespace
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = out_dir / "vectors.jsonl"
    manifest_path = out_dir / "manifest.json"
    if vectors_path.exists():
        vectors_path.unlink()

    try:
        from pinecone import Pinecone
    except ImportError as exc:
        raise ImportError("Run: pip install pinecone") from exc

    if not cfg.pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is not set in .env")

    progress(f"connecting index={cfg.pinecone_index_name} namespace={namespace}")
    pc = Pinecone(api_key=cfg.pinecone_api_key)
    env_index_host = os.getenv("PINECONE_INDEX_HOST", "").strip() if args.index_host is None else ""
    index_host = args.index_host or env_index_host
    if index_host:
        progress(f"using explicit index host={index_host}")
        index = pc.Index(host=index_host)
    else:
        try:
            index = pc.Index(cfg.pinecone_index_name)
        except Exception as exc:
            message = str(exc)
            if "api.pinecone.io" in message or "NameResolutionError" in message:
                raise RuntimeError(
                    "Pinecone control-plane DNS is unavailable, so the exporter cannot resolve the index host. "
                    "Set PINECONE_INDEX_HOST to the index host from the Pinecone console, or rerun with --index-host."
                ) from exc
            raise

    stats = index.describe_index_stats()
    ids = iter_vector_ids(index=index, namespace=namespace, page_size=args.list_page_size)
    progress(f"fetching vectors total_ids={len(ids)} batch_size={args.fetch_batch_size}")

    exported = 0
    dimensions: set[int] = set()
    for start in range(0, len(ids), args.fetch_batch_size):
        batch_ids = ids[start:start + args.fetch_batch_size]
        records = fetch_batch(index=index, namespace=namespace, ids=batch_ids)
        for record in records:
            dimensions.add(len(record.get("values") or []))
        write_jsonl(vectors_path, records)
        exported += len(records)
        progress(f"fetched batch={start // args.fetch_batch_size + 1} exported={exported}/{len(ids)}")

    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "index": cfg.pinecone_index_name,
        "namespace": namespace,
        "vectors_file": str(vectors_path),
        "vector_count": exported,
        "listed_id_count": len(ids),
        "dimensions": sorted(dimensions),
        "index_stats": {
            "total_vector_count": getattr(stats, "total_vector_count", None),
            "dimension": getattr(stats, "dimension", None),
            "namespaces": {
                name: getattr(ns, "vector_count", None)
                for name, ns in (getattr(stats, "namespaces", None) or {}).items()
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    progress(f"wrote vectors to {vectors_path}")
    progress(f"wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
