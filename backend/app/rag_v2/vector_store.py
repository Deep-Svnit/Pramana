"""Vector store wrapper for RAG v2.

Supports two backends behind the same API:
- local JSONL cache exported from Pinecone
- Pinecone cloud index

The backend is selected by config.vector_store_backend:
- "local"   → always use the JSONL cache
- "pinecone" → always use Pinecone
- "auto"    → use local cache if it exists, otherwise Pinecone
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.rag_v2.config import RagV2Config

logger = logging.getLogger(__name__)


def _progress(message: str) -> None:
    logger.info(message)
    print(f"[vector-store-v2] {message}", flush=True)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, dict) else {}


def _value_matches(value: Any, condition: Any) -> bool:
    if isinstance(condition, dict):
        if "$eq" in condition:
            return value == condition["$eq"]
        if "$in" in condition:
            return value in set(condition["$in"])
        if "$ne" in condition:
            return value != condition["$ne"]
        if "$exists" in condition:
            return (value is not None) == bool(condition["$exists"])
    return value == condition


def _matches_filter(metadata: dict[str, Any], filter_spec: dict[str, Any] | None) -> bool:
    if not filter_spec:
        return True
    for key, condition in filter_spec.items():
        if key == "$and" and isinstance(condition, list):
            if not all(_matches_filter(metadata, item) for item in condition):
                return False
            continue
        if key == "$or" and isinstance(condition, list):
            if not any(_matches_filter(metadata, item) for item in condition):
                return False
            continue
        if not _value_matches(metadata.get(key), condition):
            return False
    return True


@dataclass
class _LocalVectorIndex:
    path: Path
    records: list[dict[str, Any]]
    ids: list[str]
    matrix: np.ndarray


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype("float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype("float32")


def _record_from_vector(vector: Any) -> dict[str, Any]:
    if isinstance(vector, dict):
        return {
            "id": str(vector.get("id", "")),
            "values": list(vector.get("values") or []),
            "metadata": _as_dict(vector.get("metadata")),
            "sparse_values": vector.get("sparse_values"),
        }
    return {
        "id": str(getattr(vector, "id", "")),
        "values": list(getattr(vector, "values", None) or []),
        "metadata": _as_dict(getattr(vector, "metadata", None)),
        "sparse_values": getattr(vector, "sparse_values", None),
    }


class PineconeStore:
    def __init__(self, config: RagV2Config) -> None:
        self._config = config
        self._index: Any | None = None
        self._local_index: _LocalVectorIndex | None = None
        self._backend = self._resolve_backend()

    def _resolve_backend(self) -> str:
        backend = (self._config.vector_store_backend or "auto").strip().lower()
        if backend not in {"auto", "local", "pinecone"}:
            raise ValueError(
                "Invalid POWERMIND_V2_VECTOR_STORE value. Use auto, local, or pinecone. "
                f"Current value: {backend!r}"
            )
        if backend == "auto" and self._local_path().exists():
            return "local"
        return "local" if backend == "local" else "pinecone"

    def _local_path(self) -> Path:
        return Path(self._config.local_vector_store_path)

    def _ensure_local_index(self) -> _LocalVectorIndex:
        if self._local_index is not None:
            return self._local_index

        path = self._local_path()
        if not path.exists():
            raise FileNotFoundError(
                "Local vector store not found. Run scripts/export_pinecone_vectors.py first "
                f"or set POWERMIND_V2_VECTOR_STORE=pinecone. Expected file: {path}"
            )

        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record["id"] = str(record.get("id", ""))
                record["values"] = list(record.get("values") or [])
                record["metadata"] = _as_dict(record.get("metadata"))
                records.append(record)

        ids = [record["id"] for record in records]
        matrix = np.asarray([record["values"] for record in records], dtype="float32") if records else np.empty((0, 0), dtype="float32")
        matrix = _normalize_rows(matrix)
        self._local_index = _LocalVectorIndex(path=path, records=records, ids=ids, matrix=matrix)
        _progress(f"loaded local vector cache path={path} vectors={len(records)}")
        return self._local_index

    def _load_local_records(self) -> list[dict[str, Any]]:
        path = self._local_path()
        if not path.exists():
            return []
        return self._ensure_local_index().records

    def _write_local_index(self, records: list[dict[str, Any]]) -> None:
        path = self._local_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        tmp_path.replace(path)
        self._local_index = None

    def _ensure_pinecone_index(self):
        if self._index is not None:
            return self._index
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as exc:
            raise ImportError("Run: pip install pinecone") from exc
        if not self._config.pinecone_api_key:
            raise ValueError(
                "PINECONE_API_KEY is not set. Add it to backend/.env"
            )
        if not re.fullmatch(r"[a-z0-9-]+", self._config.pinecone_index_name):
            raise ValueError(
                "Invalid PINECONE_INDEX_NAME. Pinecone index names may contain "
                "only lowercase letters, numbers, and hyphens. Current value: "
                f"{self._config.pinecone_index_name!r}"
            )
        _progress(f"connecting to Pinecone index={self._config.pinecone_index_name}")
        pc = Pinecone(api_key=self._config.pinecone_api_key)
        existing = [i.name for i in pc.list_indexes()]
        if self._config.pinecone_index_name not in existing:
            _progress(f"creating Pinecone index={self._config.pinecone_index_name}")
            pc.create_index(
                name=self._config.pinecone_index_name,
                dimension=self._config.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        else:
            _progress(f"using existing Pinecone index={self._config.pinecone_index_name}")
        self._index = pc.Index(self._config.pinecone_index_name)
        return self._index

    def upsert(self, vectors: list[dict[str, Any]]) -> None:
        if self._backend == "local":
            existing = {record["id"]: record for record in self._load_local_records()}
            for vector in vectors:
                record = _record_from_vector(vector)
                if record["id"]:
                    existing[record["id"]] = record
            merged = list(existing.values())
            self._write_local_index(merged)
            _progress(
                f"upserted total vectors={len(vectors)} to local cache path={self._local_path()}"
            )
            return

        index = self._ensure_pinecone_index()
        for i in range(0, len(vectors), 100):
            batch = vectors[i:i + 100]
            _progress(
                f"upserting batch {i // 100 + 1} "
                f"vectors={len(batch)} namespace={self._config.pinecone_namespace}"
            )
            index.upsert(vectors=batch, namespace=self._config.pinecone_namespace)
        _progress(f"upserted total vectors={len(vectors)}")

    def query(self, vector: list[float], top_k: int | None = None, filter: dict | None = None) -> list[dict[str, Any]]:
        if self._backend == "local":
            local = self._ensure_local_index()
            if local.matrix.size == 0:
                return []

            query_vec = np.asarray(vector, dtype="float32")
            query_norm = float(np.linalg.norm(query_vec))
            if math.isclose(query_norm, 0.0):
                return []
            query_vec = query_vec / query_norm

            scores = local.matrix @ query_vec
            candidates: list[dict[str, Any]] = []
            for idx, record in enumerate(local.records):
                metadata = _as_dict(record.get("metadata"))
                if not _matches_filter(metadata, filter):
                    continue
                candidates.append(
                    {
                        "id": record.get("id", ""),
                        "score": float(scores[idx]),
                        "metadata": metadata,
                    }
                )

            candidates.sort(key=lambda item: item["score"], reverse=True)
            return candidates[: (top_k or self._config.top_k)]

        index = self._ensure_pinecone_index()
        resp = index.query(
            vector=vector,
            top_k=top_k or self._config.top_k,
            namespace=self._config.pinecone_namespace,
            include_metadata=True,
            filter=filter,
        )
        return [
            {"id": m.id, "score": float(m.score), "metadata": dict(m.metadata or {})}
            for m in resp.matches
        ]

    def delete_document(self, document_id: str) -> None:
        if self._backend == "local":
            path = self._local_path()
            if not path.exists():
                return
            local = self._ensure_local_index()
            kept = [
                record for record in local.records
                if _as_dict(record.get("metadata")).get("document_id") != document_id
            ]
            self._write_local_index(kept)
            _progress(f"deleted document_id={document_id} from local cache")
            return

        self._ensure_pinecone_index().delete(
            filter={"document_id": {"$eq": document_id}},
            namespace=self._config.pinecone_namespace,
        )

    def describe(self) -> dict[str, Any]:
        if self._backend == "local":
            local = self._ensure_local_index()
            dimension = int(local.matrix.shape[1]) if local.matrix.ndim == 2 and local.matrix.size else 0
            return {
                "backend": "local",
                "path": str(local.path),
                "total_vector_count": len(local.records),
                "dimension": dimension,
                "namespaces": {
                    self._config.pinecone_namespace: len(local.records),
                },
            }

        stats = self._ensure_pinecone_index().describe_index_stats()
        return {
            "backend": "pinecone",
            "total_vector_count": stats.total_vector_count,
            "dimension": stats.dimension,
            "namespaces": {ns: v.vector_count for ns, v in (stats.namespaces or {}).items()},
        }
