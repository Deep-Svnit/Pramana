from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from powermind_rag.faiss_store import FaissStore
from powermind_rag.schema import RetrievedChunk, VisualPageRecord


class ColPaliVisualIndex:
    """ColPali/byaldi page-level index plus FAISS storage for pooled page embeddings."""

    def __init__(self, model_name: str, device: str):
        try:
            from byaldi import RAGMultiModalModel
        except ImportError as exc:
            raise RuntimeError("ColPali visual ingestion must use byaldi. Install byaldi.") from exc
        self.model_name = model_name
        self.device = device
        try:
            self.model = RAGMultiModalModel.from_pretrained(model_name, device=device)
        except TypeError:
            self.model = RAGMultiModalModel.from_pretrained(model_name)
            nested = getattr(self.model, "model", self.model)
            if hasattr(nested, "to"):
                nested.to(device)
        self.records: list[VisualPageRecord] = []
        self.faiss: FaissStore[VisualPageRecord] | None = None

    def build(self, image_paths: list[Path], document_id: str, metadata: dict[str, Any]) -> None:
        self._build_byaldi_collection(image_paths, document_id)
        records: list[VisualPageRecord] = []
        vectors: list[np.ndarray] = []
        for page_number, image_path in enumerate(image_paths, start=1):
            embedding = self._embed_page(image_path)
            vectors.append(embedding)
            records.append(
                VisualPageRecord(
                    embedding=embedding.tolist(),
                    document_id=document_id,
                    page_number=page_number,
                    modality="image",
                    raw_image_path=image_path,
                    metadata=metadata,
                )
            )
        matrix = np.asarray(vectors, dtype="float32")
        self.faiss = FaissStore(dimension=matrix.shape[1])
        self.faiss.add(matrix, records)
        self.records.extend(records)

    def _build_byaldi_collection(self, image_paths: list[Path], document_id: str) -> None:
        index_method = getattr(self.model, "index", None)
        if index_method is None:
            return
        try:
            index_method(
                input_path=str(image_paths[0].parent),
                index_name=f"{document_id}_visual",
                store_collection_with_index=True,
                overwrite=True,
            )
        except TypeError:
            index_method(str(image_paths[0].parent), index_name=f"{document_id}_visual", overwrite=True)

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        byaldi_hits = self._byaldi_search(query, top_k)
        if byaldi_hits:
            return byaldi_hits
        if self.faiss is None:
            raise RuntimeError("Visual FAISS index has not been built.")
        query_vector = self._embed_query(query)
        return [
            self._to_retrieved(record, rank, score)
            for record, score, rank in self.faiss.search(query_vector, top_k)
        ]

    def _byaldi_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if not hasattr(self.model, "search"):
            return []
        try:
            hits = self.model.search(query, k=top_k)
        except TypeError:
            return []
        retrieved: list[RetrievedChunk] = []
        for rank, hit in enumerate(hits, start=1):
            page_number = int(getattr(hit, "page_num", getattr(hit, "page_number", 0)) or 0)
            score = float(getattr(hit, "score", 0.0))
            hit_doc = getattr(hit, "doc_id", getattr(hit, "document_id", None))
            candidates = [item for item in self.records if item.page_number == page_number]
            if hit_doc:
                candidates = [item for item in candidates if item.document_id == str(hit_doc)]
            if len(candidates) != 1:
                return []
            record = candidates[0]
            if not record:
                continue
            retrieved.append(self._to_retrieved(record, rank, score))
        return retrieved

    def _embed_page(self, image_path: Path) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        for method_name in ("encode_images", "encode_image", "embed_images", "embed_image"):
            method = getattr(self.model, method_name, None)
            if method is None:
                continue
            try:
                raw = method([image]) if method_name.endswith("s") else method(image)
                return self._pool_normalize(raw)
            except Exception:
                continue
        nested = getattr(self.model, "model", None)
        if nested is not None:
            for method_name in ("encode_images", "encode_image", "embed_images", "embed_image"):
                method = getattr(nested, method_name, None)
                if method is None:
                    continue
                try:
                    raw = method([image]) if method_name.endswith("s") else method(image)
                    return self._pool_normalize(raw)
                except Exception:
                    continue
        raise RuntimeError("Unable to obtain page-level ColPali embeddings from byaldi model.")

    def _embed_query(self, query: str) -> np.ndarray:
        for method_name in ("encode_queries", "encode_query", "embed_queries", "embed_query"):
            method = getattr(self.model, method_name, None)
            if method is None:
                continue
            try:
                raw = method([query]) if method_name.endswith("s") else method(query)
                return self._pool_normalize(raw)
            except Exception:
                continue
        raise RuntimeError("Unable to obtain ColPali query embedding from byaldi model.")

    @staticmethod
    def _pool_normalize(raw: Any) -> np.ndarray:
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu().numpy()
        array = np.asarray(raw, dtype="float32")
        while array.ndim > 1:
            array = array.mean(axis=0)
        norm = np.linalg.norm(array)
        if norm == 0:
            raise RuntimeError("ColPali produced a zero vector.")
        return array / norm

    @staticmethod
    def _to_retrieved(record: VisualPageRecord, rank: int, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            id=f"{record.document_id}:p{record.page_number}:image",
            text=(
                f"Visual page evidence only. Source image: {record.raw_image_path}. "
                "Do not extract numeric values unless separate textual evidence supports them."
            ),
            document_id=record.document_id,
            page_number=record.page_number,
            modality="image",
            rank=rank,
            score=score,
            metadata={**record.metadata, "chunk_label": "image"},
        )

    def save_records(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(record) for record in self.records]
        path.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")

    def save_document_records(self, document_id: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(record) for record in self.records if record.document_id == document_id]
        path.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")

    def load_records(self, records: list[VisualPageRecord]) -> None:
        self.records = records
        if not records:
            return
        matrix = np.asarray([record.embedding for record in records], dtype="float32")
        self.faiss = FaissStore(dimension=matrix.shape[1])
        self.faiss.add(matrix, records)


def load_visual_records(path: Path) -> list[VisualPageRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: list[VisualPageRecord] = []
    for item in payload:
        item["raw_image_path"] = Path(item["raw_image_path"])
        records.append(VisualPageRecord(**item))
    return records
