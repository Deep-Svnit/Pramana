"""Embedder: local CPU/GPU models only, no embedding API calls."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.rag_v2.config import RagV2Config


class LocalEmbedder:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self._model = None
        self._hf_model = None
        self._tokenizer = None
        self._device = device
        self._is_e5 = "e5" in model_name.lower() or Path(model_name).name.lower().startswith("e5")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Run: pip install sentence-transformers") from exc

        try:
            self._model = SentenceTransformer(model_name, device=device)
        except TypeError:
            # Some local sentence-transformers exports miss the Pooling config
            # required by newer sentence-transformers. Load the underlying HF
            # transformer and do normalized mean pooling locally.
            self._load_hf_model(model_name, device)

    def _load_hf_model(self, model_name: str, device: str) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Run: pip install transformers torch") from exc

        local_only = Path(model_name).exists()
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
        self._hf_model = AutoModel.from_pretrained(model_name, local_files_only=local_only)
        self._hf_model.to(device)
        self._hf_model.eval()

    def _format_texts(self, texts: list[str]) -> list[str]:
        if not self._is_e5:
            return texts
        return [
            text if text.startswith(("query:", "passage:")) else f"passage: {text}"
            for text in texts
        ]

    def embed(self, texts: list[str]) -> np.ndarray:
        texts = self._format_texts(texts)
        if self._model is not None:
            vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vecs, dtype="float32")

        assert self._hf_model is not None and self._tokenizer is not None
        with self._torch.no_grad():
            encoded = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._device)
            output = self._hf_model(**encoded)
            token_embeddings = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            vecs = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            vecs = self._torch.nn.functional.normalize(vecs, p=2, dim=1)
        return np.asarray(vecs.detach().cpu().numpy(), dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        if self._is_e5 and not query.startswith("query:"):
            return self.embed([f"query: {query}"])[0]
        return self.embed([query])[0]

    @property
    def dimension(self) -> int:
        return int(self.embed(["probe"]).shape[1])


def build_embedder(config: RagV2Config) -> LocalEmbedder:
    return LocalEmbedder(model_name=config.embedding_model, device=config.embedding_device)
