from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class DenseEmbedder:
    def __init__(self, model_name: str, device: str):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype="float32")

    @property
    def dimension(self) -> int:
        return int(self.encode(["dimension probe"]).shape[1])
