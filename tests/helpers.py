"""Deterministic fake embedders for tests (no model server needed)."""
import numpy as np

from rag.embeddings import EmbeddingError
from rag.text import tokenize


class HashEmbedder:
    """Bag-of-tokens hashed into 64 dims: similar token sets give similar vectors."""
    model = "hash-test"

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += len(texts)
        out = np.zeros((len(texts), 64), np.float32)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                out[i, sum(map(ord, tok)) % 64] += 1.0
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


class DownEmbedder:
    model = "down"

    def embed(self, texts):
        raise EmbeddingError("connection refused")
