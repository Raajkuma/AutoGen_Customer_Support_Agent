"""Embedding client for any OpenAI-compatible /embeddings endpoint (e.g. Docker Model Runner)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence

import numpy as np

import config


class EmbeddingError(RuntimeError):
    """Embedding service unreachable or returned something unusable."""


class Embedder:
    def __init__(self, base_url: str = config.BASE_URL, model: str = config.EMBED_MODEL,
                 api_key: str = config.API_KEY, timeout: float = 30.0, batch_size: int = 16) -> None:
        self.url, self.model, self.api_key = base_url.rstrip("/") + "/embeddings", model, api_key
        self.timeout, self.batch_size = timeout, batch_size

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalised float32 vectors, so inner product equals cosine similarity."""
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            body = json.dumps({"model": self.model, "input": list(texts[i:i + self.batch_size])}).encode()
            req = urllib.request.Request(self.url, data=body, headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.load(resp)["data"]
            except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError) as exc:
                raise EmbeddingError(f"{type(exc).__name__}: {exc}") from exc
            vectors += [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]
        mat = np.asarray(vectors, dtype=np.float32)
        return mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
