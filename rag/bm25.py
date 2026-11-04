"""Okapi BM25 (no external dependency)."""
from __future__ import annotations

import math
from collections import Counter


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.avgdl = (sum(self.lengths) / len(docs)) if docs else 0.0
        self.df: Counter[str] = Counter(t for d in self.docs for t in d)
        self.n = len(docs)

    def idf(self, token: str) -> float:
        df = self.df.get(token, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for doc, length in zip(self.docs, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / (self.avgdl or 1))
            out.append(sum(self.idf(t) * doc[t] * (self.k1 + 1) / (doc[t] + norm) for t in set(query) if t in doc))
        return out
