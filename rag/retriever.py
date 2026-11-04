"""Hybrid retrieval: BM25 + dense search run in parallel, fused with reciprocal rank fusion, then reranked."""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

import config
from rag.bm25 import BM25
from rag.embeddings import Embedder, EmbeddingError
from rag.index import Index, build_index, kb_digest
from rag.rerank import rerank
from rag.text import tokenize
from rag.types import Hit, SearchResult

log = logging.getLogger(__name__)


class _Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


def rrf_fuse(rankings: list[list[int]], k: int = config.RRF_K) -> dict[int, float]:
    """Reciprocal rank fusion: score(d) = sum over rankers of 1 / (k + rank)."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return fused


@dataclass
class Retriever:
    index: Index
    embedder: _Embedder | None = None
    k: int = config.RETRIEVAL_K
    top_n: int = config.RERANK_TOP_N
    min_mass_routed: float = config.MIN_MASS_ROUTED
    min_mass_open: float = config.MIN_MASS_OPEN
    min_coverage_open: float = config.MIN_COVERAGE_OPEN
    dense_min: float = config.DENSE_MIN
    bm25: BM25 = field(init=False)

    def __post_init__(self) -> None:
        self.bm25 = BM25([tokenize(c.index_text) for c in self.index.chunks])

    def _bm25_rank(self, tokens: list[str]) -> list[tuple[int, float]]:
        scores = self.bm25.scores(tokens)
        order = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: scores[i], reverse=True)
        return [(i, scores[i]) for i in order[: self.k * 2]]

    def _dense_rank(self, query: str) -> tuple[list[tuple[int, float]], str]:
        if self.index.matrix is None or self.embedder is None:
            return [], "no dense index"
        try:
            q = self.embedder.embed([query])[0]
        except EmbeddingError as exc:
            return [], f"query embedding failed: {exc}"
        sims = self.index.matrix @ q
        order = np.argsort(-sims)[: self.k * 2]
        return [(int(i), float(sims[i])) for i in order], ""

    def _routes(self, category: str | None) -> set[str]:
        """Documents the Query agent's category points at (knowledge_base/routing.json, optional)."""
        path = config.KB_DIR / "routing.json"
        if not category or not path.exists():
            return set()
        return set(json.loads(path.read_text(encoding="utf-8")).get(category, []))

    def _relevant(self, h: Hit, routed: set[str]) -> bool:
        """Evidence gate. Dense similarity alone can pass; otherwise lexical evidence is required, stricter
        outside the routed document. Failing closed here means 'no match' and therefore escalation."""
        if h.dense_sim is not None and h.dense_sim >= self.dense_min:
            return True
        if h.chunk.source in routed:
            return h.mass >= self.min_mass_routed
        return h.mass >= self.min_mass_open and h.coverage >= self.min_coverage_open

    def search(self, query: str, category: str | None = None) -> SearchResult:
        t0 = time.perf_counter()
        tokens = tokenize(query)
        with ThreadPoolExecutor(max_workers=2) as pool:  # sparse and dense retrieval run concurrently
            f_sparse, f_dense = pool.submit(self._bm25_rank, tokens), pool.submit(self._dense_rank, query)
            sparse, (dense, dense_note) = f_sparse.result(), f_dense.result()
        fused = rrf_fuse([[i for i, _ in sparse], [i for i, _ in dense]])
        hits = {i: Hit(self.index.chunks[i], rrf=s) for i, s in fused.items()}
        for rank, (i, _) in enumerate(sparse, 1):
            hits[i].bm25_rank = rank
        for rank, (i, sim) in enumerate(dense, 1):
            hits[i].dense_rank, hits[i].dense_sim = rank, sim
        candidates = sorted(hits.values(), key=lambda h: h.rrf, reverse=True)[: self.k]
        routed = self._routes(category)
        ranked = [h for h in rerank(candidates, set(tokens), self.bm25.idf, routed) if self._relevant(h, routed)]
        # Drop weak tails by *evidence* (dense similarity, else lexical mass) so the routing prior cannot prop them up.
        evidence = (lambda h: h.dense_sim if h.dense_sim is not None else h.mass)
        best = max((evidence(h) for h in ranked), default=0.0)
        final = [h for h in ranked[: self.top_n] if evidence(h) >= 0.5 * best]
        matched = bool(final)
        mode = "hybrid" if dense else f"bm25-only ({dense_note})"
        debug: dict[str, Any] = {
            "mode": mode, "seconds": round(time.perf_counter() - t0, 4), "candidates": len(candidates),
            "hits": [{"chunk_id": h.chunk.chunk_id, "rrf": round(h.rrf, 5), "bm25_rank": h.bm25_rank,
                      "dense_rank": h.dense_rank, "dense_sim": None if h.dense_sim is None else round(h.dense_sim, 3),
                      "coverage": round(h.coverage, 3), "mass": round(h.mass, 2), "final": round(h.final, 3)} for h in final]}
        log.info("search mode=%s matched=%s hits=%s", mode, matched, [h.chunk.chunk_id for h in final])
        return SearchResult(query, final if matched else [], matched, mode, debug)


_cache: dict[str, Any] = {"key": None, "retriever": None, "built": 0.0}


def get_retriever() -> Retriever:
    """Process-wide retriever, rebuilt when the KB changes or when dense search was unavailable (retry after 30 s)."""
    key = str(sorted(kb_digest().items()))
    stale_dense = (config.EMBED_ENABLED and _cache["retriever"] is not None
                   and _cache["retriever"].index.matrix is None and time.time() - _cache["built"] > 30)
    if _cache["key"] != key or stale_dense:
        embedder = Embedder() if config.EMBED_ENABLED else None
        _cache.update(key=key, retriever=Retriever(build_index(embedder), embedder), built=time.time())
    return _cache["retriever"]
