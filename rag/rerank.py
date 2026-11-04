"""Second-stage reranking of fused candidates.

Score = idf-weighted query-term coverage (lexical evidence) blended with dense cosine similarity
(semantic evidence, when embeddings exist) plus a small prior for passages from the document the
Query agent's category routes to. This is the swap-in point for a cross-encoder such as
bge-reranker-v2-m3 if you have the hardware for it.
"""
from __future__ import annotations

from collections.abc import Callable

from rag.text import tokenize
from rag.types import Hit


def overlap(query_tokens: set[str], chunk_tokens: set[str], idf: Callable[[str], float]) -> tuple[float, float]:
    """Return (idf mass of shared terms, fraction of the query's idf mass that is covered)."""
    total = sum(idf(t) for t in query_tokens)
    mass = sum(idf(t) for t in query_tokens & chunk_tokens)
    return mass, (mass / total if total else 0.0)


def rerank(hits: list[Hit], query_tokens: set[str], idf: Callable[[str], float], routed: set[str],
           dense_weight: float = 0.5, routing_prior: float = 0.15) -> list[Hit]:
    for h in hits:
        h.mass, h.coverage = overlap(query_tokens, set(tokenize(f"{h.chunk.heading} {h.chunk.text}")), idf)
        lexical = h.coverage
        sim = max(h.dense_sim, 0.0) if h.dense_sim is not None else None
        h.final = lexical if sim is None else (1 - dense_weight) * lexical + dense_weight * sim
        if h.chunk.source in routed:
            h.final += routing_prior
    return sorted(hits, key=lambda h: h.final, reverse=True)
