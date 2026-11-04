"""Retrieval and escalation metrics."""
from __future__ import annotations

from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of relevant chunk ids found in the top-k results."""
    if not relevant:
        raise ValueError("recall is undefined without relevant ids")
    return len(set(retrieved[:k]) & set(relevant)) / len(set(relevant))


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """1 / rank of the first relevant result (0 if none)."""
    wanted = set(relevant)
    return next((1.0 / r for r, cid in enumerate(retrieved, 1) if cid in wanted), 0.0)


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)
