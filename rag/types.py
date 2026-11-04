"""Shared data types for retrieval."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

NO_MATCH = "NO_MATCH: no relevant document found."


@dataclass(frozen=True)
class Chunk:
    chunk_id: str   # stable id: "<file>#<heading-slug>[~n]"
    source: str     # file name
    heading: str
    text: str
    sha: str        # sha256 of heading+text, used to skip re-embedding unchanged chunks

    @property
    def index_text(self) -> str:
        return f"{self.heading}\n{self.text}"


@dataclass
class Hit:
    chunk: Chunk
    rrf: float = 0.0
    bm25_rank: int | None = None
    dense_rank: int | None = None
    dense_sim: float | None = None
    coverage: float = 0.0
    mass: float = 0.0
    final: float = 0.0


@dataclass
class SearchResult:
    query: str
    hits: list[Hit]
    matched: bool
    mode: str                                   # "hybrid" or "bm25-only (<reason>)"
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_ids(self) -> list[str]:
        return [h.chunk.chunk_id for h in self.hits]

    def format(self) -> str:
        """Text handed to the Resolution agent; every passage gets a citation label [S1], [S2]..."""
        if not self.matched:
            return NO_MATCH
        return "\n\n".join(f"[S{i}] {h.chunk.chunk_id}\n{h.chunk.text}" for i, h in enumerate(self.hits, 1))
