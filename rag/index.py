"""Persisted knowledge-base index: chunks.jsonl + dense.npy + manifest.json, incremental by chunk hash."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

import config
from rag.chunker import load_chunks
from rag.embeddings import EmbeddingError
from rag.types import Chunk

log = logging.getLogger(__name__)


class _Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class Index:
    chunks: list[Chunk]
    matrix: np.ndarray | None
    stats: dict[str, Any] = field(default_factory=dict)


def kb_digest(kb_dir: Path | None = None) -> dict[str, str]:
    """sha256 per knowledge-base file (the audit trail of what was indexed)."""
    kb_dir = kb_dir or config.KB_DIR
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(kb_dir.glob("*.md"))}


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def build_index(embedder: _Embedder | None, kb_dir: Path | None = None, index_dir: Path | None = None,
                size: int | None = None, overlap: int | None = None) -> Index:
    """Chunk the KB and embed only chunks that are new or changed (same chunk_id + sha are reused)."""
    kb_dir, index_dir = kb_dir or config.KB_DIR, index_dir or config.INDEX_DIR
    chunks = load_chunks(kb_dir, size, overlap)
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, dense_path = index_dir / "manifest.json", index_dir / "dense.npy"

    cache: dict[str, tuple[str, np.ndarray]] = {}
    if embedder and manifest_path.exists() and dense_path.exists():
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            vecs = np.load(dense_path)
            if old.get("embed_model") == embedder.model and len(old.get("order", [])) == len(vecs):
                cache = {cid: (old["chunks"][cid], vecs[i]) for i, cid in enumerate(old["order"])}
        except (ValueError, KeyError, OSError):
            log.warning("index cache unreadable; rebuilding")

    stats: dict[str, Any] = {"chunks": len(chunks), "reused": 0, "embedded": 0, "dense": False}
    matrix: np.ndarray | None = None
    if embedder:
        todo = [i for i, c in enumerate(chunks) if not (c.chunk_id in cache and cache[c.chunk_id][0] == c.sha)]
        try:
            new = embedder.embed([chunks[i].index_text for i in todo]) if todo else np.zeros((0, 0), np.float32)
            fresh = dict(zip(todo, new))
            matrix = np.vstack([fresh[i] if i in fresh else cache[c.chunk_id][1] for i, c in enumerate(chunks)])
            stats.update(dense=True, reused=len(chunks) - len(todo), embedded=len(todo),
                         removed=len(set(cache) - {c.chunk_id for c in chunks}))
        except EmbeddingError as exc:
            stats["dense_error"] = str(exc)
            log.warning("dense index unavailable: %s", exc)

    _atomic_write(index_dir / "chunks.jsonl", "\n".join(json.dumps(asdict(c), ensure_ascii=False)
                                                        for c in chunks).encode("utf-8"))
    if matrix is not None and embedder:
        buf = index_dir / "dense.tmp.npy"
        np.save(buf, matrix)
        os.replace(buf, dense_path)
        _atomic_write(manifest_path, json.dumps({
            "embed_model": embedder.model, "files": kb_digest(kb_dir), "order": [c.chunk_id for c in chunks],
            "chunks": {c.chunk_id: c.sha for c in chunks}}, indent=2).encode("utf-8"))
    log.info("index built: %s", stats)
    return Index(chunks, matrix, stats)
