"""Heading-aware chunking with word-window overlap and stable chunk ids."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import config
from rag.text import slugify
from rag.types import Chunk


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_markdown(source: str, markdown: str, size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    """Split on '## ' sections; sections longer than `size` words become overlapping windows."""
    size = config.CHUNK_SIZE if size is None else size
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("require size > 0 and 0 <= overlap < size")
    chunks: list[Chunk] = []
    for block in re.split(r"(?m)^## ", markdown)[1:]:
        heading, _, body = block.partition("\n")
        heading, body = heading.strip(), body.strip()
        if not body:
            continue
        base_id = f"{source}#{slugify(heading)}"
        words = body.split()
        if len(words) <= size:
            windows = [body]
        else:
            step = size - overlap
            windows = [" ".join(words[i:i + size]) for i in range(0, len(words), step)
                       if len(words[i:i + size]) > overlap or i == 0]
        for n, text in enumerate(windows):
            chunks.append(Chunk(base_id if n == 0 else f"{base_id}~{n + 1}", source, heading, text,
                                _sha(f"{heading}\n{text}")))
    return chunks


def load_chunks(kb_dir: Path | None = None, size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    kb_dir = kb_dir or config.KB_DIR
    return [c for path in sorted(kb_dir.glob("*.md"))
            for c in chunk_markdown(path.name, path.read_text(encoding="utf-8"), size, overlap)]
