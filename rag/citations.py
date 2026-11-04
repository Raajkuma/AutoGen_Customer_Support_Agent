"""Parse retrieved passages and verify the [S1]-style citations in a generated reply."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_PASSAGE = re.compile(r"^\[(S\d+)\] (\S+)\n(.*?)(?=^\[S\d+\] |\Z)", re.M | re.S)
_CITE = re.compile(r"\[(S\d+)\]")
_SENTENCE = re.compile(r"(?<=[.!?।])\s+")


def parse_passages(retrieval_text: str) -> dict[str, dict[str, str]]:
    return {m.group(1): {"chunk_id": m.group(2), "text": m.group(3).strip()}
            for m in _PASSAGE.finditer(retrieval_text)}


@dataclass
class CitationCheck:
    reply: str                                   # reply with invalid citations removed
    cited: list[str] = field(default_factory=list)      # valid labels, in order of first use
    invalid: list[str] = field(default_factory=list)    # labels that do not exist in the retrieved passages
    uncited_sentences: list[str] = field(default_factory=list)  # substantive sentences without a citation


def check_citations(reply: str, passages: dict[str, dict[str, str]], min_words: int = 7) -> CitationCheck:
    labels = _CITE.findall(reply)
    cited = list(dict.fromkeys(label for label in labels if label in passages))
    invalid = list(dict.fromkeys(label for label in labels if label not in passages))
    cleaned = _CITE.sub(lambda m: m.group(0) if m.group(1) in passages else "", reply)
    cleaned = re.sub(r"[ \t]+([.,;!?])", r"\1", re.sub(r"[ \t]{2,}", " ", cleaned)).strip()
    uncited = [s for s in _SENTENCE.split(cleaned) if len(s.split()) >= min_words and not _CITE.search(s)]
    return CitationCheck(cleaned, cited, invalid, uncited)


def citation_map(cited: list[str], passages: dict[str, dict[str, str]]) -> str:
    return "; ".join(f"{label}={passages[label]['chunk_id']}" for label in cited)
