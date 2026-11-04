"""Tokenisation shared by BM25, reranking and evaluation. Keeps Tamil/Indic words intact."""
from __future__ import annotations

import re

# \w alone splits Tamil words at vowel signs (combining marks), so the Indic blocks are listed explicitly.
_TOKEN = re.compile(r"[\w\u0900-\u0DFF]+", re.UNICODE)
STOP = frozenset("a an the i my me is are was were to of for and or it in on at do does can you your we our this "
                 "that with have has had be been not no how what when where why please want get just now will would "
                 "should could if so as by from about up out any some there their they them long take took but "
                 "still never very much many".split())
# Concept normalisation on stems: query wording -> the vocabulary the policy documents use.
CONCEPTS = {"return": "refund", "money": "refund", "delivery": "ship", "deliver": "ship", "arriv": "ship",
            "arrival": "ship", "parcel": "order", "packag": "order", "late": "delay", "broken": "damag",
            "defective": "damag", "faulty": "damag", "replac": "replacement", "charg": "bill", "payment": "bill",
            "debit": "bill", "pay": "bill", "paid": "bill", "forgot": "password", "login": "password",
            "upi": "bill"}


def stem(word: str) -> str:
    """Tiny suffix stripper (ing/ed/es/s, doubled consonants, trailing e) - enough for policy vocabulary."""
    if not word.isascii() or word.isdigit():
        return word
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            word = word[: -len(suffix)]
            if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiou":
                word = word[:-1]
            break
    return word[:-1] if len(word) > 4 and word.endswith("e") else word


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for word in _TOKEN.findall(text.lower()):
        if word in STOP or len(word) < 2:
            continue
        s = stem(word)
        out.append(CONCEPTS.get(s, CONCEPTS.get(word, s)))
    return out


def slugify(text: str) -> str:
    return re.sub(r"[^\w\u0900-\u0DFF]+", "-", text.lower()).strip("-")
