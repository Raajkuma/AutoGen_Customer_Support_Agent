"""Pure text helpers that make small-model output safe to use (no AutoGen dependency)."""
from __future__ import annotations

import re

CATEGORIES = ["Shipping", "Refund", "Account", "Product", "Billing", "Other"]
URGENCY = ["Low", "Medium", "High"]
SENTIMENT = ["Calm", "Frustrated", "Angry"]


def strip_think(text: str) -> str:
    """Remove Qwen-style <think>...</think> reasoning from a model reply."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    return text.split("</think>")[-1].replace("<think>", "").strip()


def _field(text: str, name: str) -> str:
    m = re.search(rf"(?im)^\W*{name}\W*[:\-]\s*(.+)$", text)
    return m.group(1).strip() if m else ""


def _pick(value: str, allowed: list[str], default: str) -> str:
    v = value.lower()
    return next((a for a in allowed if a.lower() in v), default)


def parse_query(text: str) -> dict:
    summary = _field(text, "summary") or (text.strip().splitlines() or [""])[0][:160]
    return {"category": _pick(_field(text, "category"), CATEGORIES, "Other"),
            "urgency": _pick(_field(text, "urgency"), URGENCY, "Medium"),
            "sentiment": _pick(_field(text, "sentiment"), SENTIMENT, "Calm"),
            "summary": summary}


def format_query(d: dict) -> str:
    return (f"Category: {d['category']}\nUrgency: {d['urgency']}\n"
            f"Sentiment: {d['sentiment']}\nSummary: {d['summary']}")


def parse_resolution(text: str) -> dict:
    m = re.search(r"(?is)reply\W*[:\-]\s*(.*?)(?=\n\W*(?:confidence|sources)\s*[:\-]|\Z)", text)
    reply = (m.group(1) if m else re.sub(r"(?im)^\W*(confidence|sources)\W*[:\-].*$", "", text)).strip()
    conf = _field(text, "confidence")
    return {"reply": reply, "confidence": _pick(conf, ["High", "Medium", "Low"], "Low")}  # fail closed


def ungrounded_numbers(reply: str, *facts: str) -> list[str]:
    """Numbers in the reply that appear nowhere in the facts it was allowed to use."""
    reply = re.sub(r"\[S\d+\]", "", reply)  # citation labels are not claims
    allowed = set(re.findall(r"\d+", " ".join(facts)))
    return sorted(set(re.findall(r"\d+", reply)) - allowed)


def parse_task(task: str) -> dict:
    def line(label):
        m = re.search(rf"(?im)^{label}:\s*(.*)$", task)
        return m.group(1).strip() if m else ""
    m = re.search(r'"""(.*?)"""', task, re.S)
    return {"name": line("Customer name"), "order_id": line("Order ID") or "N/A",
            "order_details": line("Verified order details"), "complaint": (m.group(1) if m else task).strip()}


def format_resolution(reply: str, confidence: str, sources: list[str], citations: str = "") -> str:
    return (f"Reply: {reply}\nConfidence: {confidence}\nSources: {', '.join(sources) or 'NONE'}\n"
            f"Citations: {citations or 'NONE'}")


def parse_resolution_meta(text: str) -> dict:
    """Sources / Citations lines written by the Resolution agent (code-generated, so reliable)."""
    sources, citations = _field(text, "sources"), _field(text, "citations")
    return {"sources": "" if sources.upper() == "NONE" else sources,
            "citations": "" if citations.upper() == "NONE" else citations}


def sanitize_complaint(text: str, limit: int = 1000) -> str:
    """Neutralise delimiter tricks and control characters in untrusted customer text."""
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text).replace('"' * 3, "'" * 3)
    return re.sub(r"[ \t]+", " ", text).strip()[:limit]


def detect_language(text: str) -> str:
    """Script-based guess: 'ta' (Tamil), 'hi' (Devanagari) or 'en'."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    tamil = sum("\u0b80" <= c <= "\u0bff" for c in letters) / len(letters)
    deva = sum("\u0900" <= c <= "\u097f" for c in letters) / len(letters)
    return "ta" if tamil > 0.3 else "hi" if deva > 0.3 else "en"


LANGUAGE_NAMES = {"en": "English", "ta": "Tamil", "hi": "Hindi"}
