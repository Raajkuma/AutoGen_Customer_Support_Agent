"""Ticket store (SQLite) with deterministic escalation rules, human-approval state and an audit CSV export."""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime

import pandas as pd

import config
import db

log = logging.getLogger(__name__)

COLUMNS = ["ticket_id", "timestamp", "customer_name", "order_id", "complaint", "category", "urgency",
           "sentiment", "kb_sources", "response", "status", "escalation_reason", "confidence", "approval",
           "reviewed_by", "reviewed_at", "final_response", "citations", "language"]
REJECT_MESSAGE = "Thank you for your patience. A support specialist will contact you directly to resolve this."

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS tickets(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id TEXT UNIQUE,
  {", ".join(f"{c} TEXT DEFAULT ''" for c in COLUMNS if c != "ticket_id")});
"""
# Legal / fraud / payment-dispute language. "bank" alone is NOT a trigger ("I bank with HDFC"), only
# threats to involve it.
_RISK = re.compile(
    r"\b(lawyer|attorney|legal action|sue|suing|court|fraud|scam|chargeback|police|consumer (?:forum|court)|dispute)\b"
    r"|\b(?:contact|call|report|inform|tell|go to)(?:ing)? (?:my |the )?bank\b", re.I)


def init() -> None:
    with db.connect() as c:
        c.executescript(_SCHEMA)


def escalation_reasons(complaint: str, urgency: str, sentiment: str, kb_sources: str, confidence: str) -> list[str]:
    reasons: list[str] = []
    if urgency.strip().lower() == "high":
        reasons.append("High urgency")
    if sentiment.strip().lower() in {"angry", "very angry"}:
        reasons.append("Angry customer")
    if not kb_sources.strip() or kb_sources.strip().upper() == "NONE" or "NO_MATCH" in kb_sources.upper():
        reasons.append("No matching knowledge-base document")
    if _RISK.search(complaint):
        reasons.append("Legal / fraud / payment-dispute language")
    if confidence.strip().lower() == "low":
        reasons.append("Low reply confidence")
    return reasons


def export_csv() -> None:
    """Rewrite the human-readable audit log atomically (temp file + rename)."""
    config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = config.LOG_FILE.with_suffix(".tmp")
    read_all().to_csv(tmp, index=False)
    os.replace(tmp, config.LOG_FILE)


def read_all() -> pd.DataFrame:
    init()
    with db.connect() as c:
        rows = c.execute(f"SELECT {', '.join(COLUMNS)} FROM tickets ORDER BY id").fetchall()
    return pd.DataFrame([dict(r) for r in rows], columns=COLUMNS).fillna("")


def log_ticket(customer_name: str, order_id: str, complaint: str, category: str, urgency: str, sentiment: str,
               kb_sources: str, response: str, confidence: str = "High", citations: str = "",
               language: str = "en") -> dict:
    """Insert one ticket. Status and approval come from the escalation rules, never from the model."""
    init()
    reasons = escalation_reasons(complaint, urgency, sentiment, kb_sources, confidence)
    escalated = bool(reasons)
    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "customer_name": customer_name,
           "order_id": order_id, "complaint": complaint, "category": category, "urgency": urgency,
           "sentiment": sentiment, "kb_sources": kb_sources, "response": response,
           "status": "Escalated" if escalated else "Resolved", "escalation_reason": "; ".join(reasons),
           "confidence": confidence, "approval": "Pending" if escalated else "Not required",
           "final_response": "" if escalated else response, "citations": citations, "language": language}
    with db.connect() as c:
        cur = c.execute(f"INSERT INTO tickets({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                        list(row.values()))
        ticket_id = f"TKT-{cur.lastrowid:04d}"
        c.execute("UPDATE tickets SET ticket_id=? WHERE id=?", (ticket_id, cur.lastrowid))
    export_csv()
    log.info("ticket %s logged status=%s approval=%s reasons=%s", ticket_id, row["status"], row["approval"], reasons)
    return {"ticket_id": ticket_id, "status": row["status"], "reasons": reasons, "approval": row["approval"]}


def get_ticket(ticket_id: str) -> dict | None:
    df = read_all()
    hit = df[df["ticket_id"] == ticket_id]
    return None if hit.empty else hit.iloc[0].to_dict()


def review_ticket(ticket_id: str, reviewer: str, final_text: str, reject: bool = False) -> dict | None:
    """Human decision on a pending ticket: approve, approve with edits, or reject. Returns None if not pending."""
    init()
    with db.connect() as c:
        row = c.execute("SELECT response FROM tickets WHERE ticket_id=? AND approval='Pending'",
                        (ticket_id,)).fetchone()
        if row is None:
            return None
        draft = row["response"].strip()
        if reject:
            decision, final = "Rejected", REJECT_MESSAGE
        else:
            final = final_text.strip() or draft
            decision = "Approved" if final == draft else "Edited"
        c.execute("UPDATE tickets SET approval=?, reviewed_by=?, final_response=?, reviewed_at=? "
                  "WHERE ticket_id=? AND approval='Pending'",
                  (decision, reviewer, final, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ticket_id))
    export_csv()
    log.info("ticket %s reviewed decision=%s reviewer=%s", ticket_id, decision, reviewer)
    return {"ticket_id": ticket_id, "decision": decision, "final_response": final}
