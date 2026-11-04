"""AutoGen FunctionTools used by the Knowledge and Log agents."""
from __future__ import annotations

import json

from autogen_core.tools import FunctionTool

import tickets
from rag.retriever import get_retriever


def search_kb(query: str, category: str = "") -> str:
    """Search the company knowledge base (hybrid BM25 + dense, reranked). Returns JSON with the cited passages."""
    result = get_retriever().search(query, category or None)
    return json.dumps({"passages": result.format(), "matched": result.matched, "mode": result.mode,
                       "chunk_ids": result.chunk_ids, "debug": result.debug})


def log_ticket(customer_name: str, order_id: str, complaint: str, category: str, urgency: str, sentiment: str,
               kb_sources: str, response: str, confidence: str, citations: str, language: str) -> str:
    """Write one ticket to the ticket store. Escalation status is decided by code. Returns JSON."""
    return json.dumps(tickets.log_ticket(customer_name, order_id, complaint, category, urgency, sentiment,
                                         kb_sources, response, confidence, citations, language))


search_kb_tool = FunctionTool(search_kb, description=search_kb.__doc__ or "", name="search_kb")
log_ticket_tool = FunctionTool(log_ticket, description=log_ticket.__doc__ or "", name="log_ticket")
