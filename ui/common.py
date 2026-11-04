"""Small helpers shared by the Streamlit pages."""
from __future__ import annotations

import streamlit as st

import workflow_trace as tr


def money(amount: int) -> str:
    return f"₹{amount:,}"


def go(page: str, order_id: str | None = None) -> None:
    """Button callback: navigate to a page (optionally pre-selecting an order for support)."""
    st.session_state.page = page
    if order_id:
        st.session_state.support_order = order_id


def render_pipeline(trace: tr.Trace, slot) -> None:
    """Draw the live five-stage tracker into a placeholder."""
    with slot.container():
        for col, (key, label, _icon) in zip(st.columns(len(tr.STAGES)), tr.STAGES):
            s = trace.stages[key]
            col.markdown(f"{tr.STATE_ICON[s['state']]} **{label}**")
            col.caption(f"{s['state'].capitalize()}" + (f" · {s['seconds']}s" if s["seconds"] else ""))


def citation_rows(citations: str) -> list[tuple[str, str, str]]:
    """'S1=refund_policy.md#return-window; S2=...' -> [(label, chunk_id, passage text)]."""
    from rag.retriever import get_retriever
    chunks = {c.chunk_id: c for c in get_retriever().index.chunks}
    rows = []
    for part in filter(None, (p.strip() for p in citations.split(";"))):
        label, _, chunk_id = part.partition("=")
        chunk = chunks.get(chunk_id)
        rows.append((label, chunk_id, chunk.text if chunk else "(passage no longer in the knowledge base)"))
    return rows


def show_sources(citations: str) -> None:
    rows = citation_rows(citations) if citations else []
    if rows:
        with st.expander("Sources used for this answer"):
            for label, chunk_id, text in rows:
                st.markdown(f"**[{label}]** `{chunk_id}`")
                st.caption(text)
