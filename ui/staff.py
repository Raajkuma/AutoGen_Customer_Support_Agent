"""Staff-only pages: human approvals, workflow trace viewer, ticket log + dashboard."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import tickets
import workflow_trace as tr
from ui.common import show_sources


def approvals_page(reviewer: str) -> None:
    st.header("Human approvals")
    st.caption("Escalated tickets stop here. Review the AI draft, edit it if needed, then send or reject it.")
    df = tickets.read_all()
    pending = df[df["approval"] == "Pending"]
    if pending.empty:
        st.success("Nothing is waiting for review.")
    for _, t in pending.iterrows():
        tid = t["ticket_id"]
        with st.container(border=True):
            st.markdown(f"**{tid}** · {t['customer_name']} · {t['order_id']} · "
                        f"{t['category']} / {t['urgency']} / {t['sentiment']} · confidence {t['confidence']}")
            st.error(f"Escalated because: {t['escalation_reason']}")
            st.markdown("**Complaint**")
            st.info(t["complaint"])
            draft = st.text_area("AI draft reply (editable)", t["response"], key=f"draft_{tid}", height=120)
            show_sources(t["citations"])
            c1, c2, _ = st.columns([1, 1, 3])
            decision = None
            if c1.button("Send reply", key=f"ok_{tid}", type="primary"):
                decision = tickets.review_ticket(tid, reviewer, draft)
            if c2.button("Reject draft", key=f"no_{tid}"):
                decision = tickets.review_ticket(tid, reviewer, "", reject=True)
            if decision:
                tr.record_human_decision(tid, decision["decision"], reviewer, decision["final_response"])
                st.toast(f"{tid}: {decision['decision']}")
                st.rerun()
    done = df[df["approval"].isin(["Approved", "Edited", "Rejected"])]
    if not done.empty:
        with st.expander(f"Reviewed tickets ({len(done)})"):
            st.dataframe(done[["ticket_id", "customer_name", "approval", "reviewed_by", "reviewed_at"]],
                         hide_index=True)


def trace_page() -> None:
    st.header("Workflow trace")
    runs = tr.list_traces()
    if not runs:
        st.info("No runs yet. Submit a complaint on the Support desk.")
        return
    labels = [f"{r['ticket_id'] or 'no ticket'} · {r['created']} · "
              f"{'❌ failed' if any(s['state'] == 'failed' for s in r['stages'].values()) else r['customer']}"
              for r in runs]
    run = runs[labels.index(st.selectbox("Run", labels))]
    stages = run["stages"]
    st.caption(f"Run {run['run_id']} · model {run['model']} · customer {run['customer']} · order {run['order_id']}")
    st.graphviz_chart(tr.to_dot(run))
    st.metric("Total agent time", f"{sum(s['seconds'] or 0 for s in stages.values()):.1f}s")
    st.subheader("Stages")
    for key, label, icon in tr.STAGES:
        s = stages[key]
        with st.expander(f"{tr.STATE_ICON[s['state']]} {icon} {label} · {s['state']}"
                         + (f" · {s['seconds']}s" if s["seconds"] else "")):
            if s["output"]:
                st.text(s["output"])
            for t in s["tools"]:
                st.markdown(f"🔧 `{t['name']}`")
                st.code(json.dumps(t["args"], indent=2, ensure_ascii=False), language="json")
                st.text(t["result"])
            if s["detail"]:
                st.json(s["detail"])
    st.subheader("Event timeline")
    st.dataframe(pd.DataFrame(run["events"]), hide_index=True)
    st.download_button("Download trace JSON", json.dumps(run, indent=2), f"trace_{run['run_id']}.json",
                       "application/json")


def log_page(customer: str, staff: bool) -> None:
    st.header("Ticket log")
    df = tickets.read_all()
    if not staff:
        df = df[df["customer_name"] == customer]
    if df.empty:
        st.info("No tickets yet.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickets", len(df))
    c2.metric("Resolved", int((df["status"] == "Resolved").sum()))
    c3.metric("Escalated", int((df["status"] == "Escalated").sum()))
    c4.metric("Awaiting approval", int((df["approval"] == "Pending").sum()))
    if staff:
        left, right = st.columns(2)
        left.caption("Tickets per category")
        left.bar_chart(df["category"].value_counts())
        right.caption("Resolved vs escalated")
        right.bar_chart(df["status"].value_counts())
    shown = df if staff else df.drop(columns=["escalation_reason", "response", "reviewed_by"])
    st.dataframe(shown.iloc[::-1], hide_index=True)
    st.download_button("Download CSV", shown.to_csv(index=False).encode("utf-8"), "tickets.csv", "text/csv")
