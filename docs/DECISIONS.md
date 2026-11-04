# Design decisions

1. **Models do language work only; agents invoke the tools.** Knowledge and Log call AutoGen `FunctionTool`s themselves
   (`tools.py`) instead of asking the model to emit a tool call. A 2B local model skips or malforms tool calls, which
   silently breaks retrieval and logging. Query and Resolution use the model. Trade-off: the model never chooses to search.
2. **Escalation is code, not prompt.** `tickets.escalation_reasons` applies the six rules and is unit-tested including
   false positives ("I bank with HDFC"). A model slip cannot hide a risky ticket.
3. **Human approval is a persisted gate, not a blocking agent.** Streamlit reruns the script on every interaction, so a
   run that waits on a human would be lost. The Log agent stores the ticket as `Pending` and ends the run; staff decide
   later on the Approvals page; the trace is updated. AutoGen's `UserProxyAgent`/handoff pattern is the alternative if
   the run must stay in memory.
4. **Hybrid retrieval with a numpy index, not FAISS.** The KB has tens of chunks; exact cosine in numpy is simpler,
   dependency-free and exact. Swap in FAISS behind `Retriever._dense_rank` when the KB grows past ~100k chunks.
5. **Relevance is gated by evidence, and fails closed.** No evidence means `NO_MATCH`, a fixed honest reply, and
   escalation. Category routing (`knowledge_base/routing.json`) lowers the evidence needed inside the right document.
6. **Heuristic reranker.** Coverage + dense similarity + routing prior. A cross-encoder (bge-reranker-v2-m3) is the
   intended upgrade where hardware allows; `rag/rerank.py` is the single swap point.
7. **SQLite for tickets with an atomic CSV export.** Concurrent writes are safe (the original CSV used a thread lock
   only) and the assignment's log file still exists at `logs/tickets.csv`.
8. **Staff access via a shared PIN** (constant-time comparison, throttled). Adequate for a demo; production needs real
   identity (SSO) and per-user audit.
9. **No ticket text in application logs** (PII); ids and metrics only.
