# ShopEagle: self-assessment against the assignment rubric

Scored conservatively. Marks only count what was actually exercised; the gaps are listed so you can verify or close them.

| Area | Max | Score | Evidence | Deductions |
|---|---|---|---|---|
| Agent design | 20 | **17** | Four AutoGen agents, one job each, strict output formats, AutoGen `FunctionTool`s for search and logging, per-stage tracing, 6 scripted scenarios verified | -2 never run against real AutoGen / qwen3-vl; -1 Knowledge and Log invoke tools directly instead of the model choosing them (deliberate, see ADR 1) |
| Knowledge grounding | 20 | **16** | Hybrid BM25 + dense with RRF (k=60), incremental persisted index with sha256 manifest, reranker, `[S1]` citations validated in code, number-grounding guard, honest no-match path, labelled eval + metrics actually called | -2 dense path never measured (no embedding model in the build sandbox); -2 holdout lexical recall 0.70; reranker is heuristic, not a cross-encoder |
| Workflow and escalation | 20 | **18** | RoundRobin team, two termination words + message cap, six escalation rules in code with unit tests including false positives, persisted human-approval gate with edit/reject, trace updated after review | -2 approval is not an in-memory AutoGen handoff; live run unverified |
| Logging | 15 | **14** | SQLite with atomic CSV export, all assignment columns plus approval and citation fields, autoincrement ids, tests | -1 no Excel export |
| Streamlit UI | 15 | **12** | Live tracker, step expanders, trace viewer with flow graph, approvals page, role-gated pages, dashboard, login throttle, submit cooldown | -3 not rendered in a browser here; no screenshots |
| Code and README | 10 | **8** | Modular layout, typed new code, 43 tests (39 executed here), CI, Docker, pinned ranges, ADRs, evaluation doc | -1 ruff/mypy/coverage never executed here; -1 screenshots missing |
| **Total** | 100 | **85** | | |

## Where the remaining ~10 points come from (only you can do these)
| Action | Points | How |
|---|---|---|
| Run the app once against the real stack and fix anything AutoGen/Streamlit version-specific | +3 | `streamlit run app.py`, submit the 6 assignment complaints |
| Capture the screenshots listed in `screenshots/README.md` | +3 | UI + README completeness |
| Run `python -m evaluation.run_eval --dense` (and `--e2e`), calibrate `DENSE_MIN`, paste numbers into `docs/EVALUATION.md` | +3 | needs an embedding model in Docker Model Runner |
| Run `ruff check .`, `mypy`, `pytest --cov` and fix findings | +1 | CI does this on push |

Realistic range after those: **90-95**. Reaching 95+ also depends on dense recall@3 >= 0.90 on the holdout and a clean first run.

## What was verified in the build sandbox
- 39 unit tests pass (retrieval, chunking, RRF, incremental indexing, dense fallback, citations, tickets, escalation, approval flow, parsing, trace, metrics, auth, eval regression gate).
- 6 end-to-end agent scenarios pass with scripted model output and the real retriever and ticket store: cited reply resolves; missing citation retries; never-cited, invented-number and legal-threat cases escalate to the human gate; no-match skips the model.
- Offline evaluation: dev set (tuned) recall@3 1.00 / MRR 0.95; holdout (untouched) 0.70 / 0.63.

## What was NOT verified
Real AutoGen, Streamlit, qwen3-vl, embedding model, ruff, mypy, coverage, Docker build, GitHub Actions.
