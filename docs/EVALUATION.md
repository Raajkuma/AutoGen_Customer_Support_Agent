# Evaluation

Reproduce: `python -m evaluation.run_eval` (offline, no model) · `--dense` (hybrid, needs the embedding model) ·
`--e2e` (full agent team on the local LLM) · `--questions evaluation/questions_holdout.jsonl`.

## Method and honesty notes
- `questions.jsonl`: 26 labelled questions (23 English, 3 Tamil): the six assignment cases, plus unit-style cases for each
  KB section, a prompt-injection attempt, a vague one-liner, an angry-but-simple complaint and a "bank" false-positive trap.
  Each has relevant chunk ids, gold category/urgency/sentiment and the expected ticket status.
- **The relevance gate thresholds were tuned while looking at this set, so its scores are optimistic.** It is used as a
  CI regression gate, not as an accuracy claim.
- `questions_holdout.jsonl`: 12 new questions written before any scoring. First run: recall 0.70. After that I fixed one
  real bug (the weak-tail cut was propped up by the routing prior) and re-ran: identical numbers. No thresholds or
  synonyms were changed in response to holdout failures.
- Metrics: recall@k and MRR over retrieved chunk ids (`evaluation/metrics.py`, exercised by tests and by the runner);
  match-decision accuracy (did we correctly decide a relevant passage exists?); escalation precision/recall/F1 using the
  production rule function with gold classification.

## Results, offline lexical mode (BM25 + rerank, no embeddings), measured in the build sandbox
| Set | n | Recall@3 | MRR | Match-decision acc. | Escalation P / R / F1 |
|---|---|---|---|---|---|
| Dev (tuned on) | 23 | 1.000 | 0.950 | 1.000 | 1.00 / 1.00 / 1.00 |
| **Holdout (untouched)** | 12 | **0.700** | **0.633** | **0.750** | **0.60 / 0.75 / 0.67** |

## Holdout failures (why dense retrieval exists)
| Id | Question | Failure | Cause |
|---|---|---|---|
| h02 | "Can I send back a jacket if I don't like the fit?" | no match | no shared vocabulary with the return policy |
| h12 | "How long is the guarantee on the backpack?" | no match | "guarantee" vs "warranty" |
| h05 | "Money was taken twice ... police complaint" | wrong chunk | still escalated correctly by the legal rule |
| h10 | "Do you offer an instalment plan or EMI?" | false match | generic word overlap; needs a calibrated dense gate |

## Not yet measured (run these yourself, then paste the numbers here)
- `python -m evaluation.run_eval --dense` — hybrid recall/MRR and Tamil questions. Use it to calibrate `DENSE_MIN`.
  Target: recall@3 >= 0.90 and MRR >= 0.85 on the holdout.
- `python -m evaluation.run_eval --e2e` — status accuracy and citation rate with the real model.
