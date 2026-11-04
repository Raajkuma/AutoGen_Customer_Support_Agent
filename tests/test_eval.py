"""Regression gate: retrieval + escalation quality on the labelled dev set must not drop."""
import config
from evaluation.run_eval import evaluate_retrieval, load_questions
from rag.index import build_index
from rag.retriever import Retriever


def test_offline_quality_gate(tmp_path):
    questions = [q for q in load_questions() if q["lang"] == "en"]
    report = evaluate_retrieval(Retriever(build_index(None, index_dir=tmp_path)), questions, config.RERANK_TOP_N)
    assert len(questions) >= 20
    assert report["recall@k"] >= 0.9 and report["mrr"] >= 0.85
    assert report["escalation"]["f1"] >= 0.9 and report["match_decision_accuracy"] >= 0.9
