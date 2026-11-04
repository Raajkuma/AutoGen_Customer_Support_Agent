import pytest

from rag import citations
from rag.bm25 import BM25
from rag.chunker import chunk_markdown, load_chunks
from rag.index import build_index
from rag.retriever import Retriever, rrf_fuse
from rag.text import slugify, tokenize
from tests.helpers import DownEmbedder, HashEmbedder

DOC = "# T\n\n## Return window\nItems can be returned within 30 days.\n\n## Empty\n\n## Long section\n" + " ".join(f"w{i}" for i in range(50))


def test_chunk_ids_are_stable_and_empty_sections_skipped():
    chunks = chunk_markdown("refund.md", DOC, size=100, overlap=10)
    assert [c.chunk_id for c in chunks] == ["refund.md#return-window", "refund.md#long-section"]
    assert chunks == chunk_markdown("refund.md", DOC, size=100, overlap=10)


def test_long_sections_become_overlapping_windows():
    chunks = chunk_markdown("a.md", DOC, size=20, overlap=5)
    windows = [c for c in chunks if c.chunk_id.startswith("a.md#long-section")]
    assert len(windows) >= 3 and windows[1].chunk_id.endswith("~2")
    assert set(windows[0].text.split()[-5:]) <= set(windows[1].text.split())   # overlap carried over


def test_chunker_rejects_bad_params():
    with pytest.raises(ValueError):
        chunk_markdown("a.md", DOC, size=10, overlap=10)


def test_real_kb_loads_with_unique_ids():
    ids = [c.chunk_id for c in load_chunks()]
    assert len(ids) == len(set(ids)) >= 16


def test_tokenizer_keeps_tamil_words_intact_and_normalises_concepts():
    assert tokenize("கடவுச்சொல்லை மறந்துவிட்டேன்") == ["கடவுச்சொல்லை", "மறந்துவிட்டேன்"]
    assert tokenize("returned shoes") == tokenize("refund shoe")
    assert slugify("Change email or phone number") == "change-email-or-phone-number"


def test_bm25_prefers_matching_document():
    bm = BM25([["refund", "days"], ["password", "reset"], ["ship", "delay"]])
    scores = bm.scores(["password"])
    assert scores.index(max(scores)) == 1 and scores[0] == 0


def test_rrf_fusion_math():
    fused = rrf_fuse([[1, 2, 3], [3, 1]], k=60)
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert max(fused, key=fused.get) == 1 and fused[3] == pytest.approx(1 / 63 + 1 / 61)


@pytest.fixture
def bm25_retriever(tmp_path):
    return Retriever(build_index(None, index_dir=tmp_path / "i"))


@pytest.mark.parametrize("query,category,expected", [
    ("I forgot my password and can't log in.", "Account", "account_and_login.md#reset-your-password"),
    ("The blender I received is broken. Can I get a replacement?", "Product", "product_faq.md#damaged-or-defective-products"),
    ("How many days does standard delivery take to Mumbai?", "Shipping", "shipping_policy.md#delivery-timelines"),
])
def test_retrieval_finds_expected_chunk(bm25_retriever, query, category, expected):
    result = bm25_retriever.search(query, category)
    assert result.matched and expected in result.chunk_ids
    assert result.format().startswith("[S1] ")


def test_no_match_for_unknown_topic(bm25_retriever):
    result = bm25_retriever.search("Do you sell gift cards for 500 employees with custom branding?", "Other")
    assert not result.matched and result.format().startswith("NO_MATCH") and result.chunk_ids == []


def test_hybrid_mode_and_incremental_index(tmp_path):
    emb = HashEmbedder()
    first = build_index(emb, index_dir=tmp_path / "i")
    assert first.stats["embedded"] == first.stats["chunks"] and first.matrix.shape[0] == len(first.chunks)
    calls = emb.calls
    second = build_index(emb, index_dir=tmp_path / "i")        # nothing changed: nothing re-embedded
    assert second.stats["embedded"] == 0 and second.stats["reused"] == len(second.chunks) and emb.calls == calls
    assert (tmp_path / "i" / "manifest.json").exists() and (tmp_path / "i" / "chunks.jsonl").exists()
    result = Retriever(second, emb).search("forgot password reset", "Account")
    assert result.mode == "hybrid" and result.hits[0].dense_rank is not None


def test_changed_chunk_is_reembedded_only(tmp_path, monkeypatch):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "a.md").write_text("# A\n\n## One\nalpha text\n\n## Two\nbeta text\n", encoding="utf-8")
    emb = HashEmbedder()
    build_index(emb, kb_dir=kb, index_dir=tmp_path / "i")
    (kb / "a.md").write_text("# A\n\n## One\nalpha text\n\n## Two\nbeta CHANGED text\n", encoding="utf-8")
    idx = build_index(emb, kb_dir=kb, index_dir=tmp_path / "i")
    assert idx.stats["embedded"] == 1 and idx.stats["reused"] == 1


def test_dense_failure_degrades_to_bm25(tmp_path):
    idx = build_index(DownEmbedder(), index_dir=tmp_path / "i")
    assert idx.matrix is None and "dense_error" in idx.stats
    result = Retriever(idx, DownEmbedder()).search("forgot password", "Account")
    assert result.mode.startswith("bm25-only") and result.matched


PASSAGES = "[S1] refund_policy.md#return-window\nItems can be returned within 30 days.\n\n[S2] a.md#b\nOther."


def test_parse_passages_and_citation_check():
    passages = citations.parse_passages(PASSAGES)
    assert passages["S1"]["chunk_id"] == "refund_policy.md#return-window" and set(passages) == {"S1", "S2"}
    check = citations.check_citations("You can return items within 30 days [S1]. See also [S9].", passages)
    assert check.cited == ["S1"] and check.invalid == ["S9"] and "[S9]" not in check.reply
    assert citations.citation_map(check.cited, passages) == "S1=refund_policy.md#return-window"


def test_uncited_substantive_sentences_are_flagged():
    passages = citations.parse_passages(PASSAGES)
    check = citations.check_citations("Thanks for writing. You may return any item within thirty days of delivery.", passages)
    assert check.cited == [] and len(check.uncited_sentences) == 1


def test_weak_tail_passages_are_not_returned(bm25_retriever):
    result = bm25_retriever.search("You charged my card three times! I'm calling my lawyer.", "Billing")
    assert result.chunk_ids == ["product_faq.md#payments-and-billing"]
