"""Retriever tests: parent_collapse 步骤。"""

from __future__ import annotations

from app.services.retriever import HybridRetriever, RetrievedChunk


def _make_chunk(
    cid: str,
    text: str,
    parent_id: str,
    child_id: str = "",
    score: float = 0.5,
    rerank_score: float | None = None,
    parent_text: str = "",
) -> RetrievedChunk:
    return RetrievedChunk(
        id=cid,
        text=text,
        metadata={
            "doc_id": 1,
            "parent_id": parent_id,
            "child_index": 0,
            "child_id": child_id or cid,
            "parent_text": parent_text or text,
        },
        score=score,
        rerank_score=rerank_score,
        source="vector",
    )


def test_parent_collapse_merges_same_parent():
    """同一个 parent 的多个 child 应该合并成 1 个 parent 块，保留分数最高那个。"""
    r = HybridRetriever(knowledge_base_id=1, collection_name="test", rerank=False)
    chunks = [
        _make_chunk("c1", "child 1 text", parent_id="p1", score=0.8, rerank_score=0.9, parent_text="PARENT FULL"),
        _make_chunk("c2", "child 2 text", parent_id="p1", score=0.7, rerank_score=0.7, parent_text="PARENT FULL"),
        _make_chunk("c3", "child 3 text", parent_id="p1", score=0.6, rerank_score=0.5, parent_text="PARENT FULL"),
    ]
    out = r._parent_collapse(chunks)
    # 3 child 合并为 1
    assert len(out) == 1
    # text 应是 parent_text
    assert out[0].text == "PARENT FULL"
    # source 标记为 parent
    assert out[0].source == "parent"
    # rerank_score 取最高的（0.9）
    assert out[0].rerank_score == 0.9
    # id 是 parent: 前缀
    assert out[0].id.startswith("parent:")
    assert out[0].id.endswith("p1")
    # metadata.is_parent = True
    assert out[0].metadata.get("is_parent") is True


def test_parent_collapse_keeps_different_parents():
    """不同 parent 的 child 应该保留为多个。"""
    r = HybridRetriever(knowledge_base_id=1, collection_name="test", rerank=False)
    chunks = [
        _make_chunk("c1", "child A", parent_id="p1", score=0.8, parent_text="PARENT 1"),
        _make_chunk("c2", "child B", parent_id="p2", score=0.7, parent_text="PARENT 2"),
        _make_chunk("c3", "child C", parent_id="p1", score=0.5, parent_text="PARENT 1"),
    ]
    out = r._parent_collapse(chunks)
    assert len(out) == 2
    parent_texts = {c.text for c in out}
    assert parent_texts == {"PARENT 1", "PARENT 2"}


def test_parent_collapse_picks_highest_score_child():
    """同 parent 取 rerank_score 最高的 child 作为锚点。"""
    r = HybridRetriever(knowledge_base_id=1, collection_name="test", rerank=False)
    chunks = [
        _make_chunk("c1", "low", parent_id="p1", rerank_score=0.2, parent_text="P1 TEXT"),
        _make_chunk("c2", "high", parent_id="p1", rerank_score=0.95, parent_text="P1 TEXT"),
        _make_chunk("c3", "mid", parent_id="p1", rerank_score=0.5, parent_text="P1 TEXT"),
    ]
    out = r._parent_collapse(chunks)
    assert len(out) == 1
    # 锚点是 c2
    assert out[0].metadata["child_id"] == "c2"
    assert out[0].rerank_score == 0.95


def test_parent_collapse_preserves_no_parent_chunks():
    """没有 parent_id 的老 chunk 应保留原样，不参与合并。"""
    r = HybridRetriever(knowledge_base_id=1, collection_name="test", rerank=False)
    chunks = [
        _make_chunk("c1", "child", parent_id="p1", parent_text="P1 TEXT"),
        RetrievedChunk(id="old", text="legacy chunk", metadata={}, score=0.6),
    ]
    out = r._parent_collapse(chunks)
    assert len(out) == 2
    by_id = {c.id: c for c in out}
    # 老 chunk id 不变
    assert "old" in by_id
    assert by_id["old"].text == "legacy chunk"


def test_parent_collapse_empty():
    """空输入应返回空。"""
    r = HybridRetriever(knowledge_base_id=1, collection_name="test", rerank=False)
    assert r._parent_collapse([]) == []


def test_parent_collapse_sorts_by_score():
    """合并后应按 rerank_score 降序。"""
    r = HybridRetriever(knowledge_base_id=1, collection_name="test", rerank=False)
    chunks = [
        _make_chunk("c1", "A", parent_id="p1", rerank_score=0.3, parent_text="P1"),
        _make_chunk("c2", "B", parent_id="p2", rerank_score=0.9, parent_text="P2"),
        _make_chunk("c3", "C", parent_id="p3", rerank_score=0.6, parent_text="P3"),
    ]
    out = r._parent_collapse(chunks)
    scores = [c.rerank_score for c in out]
    assert scores == sorted(scores, reverse=True)