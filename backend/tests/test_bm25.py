"""BM25 store tests."""

from __future__ import annotations

from app.services.bm25_store import BM25Doc, BM25Store


def test_bm25_basic(tmp_path):
    from app.core.config import get_settings
    get_settings.cache_clear()
    import os
    os.environ["BM25_INDEX_DIR"] = str(tmp_path)
    store = BM25Store(999)
    store.add(
        [
            BM25Doc(chunk_id="c1", text="苹果是一种水果"),
            BM25Doc(chunk_id="c2", text="我喜欢吃香蕉"),
            BM25Doc(chunk_id="c3", text="机器学习是人工智能的分支"),
        ]
    )
    hits = store.query("水果")
    assert len(hits) >= 1
    assert hits[0][0].chunk_id == "c1"


def test_bm25_delete(tmp_path):
    from app.core.config import get_settings
    get_settings.cache_clear()
    import os
    os.environ["BM25_INDEX_DIR"] = str(tmp_path)
    store = BM25Store(1000)
    store.add([BM25Doc(chunk_id="c1", text="x", metadata={"doc_id": 1})])
    removed = store.delete_by_doc_id(1)
    assert removed == 1
    assert len(store) == 0
