"""Chunk 端点 / ChromaStore.get() 单元测试。"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest


def test_chroma_get_helper(tmp_path: Path):
    """ChromaStore.get(ids / where) 返回 [{id, text, metadata}, ...]。"""
    from app.core.config import get_settings

    get_settings.cache_clear()
    import os
    os.environ["CHROMA_PERSIST_DIR"] = str(tmp_path / "chroma")
    os.environ["EMBEDDING_PROVIDER"] = "mock"  # 避免拉真实 embedding
    get_settings.cache_clear()

    from app.embeddings.factory import EmbeddingFactory
    from app.vectorstore import ChromaStore

    emb = EmbeddingFactory.get(model_name="mock", dim=8)
    store = ChromaStore(collection_name="test_chunks", embedding=emb)

    ids = [str(uuid.uuid4()) for _ in range(3)]
    texts = ["alpha bravo", "charlie delta", "echo foxtrot"]
    metas = [
        {"doc_id": 1, "page": 1, "section": "intro", "chunk_index": 0},
        {"doc_id": 1, "page": 1, "section": "body", "chunk_index": 1},
        {"doc_id": 1, "page": 2, "section": "body", "chunk_index": 2},
    ]
    store.add(ids=ids, texts=texts, metadatas=metas)

    # 按 ids 查
    rows = store.get(ids=ids)
    assert len(rows) == 3
    assert {r["id"] for r in rows} == set(ids)
    assert {r["text"] for r in rows} == set(texts)
    assert all(r["metadata"]["doc_id"] == 1 for r in rows)

    # 按 where 查
    rows2 = store.get(where={"doc_id": 1})
    assert len(rows2) == 3
    rows_p2 = store.get(where={"page": 2})
    assert len(rows_p2) == 1
    assert rows_p2[0]["text"] == "echo foxtrot"

    # 不存在的 ids → 空列表
    assert store.get(ids=["nonexistent"]) == []


def test_chunk_schemas_basic():
    """ChunkDetail / ChunkListItem 字段与默认值正确。"""
    from app.schemas.chunk import ChunkDetail, ChunkListItem

    detail = ChunkDetail(
        chunk_id="abc",
        doc_id=1,
        kb_id=2,
        text="hello world",
        page=3,
        section="intro",
        document="doc.pdf",
    )
    assert detail.chunk_id == "abc"
    assert detail.source_type == "vector"  # default
    assert detail.score is None

    item = ChunkListItem(
        chunk_id="abc",
        doc_id=1,
        kb_id=2,
        length=11,
        preview="hello world",
        page=3,
        document="doc.pdf",
        chunk_index=0,
    )
    assert item.length == 11
    assert item.chunk_index == 0


def test_chunk_endpoints_registered():
    """两条 chunk 路由必须在 FastAPI app 中。"""
    from app.api.v1.documents import router

    paths = [r.path for r in router.routes if hasattr(r, "methods")]
    assert any(p.endswith("/{doc_id}/chunks") and "GET" in r.methods for p, r in zip(paths, router.routes) if hasattr(r, "methods"))
    assert any("/{doc_id}/chunks/{chunk_id}" in p for p in paths)


def test_list_chunks_for_unknown_doc(tmp_path: Path, monkeypatch):
    """不存在的 doc → 200 + 空列表（前端会显示「暂无切片」）。"""
    # 这个端点真正打 Chroma + DB，单测里不集成。改为验证「Chroma 查空集合」路径。
    from app.core.config import get_settings

    get_settings.cache_clear()
    import os
    os.environ["CHROMA_PERSIST_DIR"] = str(tmp_path / "chroma")
    os.environ["EMBEDDING_PROVIDER"] = "mock"
    get_settings.cache_clear()

    from app.embeddings.factory import EmbeddingFactory
    from app.vectorstore import ChromaStore

    emb = EmbeddingFactory.get(model_name="mock", dim=8)
    store = ChromaStore(collection_name="empty_coll", embedding=emb)
    rows = store.get(where={"doc_id": 999})
    assert rows == []
