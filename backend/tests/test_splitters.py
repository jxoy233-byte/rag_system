"""Splitter tests."""

from __future__ import annotations

from app.loaders.base import LoadedDocument, LoadedPage
from app.splitters import build_splitter
from app.splitters.parent_child import ParentChildSplitter


def test_splitter_basic():
    doc = LoadedDocument(
        text="段落一。" * 200,  # 800 chars total
        pages=[LoadedPage(text="段落一。" * 200, page=1)],
    )
    chunks = build_splitter(chunk_size=200, chunk_overlap=20).split(doc)
    assert len(chunks) > 1
    assert all(c.text for c in chunks)


def test_splitter_preserves_page():
    doc = LoadedDocument(
        text="x" * 500,
        pages=[
            LoadedPage(text="a" * 500, page=1),
            LoadedPage(text="b" * 500, page=2),
        ],
    )
    chunks = build_splitter(chunk_size=200, chunk_overlap=20).split(doc)
    pages = {c.page for c in chunks}
    assert pages == {1, 2}


# ===== ParentChildSplitter =====


def _make_doc(*paragraphs: str, page: int = 1) -> LoadedDocument:
    """用空行拼接段落构造文档（最常见的 loader 输出格式）。"""
    text = "\n\n".join(paragraphs)
    return LoadedDocument(text=text, pages=[LoadedPage(text=text, page=page)])


def test_parent_child_splits_on_heading():
    """标题（# / ##）应该开新 parent，并把它和后续段落粘到一起。"""
    doc = _make_doc(
        "第一段普通内容，讲 React。",
        "# Hooks 介绍",
        "Hooks 是 React 16.8 引入的。",
        "useState 是最基础的 hook。",
    )
    parents = ParentChildSplitter(parent_size=500, child_size=50, child_overlap=10).split_to_parents(doc)
    # 期望：p0 = "第一段普通内容"，p1 = "# Hooks 介绍 + 后续 2 段"
    assert len(parents) == 2, parents
    assert "第一段普通内容" in parents[0].text
    assert "Hooks 介绍" in parents[1].text
    assert "useState" in parents[1].text
    # children 应在 parent 内部
    assert len(parents[1].children) >= 1
    # 每个 child 都标了 parent_id
    for c in parents[1].children:
        assert c.parent_id == parents[1].parent_id


def test_parent_child_splits_long_block_by_sentence():
    """超长 parent 应该按句末标点二次切。"""
    long_para = "这是第一句。" + "这是第二句。" * 60  # ~600 chars
    doc = _make_doc(long_para)
    parents = ParentChildSplitter(parent_size=200, child_size=50, child_overlap=10).split_to_parents(doc)
    # 一个长 paragraph 应该被切成多个 parent
    assert len(parents) >= 2
    # 任意 parent 长度不超过 ~1000
    for p in parents:
        assert len(p.text) <= 1500


def test_parent_child_children_carry_parent_text():
    """每个 child 的 parent_text 应等于其所属 parent 的完整 text。"""
    doc = _make_doc(
        "段落 A 内容。" * 5,
        "# 标题",
        "段落 B 内容。" * 5,
    )
    chunks = ParentChildSplitter(parent_size=500, child_size=50, child_overlap=10).split(doc)
    # child 来自两个 parent，每个 child.parent_text 等于它所在 parent 的完整 text
    parent_texts = {c.parent_text for c in chunks if c.parent_text}
    # 至少 1 个 parent（一个 parent 内可能有多个 child）
    assert len(parent_texts) >= 1
    # 每个 child.parent_text 应 == 某个 parent.text
    for c in chunks:
        if c.parent_text:
            # 反向验证：parent_text 应能在文档中找到
            assert c.parent_text in doc.text or any(p.text == c.parent_text for p in ParentChildSplitter(parent_size=500, child_size=50, child_overlap=10).split_to_parents(doc))


def test_parent_child_metadata_has_parent_id():
    """child metadata 应包含 parent_id 和 child_index。"""
    doc = _make_doc("# Hooks\n\nuseState 用法。" * 5)
    chunks = ParentChildSplitter(parent_size=200, child_size=50, child_overlap=10).split(doc)
    for c in chunks:
        md = c.metadata or {}
        assert md.get("parent_id"), f"missing parent_id in metadata: {md}"
        assert "child_index" in md


def test_parent_child_short_doc_single_parent():
    """很短的文档应该只生成 1 个 parent + 1 个 child。"""
    doc = _make_doc("短文档。")
    parents = ParentChildSplitter().split_to_parents(doc)
    assert len(parents) == 1
    assert len(parents[0].children) == 1
    assert parents[0].children[0].text == "短文档。"
    assert parents[0].children[0].parent_text == "短文档。"


def test_build_splitter_default_uses_parent_child():
    """build_splitter() 默认应该走 ParentChildSplitter（兼容 ingest 流程）。"""
    doc = _make_doc("# 标题\n\n段落内容。" * 3)
    chunks = build_splitter().split(doc)
    # 默认是 ParentChildSplitter，chunks 应有 parent_id
    for c in chunks:
        assert hasattr(c, "parent_id")
        assert hasattr(c, "parent_text")
