"""Splitter tests."""

from __future__ import annotations

from app.loaders.base import LoadedDocument, LoadedPage
from app.splitters import build_splitter


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
