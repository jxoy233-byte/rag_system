"""Loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_text_loader(tmp_path: Path):
    from app.loaders import DocumentLoaderFactory
    p = tmp_path / "a.txt"
    p.write_text("Hello\n\nWorld", encoding="utf-8")
    doc = DocumentLoaderFactory.load(p)
    assert doc.text == "Hello\n\nWorld"
    assert doc.ext == ".txt"


def test_markdown_loader(tmp_path: Path):
    from app.loaders import DocumentLoaderFactory
    p = tmp_path / "a.md"
    p.write_text("# Title\n\nBody paragraph.", encoding="utf-8")
    doc = DocumentLoaderFactory.load(p)
    assert "# Title" in doc.text
    assert any(pg.section == "Title" for pg in doc.pages)


def test_csv_loader(tmp_path: Path):
    from app.loaders import DocumentLoaderFactory
    p = tmp_path / "a.csv"
    p.write_text("name,age\nAlice,30\nBob,25\n", encoding="utf-8")
    doc = DocumentLoaderFactory.load(p)
    assert "name | age" in doc.text
    assert "Alice | 30" in doc.text


def test_factory_unsupported(tmp_path: Path):
    from app.loaders import DocumentLoaderFactory
    p = tmp_path / "a.xyz"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        DocumentLoaderFactory.load(p)
