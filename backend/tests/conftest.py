"""Pytest fixtures."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import pytest
import pytest_asyncio


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def tmp_db(tmp_path) -> AsyncIterator[str]:
    """Override DATA_DIR to a temp dir for isolation."""
    import os
    os.environ["DATA_DIR"] = str(tmp_path)
    os.environ["SQLITE_PATH"] = str(tmp_path / "metadata.db")
    os.environ["CHROMA_PERSIST_DIR"] = str(tmp_path / "chroma")
    os.environ["BM25_INDEX_DIR"] = str(tmp_path / "bm25")
    os.environ["UPLOAD_DIR"] = str(tmp_path / "uploads")
    # reload settings cache
    from app.core.config import get_settings
    get_settings.cache_clear()
    yield str(tmp_path)


@pytest.fixture
def sample_text() -> str:
    return (
        "RAG (Retrieval-Augmented Generation) combines a retrieval system with "
        "a generative LLM. It first retrieves relevant documents from a knowledge "
        "base, then feeds them as context to the LLM for answer generation.\n\n"
        "Hybrid retrieval combines BM25 (keyword) and dense vector search for "
        "better recall. Reciprocal Rank Fusion (RRF) is a common merge strategy.\n\n"
        "Re-ranking with a cross-encoder (e.g., BGE-Reranker) further improves "
        "precision by scoring query-document pairs jointly."
    )
