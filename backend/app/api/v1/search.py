"""检索路由：纯向量 + BM25 混合 + Rerank，直接返回 sources。"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db
from app.models import KnowledgeBase
from app.schemas.chat import SourceItem
from app.schemas.search import SearchRequest, SearchResponse
from app.services.retriever import HybridRetriever

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(
    payload: SearchRequest,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> SearchResponse:
    if not payload.knowledge_base_id:
        raise HTTPException(status_code=400, detail="knowledge_base_id is required")
    kb = await session.get(KnowledgeBase, payload.knowledge_base_id)
    if not kb:
        raise HTTPException(status_code=404, detail="knowledge_base not found")

    t0 = time.time()
    retriever = HybridRetriever(
        knowledge_base_id=kb.id,
        collection_name=kb.collection_name,
        rerank=payload.use_rerank,
        embedding_model=kb.embedding_model,
        embedding_dim=kb.embedding_dim,
    )
    chunks = await retriever.retrieve(payload.query, top_k=payload.top_k)
    sources: list[SourceItem] = []
    for c in chunks:
        md = c.metadata or {}
        sources.append(
            SourceItem(
                kb_id=kb.id,
                doc_id=int(md.get("doc_id")) if md.get("doc_id") is not None else None,
                document=md.get("doc_title") or md.get("doc_filename") or "?",
                page=md.get("page"),
                chunk_id=c.id,
                snippet=(c.text or "")[:300],
                score=c.rerank_score if c.rerank_score is not None else c.score,
                source_type=("bm25" if c.source == "bm25" else "vector"),
                url=None,
            )
        )
    return SearchResponse(
        sources=sources,
        latency_ms=int((time.time() - t0) * 1000),
    )
