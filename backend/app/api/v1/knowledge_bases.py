"""知识库 CRUD 路由。"""

from __future__ import annotations

import re
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query, status
from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends

from app.core.config import get_settings
from app.core.deps import get_db
from app.models import Document, DocumentStatus, KnowledgeBase
from app.schemas.knowledge_base import (
    KnowledgeBaseCreate,
    KnowledgeBaseRead,
    KnowledgeBaseUpdate,
)
from app.services.bm25_store import BM25Store
from app.vectorstore import ChromaStore

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])

KbId = Annotated[int, Path(ge=1)]
DbSession = Annotated[AsyncSession, Depends(get_db)]


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower()).strip("_")
    if not s:
        s = "kb"
    return f"{s}_{abs(hash(name)) % 10_000_000:07d}"


def _new_collection_name(slug: str) -> str:
    return f"kb_{slug}".replace("-", "_")


@router.get("", response_model=list[KnowledgeBaseRead])
async def list_knowledge_bases(
    session: DbSession,
    q: str | None = Query(default=None, description="模糊匹配名称/描述"),
) -> list[KnowledgeBaseRead]:
    stmt = select(KnowledgeBase).order_by(KnowledgeBase.id.desc())
    if q:
        like = f"%{q}%"
        stmt = stmt.where(KnowledgeBase.name.like(like) | KnowledgeBase.description.like(like))
    res = await session.execute(stmt)
    items = res.scalars().all()
    return [KnowledgeBaseRead.model_validate(x) for x in items]


@router.post("", response_model=KnowledgeBaseRead, status_code=status.HTTP_201_CREATED)
async def create_knowledge_base(payload: KnowledgeBaseCreate, session: DbSession) -> KnowledgeBaseRead:
    settings = get_settings()
    kb = KnowledgeBase(
        name=payload.name,
        slug=_slugify(payload.name),
        description=payload.description,
        collection_name=_new_collection_name(_slugify(payload.name)),
        embedding_model=(
            payload.embedding_model if payload.embedding_model else None
        ),
        embedding_dim=(
            payload.embedding_dim if payload.embedding_dim is not None else None
        ),
        chunk_size=(
            payload.chunk_size if payload.chunk_size is not None else settings.chunk_size
        ),
        chunk_overlap=(
            payload.chunk_overlap
            if payload.chunk_overlap is not None
            else settings.chunk_overlap
        ),
    )
    session.add(kb)
    try:
        await session.flush()
    except IntegrityError as e:
        await session.rollback()
        raise HTTPException(status_code=409, detail=f"知识库名称已存在: {payload.name}") from e
    await session.refresh(kb)
    # 预创建 collection，避免首次入库延迟
    try:
        ChromaStore(collection_name=kb.collection_name).collection
    except Exception as e:
        logger.warning("pre-create chroma collection failed: {}", e)
    return KnowledgeBaseRead.model_validate(kb)


@router.get("/{kb_id}", response_model=KnowledgeBaseRead)
async def get_knowledge_base(kb_id: KbId, session: DbSession) -> KnowledgeBaseRead:
    kb = await session.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="knowledge_base not found")
    return KnowledgeBaseRead.model_validate(kb)


@router.patch("/{kb_id}", response_model=KnowledgeBaseRead)
async def update_knowledge_base(
    kb_id: KbId, payload: KnowledgeBaseUpdate, session: DbSession
) -> KnowledgeBaseRead:
    kb = await session.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="knowledge_base not found")
    data = payload.model_dump(exclude_unset=True)
    chunk_size = data.get("chunk_size", kb.chunk_size)
    chunk_overlap = data.get("chunk_overlap", kb.chunk_overlap)
    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=422,
            detail="chunk_overlap must be smaller than chunk_size",
        )

    embedding_dirty = (
        "embedding_model" in data or "embedding_dim" in data
    )
    chunk_count = (
        ChromaStore(collection_name=kb.collection_name).count()
        if embedding_dirty
        else 0
    )
    if embedding_dirty and chunk_count:
        raise HTTPException(
            status_code=409,
            detail="知识库已有向量，修改 embedding 配置前请先删除现有文档",
        )
    for k, v in data.items():
        setattr(kb, k, v)
    try:
        await session.flush()
    except IntegrityError as e:
        await session.rollback()
        raise HTTPException(status_code=409, detail="name conflict") from e
    await session.refresh(kb)
    return KnowledgeBaseRead.model_validate(kb)


@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_knowledge_base(kb_id: KbId, session: DbSession) -> None:
    from app.services.ingest import IngestService

    kb = (
        await session.execute(
            select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
        )
    ).scalar_one_or_none()
    if not kb:
        raise HTTPException(status_code=404, detail="knowledge_base not found")
    await IngestService(session).delete_kb(kb)


@router.get("/{kb_id}/stats")
async def knowledge_base_stats(kb_id: KbId, session: DbSession) -> dict:
    kb = await session.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="knowledge_base not found")
    doc_total = await session.scalar(
        select(func.count(Document.id)).where(Document.knowledge_base_id == kb_id)
    )
    doc_ready = await session.scalar(
        select(func.count(Document.id)).where(
            Document.knowledge_base_id == kb_id, Document.status == DocumentStatus.ready
        )
    )
    doc_failed = await session.scalar(
        select(func.count(Document.id)).where(
            Document.knowledge_base_id == kb_id, Document.status == DocumentStatus.failed
        )
    )
    bm25_len = len(BM25Store.for_kb(kb_id))
    chroma_count = ChromaStore(collection_name=kb.collection_name).count()
    return {
        "kb_id": kb_id,
        "doc_total": int(doc_total or 0),
        "doc_ready": int(doc_ready or 0),
        "doc_failed": int(doc_failed or 0),
        "bm25_chunks": bm25_len,
        "chroma_chunks": chroma_count,
        "settings": {
            "chunk_size": kb.chunk_size,
            "chunk_overlap": kb.chunk_overlap,
            "embedding_model": kb.embedding_model,
            "embedding_dim": kb.embedding_dim,
        },
    }
