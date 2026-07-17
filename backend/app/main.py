"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api.v1 import api_router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.models.base import Base
from app.core.db import engine

logger = get_logger(__name__)


async def _run_inline_migrations() -> None:
    """一次性内联迁移：补齐 SQLAlchemy create_all 不会自动加的列。

    目前处理：
    - conversations.summary（2026-07 滚动摘要功能引入）
    - documents.summary（2026-07 doc-level 预筛选引入）
    - messages.doc_hits_json（2026-07 doc-level 命中展示引入）
    """
    async with engine.begin() as conn:
        # conversations.summary
        cols = await conn.execute(text("PRAGMA table_info(conversations)"))
        names = {row[1] for row in cols.fetchall()}
        if "summary" not in names:
            logger.info("migration: adding conversations.summary column")
            await conn.execute(
                text(
                    "ALTER TABLE conversations ADD COLUMN summary TEXT NOT NULL DEFAULT ''"
                )
            )

        # documents.summary（doc-level 检索预筛选）
        cols = await conn.execute(text("PRAGMA table_info(documents)"))
        names = {row[1] for row in cols.fetchall()}
        if "summary" not in names:
            logger.info("migration: adding documents.summary column")
            await conn.execute(
                text("ALTER TABLE documents ADD COLUMN summary TEXT NOT NULL DEFAULT ''")
            )
        if "parent_count" not in names:
            logger.info("migration: adding documents.parent_count column")
            await conn.execute(
                text("ALTER TABLE documents ADD COLUMN parent_count INTEGER NOT NULL DEFAULT 0")
            )

        # messages.doc_hits_json（doc-level 命中持久化）
        cols = await conn.execute(text("PRAGMA table_info(messages)"))
        names = {row[1] for row in cols.fetchall()}
        if "doc_hits_json" not in names:
            logger.info("migration: adding messages.doc_hits_json column")
            await conn.execute(
                text("ALTER TABLE messages ADD COLUMN doc_hits_json TEXT")
            )


async def _warmup_reranker() -> None:
    """启动时把 reranker 模型下载并加载到内存。失败也不阻塞启动，
    让 chat 时再重试，避免用户首次对话时还要等几十分钟下载。

    HF_ENDPOINT 由 app.core.local_model 内部的 resolve_local_path 负责 export，
    这里不再重复处理；只负责按 provider 决定是否走 reranker 预热。
    """
    settings = get_settings()
    if settings.rerank_provider != "local":
        logger.info("rerank disabled (provider={}), skip warmup", settings.rerank_provider)
        return
    logger.info("warming up reranker: {}", settings.local_rerank_model)
    try:
        from app.rerankers.local import LocalReranker

        # 用共享实例预热：warmup 加载的模型权重会被 chat 时直接复用，
        # 不会因为 .shared() 又新建一份。
        await asyncio.to_thread(LocalReranker.shared()._ensure)
        logger.info("reranker ready: {}", settings.local_rerank_model)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(
            "reranker warmup failed ({}). Will retry lazily on first query.", e
        )


async def _warmup_embedding() -> None:
    """启动时预热 embedding 模型，避免首次对话才动态加载（之前会卡 4 路加载）。"""
    settings = get_settings()
    if settings.embedding_provider != "local":
        logger.info(
            "embedding provider={}, skip warmup", settings.embedding_provider
        )
        return
    logger.info("warming up embedding: {}", settings.local_embedding_model)
    try:
        from app.embeddings.factory import EmbeddingFactory

        # EmbeddingFactory.get() 返回单例；调用一次 embed_query 把 SentenceTransformer
        # 加载到内存。chat 时所有 retrieve 共享这一份权重。
        await asyncio.to_thread(
            EmbeddingFactory.get().embed_query, "warmup"
        )
        logger.info("embedding ready: {}", settings.local_embedding_model)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(
            "embedding warmup failed ({}). Will retry lazily on first query.", e
        )


async def _warmup_bm25() -> None:
    """启动时预热 BM25：jieba dict + 加载所有 KB 的 pickle 索引。"""
    logger.info("warming up BM25: jieba + all KB indexes")
    try:
        from app.services.bm25_store import BM25Store

        # warmup_all 串行做：先初始化 jieba，再逐个 KB 加载 pickle + 重建索引。
        # chat 时 BM25Store.for_kb 直接命中 _cache，不再触发 jieba / rebuild。
        await asyncio.to_thread(BM25Store.warmup_all)
        logger.info("BM25 ready")
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(
            "BM25 warmup failed ({}). Will retry lazily on first query.", e
        )


async def _warmup_doc_index() -> None:
    """启动时预热 DocIndex：jieba + 加载所有 KB 的 doc-level BM25。"""
    s = get_settings()
    if not s.enable_doc_index:
        logger.info("doc index disabled, skip warmup")
        return
    logger.info("warming up DocIndex: jieba + all KB doc indexes")
    try:
        from app.services.doc_index import DocIndex

        # DocIndex.warmup_all 内部已 await 各 KB 的 load_from_db。
        await DocIndex.warmup_all()
        logger.info("DocIndex ready")
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(
            "DocIndex warmup failed ({}). Will retry lazily on first query.", e
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    settings.ensure_dirs()
    # create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # 内联迁移：补齐老库缺的新列
    await _run_inline_migrations()
    logger.info("app started on {}:{}", settings.host, settings.port)
    # 预热模型：把"首次对话要等加载"挪到启动时
    # 并行预热 embedding + reranker + bm25 + doc_index（四个独立模块，无依赖）
    await asyncio.gather(
        _warmup_embedding(), _warmup_reranker(), _warmup_bm25(), _warmup_doc_index()
    )
    yield
    logger.info("app shutting down")
    await engine.dispose()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="RAG System API",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    origins = settings.cors_origins
    # When running under Tauri, also allow tauri:// and asset://
    for extra in (
        "tauri://localhost",
        "http://tauri.localhost",
        "http://localhost:1420",
        "http://127.0.0.1:1420",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ):
        if extra not in origins:
            origins.append(extra)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/")
    async def root() -> dict:
        return {
            "name": "RAG System",
            "version": "0.1.0",
            "docs": "/docs",
            "endpoints": {
                "knowledge_bases": "/api/v1/knowledge-bases",
                "documents": "/api/v1/knowledge-bases/{id}/documents",
                "chat": "/api/v1/chat",
                "search": "/api/v1/search",
            },
        }

    @app.exception_handler(Exception)
    async def unhandled(request, exc):
        logger.exception("unhandled error")
        return JSONResponse(
            status_code=500,
            content={"detail": "internal server error", "message": str(exc)},
        )

    return app


app = create_app()


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
