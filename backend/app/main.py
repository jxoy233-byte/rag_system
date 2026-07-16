"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import api_router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.models.base import Base
from app.core.db import engine

logger = get_logger(__name__)


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

        await asyncio.to_thread(LocalReranker()._ensure)
        logger.info("reranker ready: {}", settings.local_rerank_model)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(
            "reranker warmup failed ({}). Will retry lazily on first query.", e
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    settings.ensure_dirs()
    # create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("app started on {}:{}", settings.host, settings.port)
    # 预热重排序模型：把"首次对话要等下载"挪到启动时
    await _warmup_reranker()
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
