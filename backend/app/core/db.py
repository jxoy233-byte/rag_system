"""异步 SQLAlchemy 引擎与 Session。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings

_settings = get_settings()

engine = create_async_engine(
    _settings.sqlite_url,
    echo=False,
    future=True,
    connect_args={
        # aiosqlite -> sqlite3：check_same_thread 必须 False 才能跨事件循环使用
        "check_same_thread": False,
        # 5 秒超时后放弃等待（默认 SQLITE_BUSY 是立即抛错，并发上传会很惨）
        "timeout": 5.0,
    },
)

# SQLite + 多后台任务：必须开 WAL，否则并发写入互相阻塞。
# 同时设置 busy_timeout 让被锁的事务等待而不是立即失败。
@event.listens_for(engine.sync_engine, "connect")
def _sqlite_pragmas(dbapi_conn, _record):
    cursor = dbapi_conn.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()


AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
     async with AsyncSessionLocal() as session:
         try:
             yield session
             await session.commit()
         except Exception:
             await session.rollback()
             raise


async def get_session() -> AsyncIterator[AsyncSession]:
     """FastAPI Depends 用。"""
     async with AsyncSessionLocal() as session:
         try:
             yield session
             await session.commit()
         except Exception:
             await session.rollback()
             raise
