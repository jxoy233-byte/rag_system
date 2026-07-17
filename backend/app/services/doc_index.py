"""Doc-level 检索索引：每知识库一份 BM25，建在 (title + filename + summary) 上。

目的：在 chunk-level 检索之前，先用 doc-level 命中"哪些文档相关"，
再给这些文档里的 chunk 在最终排序时一个 soft boost。
和 chunk-level BM25 的区别：
- chunk-level：决定"哪个段落最匹配" → 走 hybrid retrieval (vector + BM25 + rerank)
- doc-level：  决定"哪份文档相关"     → 本索引；top 文档内的 chunk 得分 ×1.2
summary 比 title 更稳定（用户上传的标题常常就是 "scan.pdf" 这种没意义的）。
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field

import jieba
from rank_bm25 import BM25Okapi
from sqlalchemy import select

from app.core.db import AsyncSessionLocal
from app.models import Document, DocumentStatus
from app.models.knowledge_base import KnowledgeBase


@dataclass
class DocHit:
    """DocIndex 命中项。"""

    doc_id: int
    title: str
    filename: str
    summary: str
    score: float = 0.0


@dataclass
class _DocEntry:
    """DocIndex 内部条目（BM25 文档单元）。"""

    doc_id: int
    title: str
    filename: str
    summary: str
    # BM25 索引用的合成文本：title 重复一次加权、filename、summary 都拼起来。
    bm25_text: str = ""


class DocIndex:
    """Per-KB doc-level BM25 索引。

    设计原则：
    - 进程级缓存（_cache），不写 pickle；Document 表就是 source of truth，
      启动时按需 reload；doc add/delete/retry 时调 invalidate(kb_id) 让 cache 失效。
    - 异步 load (load_from_db)：从 DB 读 (id, title, filename, summary) 走 SQLAlchemy。
    - 同步 BM25 query()：与 BM25Store 一致。
    - 多线程安全：和 BM25Store 同样用 double-check lock 防止并发 load。
    """

    _cache: dict[int, "DocIndex"] = {}
    _cache_lock = threading.Lock()

    def __init__(self, kb_id: int) -> None:
        self.kb_id = kb_id
        self._entries: list[_DocEntry] = []
        self._index: BM25Okapi | None = None
        self._loaded = False

    @classmethod
    def for_kb(cls, kb_id: int) -> "DocIndex":
        """获取（或懒初始化）指定 KB 的 DocIndex。

        命中即返回；未命中则 new 一个空实例（不主动 load）。
        load 由调用方（warmup 或 query 节点）显式触发。
        """
        if kb_id in cls._cache:
            return cls._cache[kb_id]
        with cls._cache_lock:
            if kb_id not in cls._cache:
                cls._cache[kb_id] = cls(kb_id)
        return cls._cache[kb_id]

    @classmethod
    def invalidate(cls, kb_id: int) -> None:
        """删除某个 KB 的缓存（doc add/delete/refresh 时调用）。"""
        cls._cache.pop(kb_id, None)

    @classmethod
    def invalidate_all(cls) -> None:
        """启动时 / 测试时全清。"""
        with cls._cache_lock:
            cls._cache.clear()

    @classmethod
    async def warmup_all(cls) -> None:
        """启动时调用一次：warm jieba + 把所有存在 doc 的 KB 加载 DocIndex。"""
        # 1. 先 warm jieba，避免 4 线程抢 dict 初始化（和 BM25Store 同坑）。
        try:
            await asyncio.to_thread(lambda: list(jieba.cut("warmup")))
        except Exception:
            pass

        # 2. 找所有有 ready 文档的 KB
        async with AsyncSessionLocal() as session:
            stmt = select(Document.knowledge_base_id).where(
                Document.status == DocumentStatus.ready
            ).distinct()
            rows = await session.execute(stmt)
            kb_ids = [r[0] for r in rows.fetchall()]

        for kb_id in kb_ids:
            try:
                idx = cls.for_kb(kb_id)
                await idx.load_from_db()
            except Exception as e:  # pragma: no cover - defensive
                from loguru import logger

                logger.warning("DocIndex warmup failed for kb={}: {}", kb_id, e)

    async def load_from_db(self) -> None:
        """从 SQLite 读 (doc_id, title, filename, summary) 重建索引。"""
        async with AsyncSessionLocal() as session:
            # 验证 KB 存在
            kb = await session.get(KnowledgeBase, self.kb_id)
            if kb is None:
                self._entries = []
                self._index = None
                self._loaded = True
                return

            stmt = (
                select(Document.id, Document.title, Document.filename, Document.summary)
                .where(Document.knowledge_base_id == self.kb_id)
                .where(Document.status == DocumentStatus.ready)
            )
            rows = (await session.execute(stmt)).fetchall()

        entries: list[_DocEntry] = []
        for row in rows:
            doc_id, title, filename, summary = row
            # title 重复一次加权（用户给的文件名/标题最直接的判据）
            bm25_text = " ".join(
                [
                    title or "",
                    filename or "",
                    title or "",  # 二次权重
                    summary or "",
                ]
            )
            entries.append(
                _DocEntry(
                    doc_id=int(doc_id),
                    title=title or "",
                    filename=filename or "",
                    summary=summary or "",
                    bm25_text=bm25_text,
                )
            )

        # 同步 BM25 索引构建放线程里跑（jieba.cut 全文 tokenize 对大库可能慢）
        self._entries, self._index = await asyncio.to_thread(self._build_index, entries)
        self._loaded = True

    @staticmethod
    def _build_index(entries: list[_DocEntry]) -> tuple[list[_DocEntry], BM25Okapi | None]:
        if not entries:
            return entries, None
        tokenized = [DocIndex._tokenize(e.bm25_text) for e in entries]
        # BM25Okapi 构造本身在 CPU 上；为 doc-level（文档数 << chunk 数）一般 <100ms
        return entries, BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        cleaned = text.replace("\n", " ")
        tokens = []
        for tok in jieba.cut(cleaned):
            tok = tok.strip()
            if not tok or tok in {" ", "\t"}:
                continue
            tokens.append(tok)
        return tokens

    async def ensure_loaded(self) -> None:
        """确保已加载；未加载则触发 load_from_db。"""
        if self._loaded:
            return
        await self.load_from_db()

    async def query(self, text: str, top_k: int = 3) -> list[DocHit]:
        """对用户问题在 (title+filename+summary) 上做 BM25，返回 top_k 个 doc。

        返回值按 score 降序。空索引 / 加载失败 / 文本无 token 都返回 []。
        """
        await self.ensure_loaded()
        if self._index is None or not self._entries:
            return []
        tokens = self._tokenize(text)
        if not tokens:
            return []
        scores = await asyncio.to_thread(self._index.get_scores, tokens)
        order = scores.argsort()[::-1][:top_k]
        hits: list[DocHit] = []
        for i in order:
            score = float(scores[i])
            if score <= 0:
                continue
            e = self._entries[int(i)]
            hits.append(
                DocHit(
                    doc_id=e.doc_id,
                    title=e.title,
                    filename=e.filename,
                    summary=e.summary,
                    score=score,
                )
            )
        return hits

    def __len__(self) -> int:
        return len(self._entries)