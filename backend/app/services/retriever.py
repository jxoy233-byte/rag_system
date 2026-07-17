"""混合检索：向量 + BM25 → RRF → Rerank。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.embeddings.factory import EmbeddingFactory
from app.rerankers import LocalReranker
from app.services.bm25_store import BM25Store
from app.vectorstore import ChromaStore, RetrievedItem


@dataclass
class RetrievedChunk:
     id: str
     text: str
     metadata: dict = field(default_factory=dict)
     score: float = 0.0
     vector_score: float = 0.0
     bm25_score: float = 0.0
     rerank_score: float | None = None
     raw_score: float | None = None  # boost 前的原始相关度（0~1），用于 sources 展示
     source: str = "vector"  # vector / bm25 / fused


class HybridRetriever:
     """混合检索器：可对单一知识库或全局检索。"""

     def __init__(
         self,
         knowledge_base_id: int,
         collection_name: str,
         rerank: bool = True,
         embedding_model: str | None = None,
         embedding_dim: int | None = None,
     ) -> None:
         self.kb_id = knowledge_base_id
         self.collection_name = collection_name
         self.embedding_model = embedding_model
         self.embedding_dim = embedding_dim
         self._rerank_enabled = rerank
         self._chroma: ChromaStore | None = None
         self._bm25: BM25Store | None = None
         self._reranker: LocalReranker | None = None

     @property
     def chroma(self) -> ChromaStore:
         if self._chroma is None:
             embedding = EmbeddingFactory.get(
                 model_name=self.embedding_model,
                 dim=self.embedding_dim,
             )
             self._chroma = ChromaStore(
                 collection_name=self.collection_name,
                 embedding=embedding,
             )
         return self._chroma

     @property
     def bm25(self) -> BM25Store:
         if self._bm25 is None:
             self._bm25 = BM25Store.for_kb(self.kb_id)
         return self._bm25

     @property
     def reranker(self) -> LocalReranker:
         # 用进程级共享实例：cross-encoder 模型 ~280MB，加载成本高，
         # 之前每路 retrieve 都新建实例，multi-query 下会被重复加载 N 次。
         if self._reranker is None:
             self._reranker = LocalReranker.shared()
         return self._reranker

     async def retrieve(
         self,
         query: str,
         top_k: int | None = None,
         use_rerank: bool | None = None,
     ) -> list[RetrievedChunk]:
         s = get_settings()
         k = top_k or s.final_top_k
         use_rerank = self._rerank_enabled if use_rerank is None else use_rerank

         vec_task = asyncio.to_thread(self.chroma.query, query, s.rerank_top_k)
         bm25_task = asyncio.to_thread(self.bm25.query, query, s.rerank_top_k)
         vec_hits, bm25_hits = await asyncio.gather(vec_task, bm25_task, return_exceptions=True)

         vec_items: list[RetrievedItem] = vec_hits if isinstance(vec_hits, list) else []
         bm25_items: list[tuple[Any, float]] = bm25_hits if isinstance(bm25_hits, list) else []

         fused = self._rrf_fuse(vec_items, bm25_items)

         if not fused:
             return []

         if use_rerank and fused:
             # 把全部融合候选都交给 reranker；最终取 top k。
             # 之前传 s.rerank_top_k (=20) 在 fused>20 时会丢掉尾部候选的精排机会。
             texts = [c.text for c in fused]
             ranked = await asyncio.to_thread(self.reranker.rerank, query, texts, len(fused))
             for idx, score in ranked:
                 if 0 <= idx < len(fused):
                     fused[idx].rerank_score = float(score)
             fused.sort(key=lambda c: (c.rerank_score or 0.0), reverse=True)

         # 父子切片下做 parent_collapse：把同一 parent 的多个 child 合并成单个 parent 块，
         # 保留分数最高的那个 child 作为锚点。最终 LLM 看到的是完整 parent 全文，
         # 而不是孤立的 250 字片段 —— 上下文更全。
         if s.parent_collapse:
             fused = self._parent_collapse(fused)

         return fused[:k]

     def _parent_collapse(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
         """把同一 parent 的多个 child 合并成单个 parent 块。

         规则：
         - 用 metadata.parent_id 分组；空 parent_id 视为独立（不合并，老 chunk）
         - 同 parent 取 rerank_score 最高的 child 作为锚点（用它的 score / rerank_score）
         - text 替换为 metadata.parent_text（完整 parent 全文）
         - id 用锚点 child 的 id（保持 chunk_id 可追溯）+ 加 `parent:` 前缀避免重复
         - source 标记为 "parent"（区分于 "vector" / "bm25" / "fused"）
         - 不在原 chunks 顺序上的合并：保留按 rerank_score 降序，方便后续 top-k 截断
         """
         if not chunks:
             return chunks
         # 第一次扫：找每个 parent 的最高分 child
         best_by_parent: dict[str, RetrievedChunk] = {}
         # 没 parent_id 的（老 chunk / 单层 splitter 输出）单独保留
         no_parent: list[RetrievedChunk] = []
         for c in chunks:
             pid = (c.metadata or {}).get("parent_id") or ""
             if not pid:
                 no_parent.append(c)
                 continue
             score = c.rerank_score if c.rerank_score is not None else c.score
             existing = best_by_parent.get(pid)
             if existing is None:
                 best_by_parent[pid] = c
             else:
                 ex_score = existing.rerank_score if existing.rerank_score is not None else existing.score
                 if (score or 0.0) > (ex_score or 0.0):
                     best_by_parent[pid] = c
         # 第二次：把 best 替换成 parent 视图
         collapsed: list[RetrievedChunk] = []
         for pid, c in best_by_parent.items():
             md = dict(c.metadata or {})
             parent_text = md.get("parent_text") or c.text
             # 用 parent_id 作为 id（同一个 parent 多次访问指向同一段，
             # 前端打开 chunk 详情时按 id 找 — 实际我们这里 id 是 chunk_id，但前端
             # 期望 id 唯一；改成 "parent:" + parent_id 即可保持唯一且指向 parent）
             new_chunk = RetrievedChunk(
                 id=f"parent:{pid}",
                 text=parent_text,
                 metadata={**md, "child_id": c.id, "is_parent": True},
                 vector_score=c.vector_score,
                 bm25_score=c.bm25_score,
                 rerank_score=c.rerank_score,
                 source="parent",
             )
             collapsed.append(new_chunk)
         # 没 parent_id 的老 chunk 保留原样（向后兼容）
         collapsed.extend(no_parent)
         # 按 rerank_score / score 降序
         collapsed.sort(
             key=lambda c: (
                 c.rerank_score if c.rerank_score is not None else c.score,
                 c.score,
             ),
             reverse=True,
         )
         return collapsed

     def _rrf_fuse(
         self,
         vec_items: list[RetrievedItem],
         bm25_items: list[tuple[Any, float]],
         k_const: int = 60,
     ) -> list[RetrievedChunk]:
         scores: dict[str, RetrievedChunk] = {}

         for rank, item in enumerate(vec_items):
             chunk = RetrievedChunk(
                 id=item.id,
                 text=item.text,
                 metadata=item.metadata,
                 vector_score=item.score,
                 source="vector",
             )
             scores[item.id] = chunk

         # BM25-only 兜底：纯 BM25 命中的 chunk（向量没召回）也要进 scores 字典，
         # 否则后面按 cid 取 bm25_rank 时它的分数能算上但 chunk 文本/元数据丢了。
         for bm_doc, score in bm25_items:
             if bm_doc.chunk_id in scores:
                 scores[bm_doc.chunk_id].bm25_score = float(score)
                 scores[bm_doc.chunk_id].source = "fused"
             else:
                 scores[bm_doc.chunk_id] = RetrievedChunk(
                     id=bm_doc.chunk_id,
                     text=bm_doc.text,
                     metadata=bm_doc.metadata,
                     bm25_score=float(score),
                     source="bm25",
                 )

         # 用每个 source 自己的原始排名算 RRF，而不是融合后的排名 ——
         # 否则等价于「出现一次 +1 / 出现两次 +2」，丢失了 BM25/向量各自的排序信号。
         vec_rank = {item.id: r for r, item in enumerate(vec_items)}
         bm25_rank = {bm_doc.chunk_id: r for r, (bm_doc, _) in enumerate(bm25_items)}

         for cid, chunk in scores.items():
             rrf = 0.0
             if cid in vec_rank:
                 rrf += 1.0 / (k_const + vec_rank[cid] + 1)
             if cid in bm25_rank:
                 rrf += 1.0 / (k_const + bm25_rank[cid] + 1)
             chunk.score = rrf
         return sorted(scores.values(), key=lambda c: c.score, reverse=True)
