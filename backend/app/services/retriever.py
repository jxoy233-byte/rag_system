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
         if self._reranker is None:
             self._reranker = LocalReranker()
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

         if use_rerank and len(fused) > k:
             texts = [c.text for c in fused]
             ranked = await asyncio.to_thread(self.reranker.rerank, query, texts, s.rerank_top_k)
             for idx, score in ranked:
                 if 0 <= idx < len(fused):
                     fused[idx].rerank_score = float(score)
             fused.sort(key=lambda c: (c.rerank_score or 0.0), reverse=True)

         return fused[:k]

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

         for rank, (bm_doc, score) in enumerate(bm25_items):
             existing = scores.get(bm_doc.chunk_id)
             if existing:
                 existing.bm25_score = float(score)
                 existing.source = "fused"
             else:
                 scores[bm_doc.chunk_id] = RetrievedChunk(
                     id=bm_doc.chunk_id,
                     text=bm_doc.text,
                     metadata=bm_doc.metadata,
                     bm25_score=float(score),
                     source="bm25",
                 )

         for rank, chunk in enumerate(
             sorted(
                 scores.values(),
                 key=lambda c: max(c.vector_score, c.bm25_score),
                 reverse=True,
             )
):
             rrf = 0.0
             if chunk.vector_score > 0:
                 rrf += 1.0 / (k_const + rank + 1)
             if chunk.bm25_score > 0:
                 rrf += 1.0 / (k_const + rank + 1)
             chunk.score = rrf
         return sorted(scores.values(), key=lambda c: c.score, reverse=True)
