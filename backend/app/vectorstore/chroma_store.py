"""ChromaDB 持久化封装，每个 collection 对应一个知识库。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.config import get_settings
from app.embeddings.factory import EmbeddingFactory, EmbeddingsLike


@dataclass
class RetrievedItem:
     id: str
     text: str
     metadata: dict
     score: float


class ChromaStore:
     """统一管理所有 collection 的薄封装。"""

     _client = None

     def __init__(
         self,
         collection_name: str,
         embedding: EmbeddingsLike | None = None,
         persist_dir: str | None = None,
     ) -> None:
         self.collection_name = collection_name
         self.embedding = embedding or EmbeddingFactory.get()
         self.persist_dir = persist_dir or str(get_settings().chroma_persist_dir)
         self._collection = None

     @classmethod
     def _get_client(cls):
         if cls._client is None:
             import chromadb
             from chromadb.config import Settings as ChromaSettings

             cls._client = chromadb.PersistentClient(
                 path=get_settings().chroma_persist_dir,
                 settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
             )
         return cls._client

     @property
     def collection(self):
         if self._collection is None:
             client = self._get_client()
             from chromadb.utils import embedding_functions

             self._collection = client.get_or_create_collection(
                 name=self.collection_name,
                 metadata={"hnsw:space": "cosine"},
             )
         return self._collection

     # ============ Write ============

     def add(
         self,
         ids: list[str],
         texts: list[str],
         metadatas: list[dict[str, Any]],
     ) -> None:
         embeddings = self.embedding.embed_documents(texts)
         # Chroma 不接受空 metadata
         cleaned: list[dict] = []
         for m in metadatas:
             cleaned.append({k: v for k, v in m.items() if v is not None and v != ""})
         self.collection.add(
             ids=ids,
             documents=texts,
             embeddings=embeddings,
             metadatas=cleaned,
         )

     def delete_by_doc_id(self, doc_id: int) -> int:
         existing = self.collection.get(where={"doc_id": doc_id})
         ids = existing.get("ids", [])
         if ids:
             self.collection.delete(ids=ids)
         return len(ids)

     def delete_by_ids(self, ids: list[str]) -> None:
         if ids:
             self.collection.delete(ids=ids)

     def get(
         self,
         ids: list[str] | None = None,
         where: dict | None = None,
     ) -> list[dict]:
         """按 id / where 拉取 chunk 完整文本与元数据（无 embedding 计算）。

         返回 list[dict]，每项形如 {"id": str, "text": str, "metadata": dict}。
         用于「按文档列出切片」与「按 chunk_id 查详情」两个端点。
         """
         res = self.collection.get(  # 走 @property 触发 lazy init
             ids=ids,
             where=where,
             include=["documents", "metadatas"],
         )
         return [
             {"id": cid, "text": txt, "metadata": (meta or {})}
             for cid, txt, meta in zip(
                 res.get("ids", []),
                 res.get("documents", []),
                 res.get("metadatas", []),
             )
         ]

     def reset(self) -> None:
         client = self._get_client()
         try:
             client.delete_collection(self.collection_name)
         except Exception:
             pass
         self._collection = None

     # ============ Read ============

     def count(self) -> int:
         try:
             return self.collection.count()
         except Exception:
             return 0

     def query(
         self,
         text: str,
         top_k: int = 10,
         where: dict | None = None,
     ) -> list[RetrievedItem]:
         if not text.strip():
             return []
         emb = self.embedding.embed_query(text)
         kwargs: dict = {
             "query_embeddings": [emb],
             "n_results": max(1, top_k),
         }
         if where:
             kwargs["where"] = where
         res = self.collection.query(**kwargs)
         items: list[RetrievedItem] = []
         ids = (res.get("ids") or [[]])[0]
         docs = (res.get("documents") or [[]])[0]
         metas = (res.get("metadatas") or [[]])[0]
         dists = (res.get("distances") or [[]])[0]
         for i, doc in enumerate(docs):
             score = 1.0 - (dists[i] if i < len(dists) else 0.0)
             items.append(
                 RetrievedItem(
                     id=ids[i] if i < len(ids) else "",
                     text=doc,
                     metadata=metas[i] if i < len(metas) else {},
                     score=float(score),
                 )
             )
         return items
