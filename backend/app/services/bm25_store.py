"""BM25 索引：每知识库一份 pickle，支持增量更新。"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from app.core.config import get_settings


@dataclass
class BM25Doc:
     chunk_id: str
     text: str
     metadata: dict = field(default_factory=dict)


class BM25Store:
     """轻量 BM25 存储，按知识库隔离。"""

     _cache: dict[str, "BM25Store"] = {}

     def __init__(self, knowledge_base_id: int) -> None:
         self.kb_id = knowledge_base_id
         self._docs: list[BM25Doc] = []
         self._index: BM25Okapi | None = None
         s = get_settings()
         s.bm25_index_dir.mkdir(parents=True, exist_ok=True)
         self._path = s.bm25_index_dir / f"kb_{knowledge_base_id}.pkl"
         self._load()

     @classmethod
     def for_kb(cls, kb_id: int) -> "BM25Store":
         if kb_id not in cls._cache:
             cls._cache[kb_id] = cls(kb_id)
         return cls._cache[kb_id]

     # ===== persistence =====

     def _load(self) -> None:
         if self._path.exists():
             try:
                 with self._path.open("rb") as f:
                     payload = pickle.load(f)
                 self._docs = payload.get("docs", [])
                 self._rebuild()
             except Exception:
                 self._docs = []
                 self._index = None

     def _persist(self) -> None:
         with self._path.open("wb") as f:
             pickle.dump({"docs": self._docs}, f)

     def _rebuild(self) -> None:
         if not self._docs:
             self._index = None
             return
         tokenized = [self._tokenize(d.text) for d in self._docs]
         self._index = BM25Okapi(tokenized)

     @staticmethod
     def _tokenize(text: str) -> list[str]:
         # 中文按字 + 空格分词；英文按词
         cleaned = text.replace("\n", " ")
         tokens = []
         for tok in jieba.cut(cleaned):
             tok = tok.strip()
             if not tok or tok in {" ", "\t"}:
                 continue
             tokens.append(tok)
         return tokens

     # ===== ops =====

     def add(self, docs: list[BM25Doc]) -> None:
         self._docs.extend(docs)
         self._rebuild()
         self._persist()

     def delete_by_doc_id(self, doc_id: int) -> int:
         before = len(self._docs)
         self._docs = [d for d in self._docs if d.metadata.get("doc_id") != doc_id]
         removed = before - len(self._docs)
         if removed:
             self._rebuild()
             self._persist()
         return removed

     def reset(self) -> None:
         self._docs = []
         self._index = None
         if self._path.exists():
             self._path.unlink()

     def query(self, text: str, top_k: int = 10) -> list[tuple[BM25Doc, float]]:
         if self._index is None or not self._docs:
             return []
         tokens = self._tokenize(text)
         if not tokens:
             return []
         scores = self._index.get_scores(tokens)
         order = scores.argsort()[::-1][:top_k]
         return [(self._docs[i], float(scores[i])) for i in order if scores[i] > 0]

     def __len__(self) -> int:
         return len(self._docs)
