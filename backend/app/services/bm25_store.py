"""BM25 索引：每知识库一份 pickle，支持增量更新。"""

from __future__ import annotations

import pickle
import threading
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
     # for_kb() 是多线程入口：multi-query 4 路并发时 4 个线程会同时调用，
     # 之前 dict 的 check-then-set 模式在 GIL 下也能 race，导致 4 个实例
     # 同时 _load() / _rebuild()（同一份 pickle 加载 4 次 + 全文 jieba tokenize 4 次）。
     _cache_lock = threading.Lock()

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
         # 双重检查锁：fast path 不持锁，已缓存直接返回。
         if kb_id in cls._cache:
             return cls._cache[kb_id]
         with cls._cache_lock:
             if kb_id not in cls._cache:
                 cls._cache[kb_id] = cls(kb_id)
         return cls._cache[kb_id]

     @classmethod
     def warmup_all(cls) -> None:
         """启动时调用一次：把磁盘上所有 KB 的 pickle 加载并预热 jieba。

         之前 chat 第一次触发 for_kb() 时会：
         1. 加载 pickle（I/O）
         2. _rebuild() → 对每个 doc 调 jieba.cut() 全文 tokenize
         多 KB + 多线程并发时这两步都重复 4 次，体感卡死。
         """
         # 1. 先单独 warm 一下 jieba，避免 4 线程抢 dict 初始化
         try:
             list(jieba.cut("warmup"))
         except Exception:
             pass
         # 2. 串行加载所有 KB 的 pickle + rebuild（不并发，避免重复 tokenize）
         s = get_settings()
         if not s.bm25_index_dir.exists():
             return
         for pkl in sorted(s.bm25_index_dir.glob("kb_*.pkl")):
             try:
                 kb_id = int(pkl.stem.split("_", 1)[1])
             except (ValueError, IndexError):
                 continue
             # 直接构造 + 走 _load()；不绕 cache 因为 cache 还没初始化
             cls.for_kb(kb_id)

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
