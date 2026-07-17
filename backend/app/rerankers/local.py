"""本地 BGE-Reranker 封装。"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from loguru import logger

from app.core.config import get_settings


@runtime_checkable
class RerankerLike(Protocol):
     def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]: ...


@dataclass
class ScoredDoc:
     index: int
     score: float


class LocalReranker:
     """BGE-Reranker v2 M3，懒加载 + 进程级单例。

     重型 cross-encoder 模型 (~280MB) 加载成本高；
     之前每路 retrieve 都新建一个实例，multi-query 下会重复加载 4 次。
     改成 class 级缓存：所有 retrieve / 所有 chat 共享同一份模型权重。
     """

     _shared: "LocalReranker | None" = None
     # 进程级锁：多线程并发首次加载时只让一个线程执行；其余等锁。
     # 共享实例解决了重复创建问题，但 _ensure() 内部的 check-then-load
     # 在 4 路 multi-query 并发下还是会 race（4 个线程同时看到 _model is None）。
     _load_lock = threading.Lock()

     def __init__(self, model_name: str | None = None) -> None:
         s = get_settings()
         self.model_name = model_name or s.local_rerank_model
         self._resolved_path: str | None = None
         self._model = None

     @classmethod
     def shared(cls) -> "LocalReranker":
         """获取（或懒初始化）进程级共享实例。"""
         if cls._shared is None:
             cls._shared = cls()
         return cls._shared

     @classmethod
     def reset_shared(cls) -> None:
         """测试 / 切换 model_name 时清理共享实例。"""
         cls._shared = None

     def _resolve_path(self) -> str:
         if self._resolved_path is None:
             from app.core.config import get_settings
             from app.core.local_model import resolve_local_path

             s = get_settings()
             self._resolved_path = resolve_local_path(
                 self.model_name,
                 root=s.local_model_root,
                 override=s.local_rerank_path or None,
                 hf_endpoint=s.hf_endpoint or None,
             )
         return self._resolved_path

     def _ensure(self):
         # 双重检查锁：先无锁快路径，已加载直接返回。
         if self._model is not None:
             return self._model
         with self._load_lock:
             if self._model is not None:
                 return self._model
             model_path = self._resolve_path()
             # 优先尝试 FlagEmbedding；如果加载后 compute_score 失败
             # （常见原因是 transformers 新版本删了 tokenizer.prepare_for_model），
             # 自动回退到 sentence_transformers.CrossEncoder，对 BGE 系列同样支持。
             flag_model = None
             try:
                 from FlagEmbedding import FlagReranker

                 m = FlagReranker(model_path, use_fp16=False)
                 try:
                     _ = m.compute_score([["ping", "pong"]], normalize=True)
                     flag_model = m
                 except Exception as e:
                     logger.warning(
                         "FlagEmbedding compute_score smoke test failed ({}); "
                         "falling back to sentence_transformers.CrossEncoder",
                         e,
                     )
             except Exception as e:
                 logger.warning("FlagEmbedding import/load failed ({}); using CrossEncoder", e)

             if flag_model is not None:
                 self._model = flag_model
             else:
                 from sentence_transformers import CrossEncoder

                 self._model = CrossEncoder(model_path)
         return self._model

     def rerank(
         self, query: str, documents: list[str], top_k: int
     ) -> list[tuple[int, float]]:
         if not documents:
             return []
         model = self._ensure()
         # FlagEmbedding 实际返回的实例类名可能是 BaseReranker / LLMReranker 等子类，
         # 不要用 == 比较类名，用 isinstance 覆盖整个 FlagReranker 继承链。
         try:
             from FlagEmbedding import FlagReranker as _FlagRerankerCls
             is_flag = isinstance(model, _FlagRerankerCls)
         except Exception:
             is_flag = False
         if is_flag:
             pairs = [[query, d] for d in documents]
             scores = model.compute_score(pairs, normalize=True)
             scored = list(enumerate(scores))
         else:
             pairs = [(query, d) for d in documents]
             raw = model.predict(pairs, show_progress_bar=False)
             scored = list(enumerate(raw))

         scored.sort(key=lambda x: x[1], reverse=True)
         top_k = max(1, min(top_k, len(scored)))
         return scored[:top_k]
