"""Embedding 工厂：根据配置选择本地 BGE / OpenAI 兼容接口 / mock。"""

from __future__ import annotations

import hashlib
import threading
from typing import Protocol, runtime_checkable

from app.core.config import Settings, get_settings


@runtime_checkable
class EmbeddingsLike(Protocol):
    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


def _resize_and_normalize(vector: list[float], dim: int) -> list[float]:
    if len(vector) < dim:
        raise ValueError(f"embedding model returned {len(vector)} dimensions, expected at least {dim}")
    resized = vector[:dim]
    norm = sum(x * x for x in resized) ** 0.5 or 1.0
    return [x / norm for x in resized]


class EmbeddingFactory:
    """根据 settings 构建可复用的 Embeddings 实例。"""

    _instances: dict[str, EmbeddingsLike] = {}

    @classmethod
    def get(
        cls,
        settings: Settings | None = None,
        *,
        model_name: str | None = None,
        dim: int | None = None,
    ) -> EmbeddingsLike:
        s = settings or get_settings()
        effective_model = model_name or (
            s.local_embedding_model
            if s.embedding_provider in {"local", "mock"}
            else s.openai_embedding_model
        )
        effective_dim = dim or s.local_embedding_dim
        key = cls._cache_key(s, effective_model, effective_dim)
        if key in cls._instances:
            return cls._instances[key]
        emb: EmbeddingsLike
        if s.embedding_provider == "local":
            emb = _BGEEmbeddings(
                model_name=effective_model,
                device=s.local_embedding_device,
                dim=effective_dim,
            )
        elif s.embedding_provider == "mock":
            emb = _MockEmbeddings(dim=effective_dim)
        else:
            emb = _OpenAIEmbeddings(
                api_key=s.openai_embedding_api_key or s.openai_api_key,
                base_url=s.openai_embedding_base_url or s.openai_base_url,
                model=effective_model,
                dim=effective_dim,
            )
        cls._instances[key] = emb
        return emb

    @staticmethod
    def _cache_key(s: Settings, model_name: str, dim: int) -> str:
        base_url = s.openai_embedding_base_url or s.openai_base_url
        return f"{s.embedding_provider}|{model_name}|{dim}|{base_url}"


class _BGEEmbeddings:
    """通过 FlagEmbedding 调用 BGE-M3，懒加载避免启动慢。"""

    # 进程级锁：多线程并发首次访问时只让一个线程加载模型，
    # 其余线程等锁；避免 SentenceTransformer / tqdm 在并发初始化时死锁。
    _load_lock = threading.Lock()

    def __init__(self, model_name: str, device: str = "cpu", dim: int = 768) -> None:
        self.model_name = model_name
        self.device = device
        self.dim = dim
        self._resolved_path: str | None = None
        self._model = None

    def _resolve_path(self) -> str:
        if self._resolved_path is None:
            from app.core.config import get_settings
            from app.core.local_model import resolve_local_path

            s = get_settings()
            self._resolved_path = resolve_local_path(
                self.model_name,
                root=s.local_model_root,
                override=s.local_embedding_path or None,
                hf_endpoint=s.hf_endpoint or None,
            )
        return self._resolved_path

    def _ensure(self):
        # 双重检查锁：先无锁快路径，已加载直接返回；否则抢锁后再检查一次。
        # asyncio.to_thread 把 encode 扔到多线程，4 路 multi-query 时
        # 4 个线程会同时看到 _model is None 然后并发加载（之前会卡死/4× 内存）。
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is None:
                model_path = self._resolve_path()
                if "bge-m3" in self.model_name.lower():
                    from FlagEmbedding import BGEM3FlagModel

                    self._model = BGEM3FlagModel(
                        model_path,
                        use_fp16=False,
                        devices=self.device if self.device != "auto" else None,
                    )
                else:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(model_path, device=self.device)
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        m = self._ensure()
        cls = m.__class__.__name__
        if cls == "BGEM3FlagModel":
            out = m.encode(
                texts,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            vectors = out["dense_vecs"].tolist()
        else:
            vectors = m.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            ).tolist()
        return [_resize_and_normalize(vector, self.dim) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts)


class _OpenAIEmbeddings:
    def __init__(self, api_key: str, base_url: str, model: str, dim: int) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._dim = dim
        self._client = None

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def _call(self, texts: list[str]) -> list[list[float]]:
        c = self._ensure()
        resp = c.embeddings.create(model=self._model, input=texts)
        return [_resize_and_normalize(list(d.embedding), self._dim) for d in resp.data]

    def embed_query(self, text: str) -> list[float]:
        return self._call([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._call(texts)


class _MockEmbeddings:
    """Mock Embedding：用于测试/沙箱环境。
    用文本 hash + 维度归一化生成向量；与 bge-base-zh-v1.5 同维度（768），不调用任何外部 API。
    """

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim

    @staticmethod
    def _hash_vec(text: str, dim: int) -> list[float]:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        vec: list[float] = []
        for i in range(dim):
            seed = (seed * 1103515245 + 12345 + i) & 0x7FFFFFFF
            v = ((seed >> 8) & 0xFFFF) / 0xFFFF
            vec.append(v * 2 - 1)
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def embed_query(self, text: str) -> list[float]:
        return self._hash_vec(text, self.dim)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_vec(t, self.dim) for t in texts]
