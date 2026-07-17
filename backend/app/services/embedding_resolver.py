"""Embedding 模型解析：优先用 KB 显式记录，否则从 settings 取。

设计目的：
- KB 创建时若用 local/mock provider（绝大多数场景），不把模型名/维度落表，
  运行时永远从 settings 读。这样改 .env 后所有 local KB 自动跟随，
  不用手动 UPDATE 数据库。
- KB 显式传了 embedding_model/dim（典型场景：用 OpenAI 兼容接口做 embedding），
  按 KB 记录走，保留「同一系统里不同 KB 走不同模型」的能力。

调用方：ingest / retriever / agent / search 等都走这里，别再直接用 kb.embedding_model。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.config import Settings, get_settings

if TYPE_CHECKING:
    from app.models.knowledge_base import KnowledgeBase


def resolve_embedding(
    kb: "KnowledgeBase | None",
    settings: Settings | None = None,
) -> tuple[str, int]:
    """返回 KB 当前应使用的 (model_name, dim)。

    优先级：
    1. KB 显式落了非空值 → 用之（per-KB 覆盖，典型是 OpenAI provider）
    2. 否则 → 从 settings 实时取（local/openai/mock 都按当前 .env 走）

    之所以第二个分支覆盖 openai/mock 而不是只覆盖 local：单一逻辑路径，
    避免「KB 没填 → 不知道该用 local 还是 openai」的二义性，统一由 settings 决定。
    """
    s = settings or get_settings()
    if (
        kb is not None
        and kb.embedding_model is not None
        and kb.embedding_dim is not None
    ):
        return kb.embedding_model, kb.embedding_dim

    if s.embedding_provider in {"local", "mock"}:
        return s.local_embedding_model, s.local_embedding_dim

    # OpenAI 兼容：模型名取 settings.openai_embedding_model；维度跟模型走，
    # 这里给个常用默认值（1536 = text-embedding-3-small / ada-002）。
    # 真要严格可让 KB 显式落 embedding_dim 覆盖。
    return s.openai_embedding_model, 1536