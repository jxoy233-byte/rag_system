"""Chunk 端点 Pydantic DTO。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal["vector", "bm25", "web"]


class ChunkDetail(BaseModel):
    """单个 chunk 完整信息（按 chunk_id 查询返回）。"""

    chunk_id: str
    doc_id: int
    kb_id: int
    text: str = Field(..., description="完整文本（不限长）")
    page: int | None = None
    section: str | None = None
    score: float | None = None
    rerank_score: float | None = None
    document: str | None = Field(None, description="doc_title")
    source_type: SourceType = "vector"


class ChunkListItem(BaseModel):
    """按文档列出的切片概要（用于「切片预览」面板）。"""

    chunk_id: str
    doc_id: int
    kb_id: int
    length: int
    preview: str = Field(..., description="text[:200]")
    page: int | None = None
    section: str | None = None
    document: str | None = None
    chunk_index: int | None = Field(None, description="入库时的切片顺序")


class ContentSegment(BaseModel):
    """原文视图里的一段（对应一个切片的完整文本）。"""

    chunk_index: int | None = None
    page: int | None = None
    section: str | None = None
    text: str


class DocumentContent(BaseModel):
    """文档「原文」视图：解析后拼接的连续全文 + 分段明细。"""

    doc_id: int
    kb_id: int
    title: str | None = None
    segments: list[ContentSegment] = Field(default_factory=list)
    full_text: str = Field("", description="各段 text 以空行连接")
