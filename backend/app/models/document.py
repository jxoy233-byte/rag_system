from __future__ import annotations

import enum
from typing import TYPE_CHECKING

from sqlalchemy import Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
     from app.models.knowledge_base import KnowledgeBase


class DocumentStatus(str, enum.Enum):
     pending = "pending"
     processing = "processing"
     ready = "ready"
     failed = "failed"


class Document(Base):
     """文档：原始文件 + 入库状态。"""

     __tablename__ = "documents"

     knowledge_base_id: Mapped[int] = mapped_column(
         ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True, nullable=False
     )
     title: Mapped[str] = mapped_column(String(512), nullable=False)
     filename: Mapped[str] = mapped_column(String(512), nullable=False)
     file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
     file_ext: Mapped[str] = mapped_column(String(16), nullable=False)
     file_size: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
     mime_type: Mapped[str] = mapped_column(String(128), default="", nullable=False)
     status: Mapped[DocumentStatus] = mapped_column(
         Enum(DocumentStatus), default=DocumentStatus.pending, nullable=False, index=True
     )
     error: Mapped[str | None] = mapped_column(Text, default=None)
     chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
     # parent chunk 数（父子切片下的"父块"数，去重后）。child 数 = chunk_count。
     # 用来给前端展示"文档被切成 N 大段 / M 小段"。
     parent_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
     source_uri: Mapped[str | None] = mapped_column(String(1024), default=None)
     # 文档级摘要：入库时由 LLM 生成（~150-300 字），用于 doc-level BM25 预筛选。
     # 与 chunk-level 检索互不替代：doc-level 决定"哪些文档相关"，chunk-level
     # 决定"哪些段落最匹配"。失败/未生成时为空字符串，DocIndex 自动跳过。
     summary: Mapped[str] = mapped_column(Text, default="", nullable=False)

     knowledge_base: Mapped["KnowledgeBase"] = relationship(back_populates="documents")
