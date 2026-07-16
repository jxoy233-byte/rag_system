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
     source_uri: Mapped[str | None] = mapped_column(String(1024), default=None)

     knowledge_base: Mapped["KnowledgeBase"] = relationship(back_populates="documents")
