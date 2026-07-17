from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
     from app.models.document import Document


class KnowledgeBase(Base):
     """知识库：对应一个 Chroma collection 与一份 BM25 索引。"""

     __tablename__ = "knowledge_bases"

     name: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
     slug: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
     description: Mapped[str | None] = mapped_column(Text, default=None)
     collection_name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
     embedding_model: Mapped[str | None] = mapped_column(String(256), nullable=True)
     embedding_dim: Mapped[int | None] = mapped_column(default=None, nullable=True)
     chunk_size: Mapped[int] = mapped_column(default=600, nullable=False)
     chunk_overlap: Mapped[int] = mapped_column(default=120, nullable=False)
     doc_count: Mapped[int] = mapped_column(default=0, nullable=False)
     chunk_count: Mapped[int] = mapped_column(default=0, nullable=False)

     documents: Mapped[list["Document"]] = relationship(
         back_populates="knowledge_base",
         cascade="all, delete-orphan",
         lazy="selectin",
     )
