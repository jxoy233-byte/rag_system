from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
     from app.models.knowledge_base import KnowledgeBase


class Conversation(Base):
     """会话：按知识库组织多轮问答。"""

     __tablename__ = "conversations"

     knowledge_base_id: Mapped[int | None] = mapped_column(
         ForeignKey("knowledge_bases.id", ondelete="SET NULL"), index=True, nullable=True
     )
     title: Mapped[str] = mapped_column(String(256), default="新会话", nullable=False)
     # 滚动摘要：当 messages 数量超过阈值时，把更早的对话压缩到这里。
     # 注入到 system message，让长对话仍然能引用早前内容。
     summary: Mapped[str] = mapped_column(Text, default="", nullable=False)

     messages: Mapped[list["Message"]] = relationship(
         back_populates="conversation",
         cascade="all, delete-orphan",
         lazy="selectin",
         order_by="Message.id",
     )


class Message(Base):
     """消息：user / assistant / system。"""

     __tablename__ = "messages"

     conversation_id: Mapped[int] = mapped_column(
         ForeignKey("conversations.id", ondelete="CASCADE"), index=True, nullable=False
     )
     role: Mapped[str] = mapped_column(String(16), nullable=False)  # user|assistant|system
     content: Mapped[str] = mapped_column(Text, nullable=False)
     sources_json: Mapped[str | None] = mapped_column(Text, default=None)
     # doc-level 命中：BM25(title+filename+summary) 出来的 top-K 文档列表 + summary。
     # 与 sources_json（chunk-level）独立：前端 "相关文档" 区显示这些，
     # 引用 chip 仍走 sources。
     doc_hits_json: Mapped[str | None] = mapped_column(Text, default=None)
     intent: Mapped[str | None] = mapped_column(String(32), default=None)
     latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

     conversation: Mapped["Conversation"] = relationship(back_populates="messages")
