from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
     role: Literal["user", "assistant", "system"]
     content: str


class ChatRequest(BaseModel):
     question: str = Field(min_length=1, max_length=4000)
     knowledge_base_id: int | None = None
     conversation_id: int | None = None
     history: list[ChatMessage] = Field(default_factory=list)
     enable_web: bool = True
     stream: bool = True


class SourceItem(BaseModel):
     kb_id: int | None = None
     doc_id: int | None = None
     document: str
     page: int | None = None
     chunk_id: str | None = None
     snippet: str
     score: float | None = None
     source_type: Literal["vector", "bm25", "web"] = "vector"
     url: str | None = None


class DocHitItem(BaseModel):
     """doc-level 命中：相关文档清单 + summary。前端"相关文档"区展示。"""
     doc_id: int
     title: str
     filename: str
     summary: str
     score: float | None = None


class ChatMeta(BaseModel):
     intent: str
     latency_ms: int = 0
     used_web: bool = False
     used_rag: bool = False
     conversation_id: int | None = None
     message_id: int | None = None
     # 信心闸门标记：检索+web 都失败时为 True，前端据此切换"暂未收录"渲染样式
     # （区别于 error 的红色警示）。
     refused: bool = False


class ChatFinalEvent(BaseModel):
     type: Literal["final"] = "final"
     meta: ChatMeta
     sources: list[SourceItem] = Field(default_factory=list)
     doc_hits: list[DocHitItem] = Field(default_factory=list)


class ChatTokenEvent(BaseModel):
     type: Literal["token"] = "token"
     content: str


class ChatErrorEvent(BaseModel):
     type: Literal["error"] = "error"
     message: str


class ChatEndEvent(BaseModel):
     type: Literal["end"] = "end"


class ConversationRead(BaseModel):
     model_config = ConfigDict(from_attributes=True)

     id: int
     knowledge_base_id: int | None
     title: str
     created_at: datetime
     updated_at: datetime
     # 滚动摘要（前端通常不展示，但保留便于调试/未来「查看记忆」功能）
     summary: str = ""


class ConversationUpdate(BaseModel):
     title: str | None = Field(default=None, min_length=1, max_length=256)


class MessageRead(BaseModel):
     model_config = ConfigDict(from_attributes=True)

     id: int
     conversation_id: int
     role: str
     content: str
     intent: str | None
     latency_ms: int
     created_at: datetime
     # 助手消息的引用来源；从 Message.sources_json 解析回填（端点里手动填），
     # 用户消息为空列表。前端据此渲染可点击引用 chip。
     sources: list[SourceItem] = Field(default_factory=list)
     # doc-level 命中：从 Message.doc_hits_json 解析回填。前端"相关文档"区展示
     # （系统判断哪些文档与问题相关 + 每份文档的 summary），与 sources 独立。
     doc_hits: list[DocHitItem] = Field(default_factory=list)
