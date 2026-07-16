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


class ChatMeta(BaseModel):
     intent: str
     latency_ms: int = 0
     used_web: bool = False
     used_rag: bool = False
     conversation_id: int | None = None
     message_id: int | None = None


class ChatFinalEvent(BaseModel):
     type: Literal["final"] = "final"
     meta: ChatMeta
     sources: list[SourceItem] = Field(default_factory=list)


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
