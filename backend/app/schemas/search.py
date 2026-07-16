from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.chat import SourceItem


class SearchRequest(BaseModel):
     query: str = Field(min_length=1, max_length=2000)
     knowledge_base_id: int | None = None
     top_k: int = Field(default=5, ge=1, le=50)
     use_rerank: bool = True


class SearchResponse(BaseModel):
     sources: list[SourceItem]
     latency_ms: int = 0
