from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.document import DocumentStatus


class DocumentRead(BaseModel):
     model_config = ConfigDict(from_attributes=True)

     id: int
     knowledge_base_id: int
     title: str
     filename: str
     file_ext: str
     file_size: int
     status: DocumentStatus
     error: str | None
     chunk_count: int
     # parent chunk 数（父子切片下的"父块"数，去重后）。child 数 = chunk_count。
     parent_count: int = 0
     # 入库时由 LLM 生成的文档摘要（~150-300 字）。
     # DocIndex 用它做"哪些文档相关"的 BM25 匹配；
     # 前端在文档列表/抽屉里展示给用户，方便快速浏览每份文档的内容范围。
     summary: str = ""
     created_at: datetime
     updated_at: datetime


class DocumentChunkRead(BaseModel):
     id: str
     doc_id: int
     text: str
     metadata: dict
     score: float | None = None


class DocumentListResponse(BaseModel):
     items: list[DocumentRead]
     total: int


class UploadResponse(BaseModel):
     document: DocumentRead
     accepted: bool
     message: str = ""


class BatchUploadResponse(BaseModel):
     documents: list[DocumentRead]
     failed: list[dict] = Field(default_factory=list)
