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
