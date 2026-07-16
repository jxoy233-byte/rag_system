from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class KnowledgeBaseCreate(BaseModel):
     model_config = ConfigDict(str_strip_whitespace=True)

     name: str = Field(min_length=1, max_length=128)
     description: str | None = None
     embedding_model: str | None = None
     embedding_dim: int | None = Field(default=None, ge=64, le=4096)
     chunk_size: int | None = Field(default=None, ge=64, le=4000)
     chunk_overlap: int | None = Field(default=None, ge=0, le=2000)

     @model_validator(mode="after")
     def validate_chunk_settings(self):
         if (
             self.chunk_size is not None
             and self.chunk_overlap is not None
             and self.chunk_overlap >= self.chunk_size
         ):
             raise ValueError("chunk_overlap must be smaller than chunk_size")
         return self


class KnowledgeBaseUpdate(BaseModel):
     model_config = ConfigDict(str_strip_whitespace=True)

     name: str | None = Field(default=None, min_length=1, max_length=128)
     description: str | None = None
     embedding_model: str | None = None
     embedding_dim: int | None = Field(default=None, ge=64, le=4096)
     chunk_size: int | None = Field(default=None, ge=64, le=4000)
     chunk_overlap: int | None = Field(default=None, ge=0, le=2000)

     @model_validator(mode="after")
     def validate_chunk_settings(self):
         if (
             self.chunk_size is not None
             and self.chunk_overlap is not None
             and self.chunk_overlap >= self.chunk_size
         ):
             raise ValueError("chunk_overlap must be smaller than chunk_size")
         return self


class KnowledgeBaseRead(BaseModel):
     model_config = ConfigDict(from_attributes=True)

     id: int
     name: str
     slug: str
     description: str | None
     collection_name: str
     embedding_model: str
     embedding_dim: int
     chunk_size: int
     chunk_overlap: int
     doc_count: int
     chunk_count: int
     created_at: datetime
     updated_at: datetime
