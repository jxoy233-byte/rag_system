"""切片器工厂：基于 langchain-text-splitters 构建。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from langchain_text_splitters import (
     RecursiveCharacterTextSplitter,
)

from app.core.config import get_settings
from app.loaders.base import LoadedDocument


@dataclass
class TextChunk:
     text: str
     page: int | None = None
     section: str | None = None
     metadata: dict | None = None


class TextSplitter(Protocol):
     def split(self, doc: LoadedDocument) -> list[TextChunk]: ...


class RecursiveSplitter:
     """按段落优先 + 字符回退的通用切片。"""

     def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
         s = get_settings()
         self._size = chunk_size if chunk_size is not None else s.chunk_size
         self._overlap = chunk_overlap if chunk_overlap is not None else s.chunk_overlap
         self._impl = RecursiveCharacterTextSplitter(
             chunk_size=self._size,
             chunk_overlap=self._overlap,
             separators=["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""],
             length_function=len,
         )

     def split(self, doc: LoadedDocument) -> list[TextChunk]:
         chunks: list[TextChunk] = []
         for page in doc.to_pages():
             text = page.text.strip()
             if not text:
                 continue
             sub = self._impl.split_text(text)
             for i, t in enumerate(sub):
                 if not t.strip():
                     continue
                 chunks.append(
                     TextChunk(
                         text=t.strip(),
                         page=page.page,
                         section=page.section,
                         metadata={
                             "page": page.page,
                             "section": page.section,
                             "chunk_index": i,
                             **(page.metadata or {}),
                         },
                     )
                 )
         return chunks


def build_splitter(
     chunk_size: int | None = None, chunk_overlap: int | None = None
) -> TextSplitter:
     return RecursiveSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
