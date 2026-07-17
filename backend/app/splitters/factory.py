"""切片器工厂。"""

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
     """按段落优先 + 字符回退的通用切片（单层，旧实现，保留作 fallback）。"""

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
     """默认走 ParentChildSplitter（父子两段切片），更利于召回 + 上下文完整。

     旧的 RecursiveSplitter 保留为 fallback（用 splitter="recursive" 显式选）。
     """
     from app.splitters.parent_child import ParentChildSplitter

     if chunk_size is not None and chunk_size > 0:
         # 调用方显式给了 chunk_size，按 legacy 行为走 single-level
         # （主要是兼容旧测试 build_splitter(chunk_size=200).split(...) 的形态）
         return RecursiveSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap or 0)
     return ParentChildSplitter()
