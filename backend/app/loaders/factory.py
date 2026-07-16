"""Loader 工厂：根据扩展名自动选择。"""

from __future__ import annotations

from pathlib import Path

from app.loaders.base import BaseLoader, LoadedDocument
from app.loaders.docx_loader import DocxLoader
from app.loaders.markdown_loader import (
    CsvLoader,
    HtmlLoader,
    MarkdownLoader,
    TextLoader,
    XlsxLoader,
)
from app.loaders.pdf_loader import PDFLoader
from app.loaders.pptx_loader import PptxLoader


class DocumentLoaderFactory:
     """Loader 注册表 + 入口。"""

     _loaders: list[BaseLoader] = [
         PDFLoader(),
         DocxLoader(),
         PptxLoader(),
         MarkdownLoader(),
         TextLoader(),
         HtmlLoader(),
         CsvLoader(),
         XlsxLoader(),
     ]

     @classmethod
     def supported_extensions(cls) -> set[str]:
         exts: set[str] = set()
         for loader in cls._loaders:
             exts.update(loader.extensions)
         return exts

     @classmethod
     def get(cls, ext: str) -> BaseLoader | None:
         ext = ext.lower()
         for loader in cls._loaders:
             if loader.supports(ext):
                 return loader
         return None

     @classmethod
     def load(cls, path: str | Path) -> LoadedDocument:
         p = Path(path)
         loader = cls.get(p.suffix)
         if loader is None:
             raise ValueError(f"Unsupported file type: {p.suffix}")
         return loader.load(p)
