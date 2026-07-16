"""Loader 抽象与统一数据结构。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ImageInfo:
    """待描述的图片信息。"""

    image_bytes: bytes
    mime_type: str  # image/png, image/jpeg 等
    position: int = 0  # 在文本中的插入位置（由 loader 填充）
    alt_text: str = ""  # 原始 alt/caption（如有）
    description: str = ""  # VLM 生成的描述（后填）


@dataclass
class LoadedPage:
    text: str
    page: int | None = None
    section: str | None = None
    metadata: dict = field(default_factory=dict)
    images: list[ImageInfo] = field(default_factory=list)


@dataclass
class LoadedDocument:
    text: str
    pages: list[LoadedPage] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    source: str = ""
    ext: str = ""

    def to_pages(self) -> list[LoadedPage]:
        if self.pages:
            return self.pages
        if self.text.strip():
            return [LoadedPage(text=self.text, page=1)]
        return []

    def has_images(self) -> bool:
        """检查文档是否包含图片。"""
        return any(p.images for p in self.pages)

    def all_images(self) -> list[ImageInfo]:
        """获取所有页面的图片。"""
        result = []
        for p in self.pages:
            result.extend(p.images)
        return result


class BaseLoader(ABC):
    """Loader 接口。"""

    extensions: tuple[str, ...] = ()

    @abstractmethod
    def load(self, path: str | Path) -> LoadedDocument: ...

    def supports(self, ext: str) -> bool:
        return ext.lower() in self.extensions
