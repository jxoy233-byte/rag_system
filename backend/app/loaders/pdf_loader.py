"""PDF 加载器：优先 pdfplumber（支持图片提取），回退 pypdf。"""

from __future__ import annotations

from pathlib import Path

from app.loaders.base import BaseLoader, ImageInfo, LoadedDocument, LoadedPage


class PDFLoader(BaseLoader):
    extensions = (".pdf",)

    def load(self, path: str | Path) -> LoadedDocument:
        p = Path(path)
        pages: list[LoadedPage] = []
        meta: dict = {}

        try:
            import pdfplumber

            with pdfplumber.open(str(p)) as pdf:
                meta["page_count"] = len(pdf.pages)
                meta["metadata"] = dict(pdf.metadata or {})
                img_idx = 0

                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    images: list[ImageInfo] = []

                    # 提取表格
                    tables = page.extract_tables() or []
                    if tables:
                        for tbl in tables:
                            text += "\n" + self._table_to_text(tbl)

                    # 提取图片
                    try:
                        for img_ref in page.images or []:
                            img_data = page.extract_image(img_ref["xobj_id"])
                            if img_data:
                                img_idx += 1
                                images.append(
                                    ImageInfo(
                                        image_bytes=img_data["image_bytes"],
                                        mime_type=img_data.get("mime_type", "image/png"),
                                        position=len(text),
                                    )
                                )
                                text += f"\n[图片:{img_idx}]"
                    except Exception:
                        pass

                    pages.append(
                        LoadedPage(
                            text=text,
                            page=i,
                            images=images,
                            metadata={
                                "page": i,
                                "tables": len(tables),
                                "images": len(images),
                            },
                        )
                    )
        except Exception:
            # 回退到 pypdf（不支持图片）
            from pypdf import PdfReader

            reader = PdfReader(str(p))
            meta["page_count"] = len(reader.pages)
            for i, page in enumerate(reader.pages, start=1):
                pages.append(
                    LoadedPage(
                        text=page.extract_text() or "",
                        page=i,
                        images=[],
                        metadata={"page": i},
                    )
                )

        full = "\n\n".join(p.text for p in pages if p.text)
        total_images = sum(len(p.images) for p in pages)

        return LoadedDocument(
            text=full,
            pages=pages,
            metadata={**meta, "image_count": total_images},
            source=str(p),
            ext=p.suffix.lower(),
        )

    @staticmethod
    def _table_to_text(table: list[list[str | None]]) -> str:
        rows: list[str] = []
        for row in table:
            cells = [str(c).strip() if c else "" for c in row]
            if any(cells):
                rows.append(" | ".join(cells))
        return "\n".join(rows)