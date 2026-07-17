"""入库编排：文件 → 解析（文本+图片）→ 图片描述 → 切片 → 向量化 → 存储。"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.embeddings.factory import EmbeddingFactory
from app.llm.factory import LLMFactory
from app.loaders import DocumentLoaderFactory
from app.loaders.base import ImageInfo, LoadedDocument
from app.models import Document, KnowledgeBase
from app.models.document import DocumentStatus
from app.services.bm25_store import BM25Doc, BM25Store
from app.services.embedding_resolver import resolve_embedding
from app.services.image_describer import create_image_describer
from app.splitters import build_splitter
from app.vectorstore import ChromaStore

if TYPE_CHECKING:
    from app.loaders.base import LoadedDocument


# 文档级摘要 prompt：~150-300 字，描述"这份文档主要讲什么"。
# 与 chunk 级检索互不替代 —— doc 级用于"先筛哪些文档相关"，
# chunk 级用于"在这份文档里挑哪些段落最匹配"。
DOC_SUMMARY_PROMPT = """你为入库的文档写一段简短摘要，用于检索系统判断"这份文档是否相关"。

文档文件名: {filename}
文档标题: {title}

文档正文（已截断到前 ~3000 字）:
\"\"\"
{preview}
\"\"\"

要求：
- 用中文写（与正文同语种）。
- 控制在 150-300 字内；信息密度要高，避免套话。
- 概括：①文档主题/领域；②主要内容/章节类型；③关键概念、术语或数据点。
- 不要逐字复述；不要写"本文档..."这种开场。
- 如果正文明显被截断或无意义（仅元数据），只输出"EMPTY"。
- 只输出摘要文本本身，不要任何标签或前缀。"""


class IngestService:
    """文档入库主流程。"""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_kb(self, kb_id: int) -> KnowledgeBase | None:
        stmt = (
            select(KnowledgeBase)
            .options(selectinload(KnowledgeBase.documents))
            .where(KnowledgeBase.id == kb_id)
        )
        res = await self.session.execute(stmt)
        return res.scalar_one_or_none()

    async def _refresh_doc_count(self, kb: KnowledgeBase) -> None:
        await self.session.flush()
        ready_count = await self.session.scalar(
            select(func.count(Document.id)).where(
                Document.knowledge_base_id == kb.id,
                Document.status == DocumentStatus.ready,
            )
        )
        kb.doc_count = int(ready_count or 0)

    async def ingest_path(
        self, kb: KnowledgeBase, src_path: str | Path, title: str | None = None
    ) -> Document:
        p = Path(src_path)
        s = get_settings()
        ext = p.suffix.lower()
        if ext not in s.allowed_extensions:
            raise ValueError(f"Unsupported file type: {ext}")

        doc = Document(
            knowledge_base_id=kb.id,
            title=title or p.stem,
            filename=p.name,
            file_path=str(p),
            file_ext=ext,
            file_size=p.stat().st_size,
            mime_type="",
            status=DocumentStatus.processing,
        )
        self.session.add(doc)
        await self.session.commit()
        await self.session.refresh(doc)

        try:
            # 解析文档（同步）
            loaded = await asyncio.to_thread(DocumentLoaderFactory.load, doc.file_path)

            # 处理图片描述（异步）
            if loaded.has_images():
                loaded = await self._process_images(loaded)

            # 切片 + 入库（同步）
            await asyncio.to_thread(self._ingest_chunks, kb, doc, loaded)

            # 文档级摘要（异步 LLM 调用）—— 失败不影响主流程。
            # 必须放在 status=ready 之前；否则摘要写入会被 status 分支跳过。
            doc.summary = await self._generate_doc_summary(doc, loaded)

            doc.status = DocumentStatus.ready
            doc.error = None
        except Exception as e:  # pragma: no cover - runtime error path
            logger.exception("ingest failed: {}", e)
            doc.status = DocumentStatus.failed
            doc.error = str(e)[:2000]
        finally:
            await self._refresh_doc_count(kb)
            await self.session.commit()
            await self.session.refresh(doc)
            # 文档入/退库后让 DocIndex 失效，下次 query 重新加载。
            # 放在 commit 之后避免 doc.summary 还没写入就被读到旧值。
            from app.services.doc_index import DocIndex

            try:
                DocIndex.invalidate(kb.id)
            except Exception as e:  # pragma: no cover - cache best-effort
                logger.warning("DocIndex.invalidate failed: {}", e)
        return doc

    async def _process_images(self, loaded: "LoadedDocument") -> "LoadedDocument":
        """为文档中的图片生成描述并替换占位符。"""
        s = get_settings()
        all_images = loaded.all_images()

        # 限制图片数量
        if len(all_images) > s.max_images_per_doc:
            logger.warning(
                "Document has {} images, limiting to {}",
                len(all_images),
                s.max_images_per_doc,
            )
            all_images = all_images[: s.max_images_per_doc]

        if not all_images:
            return loaded

        # 创建描述器（自动检测：有 Key 用 API，无 Key 用本地）
        describer = create_image_describer(
            api_key=s.image_description_api_key,
            base_url=s.image_description_base_url,
            model=s.image_description_model,
            local_model_id=s.local_vlm_model,
            local_dir=s.local_vlm_path or None,
            device=s.local_vlm_device,
        )

        try:
            logger.info("Describing {} images...", len(all_images))
            all_images = await describer.describe(all_images)
            logger.info("Image description completed")

            # 替换文本中的占位符
            self._replace_image_placeholders(loaded, all_images)
        except Exception as e:
            logger.warning("Image description failed: {}, using placeholders", e)
        finally:
            describer.close()

        return loaded

    async def _generate_doc_summary(
        self, doc: Document, loaded: "LoadedDocument"
    ) -> str:
        """为文档生成 ~150-300 字摘要。失败/正文为空时返回空串。

        摘要用途：DocIndex 用它和 title/filename 一起建 BM25 索引，
        在 chunk-level 检索前先判断"哪些文档相关"。
        """
        text = (loaded.text or "").strip()
        if not text:
            return ""

        # 取前 ~3000 字作为摘要输入；超长文档不需要把全文都喂给 LLM。
        preview = text[:3000]
        if len(text) > 3000:
            preview += "\n...（后续已截断）"

        prompt = DOC_SUMMARY_PROMPT.format(
            filename=doc.filename, title=doc.title, preview=preview
        )
        try:
            summary = await LLMFactory.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )
        except Exception as e:
            logger.warning("doc summary LLM failed for doc={}: {}", doc.id, e)
            return ""

        summary = (summary or "").strip()
        # 模型偶尔只回 "EMPTY" —— 当作"这份文档没有可用摘要"。
        if not summary or summary.upper() == "EMPTY":
            return ""
        # 极端 case：模型回一段 prompt 模板或太长，截断到 ~600 字避免 DB 体积膨胀。
        if len(summary) > 600:
            summary = summary[:600]
        return summary

    def _replace_image_placeholders(
        self, loaded: "LoadedDocument", images: list[ImageInfo]
    ) -> None:
        """将图片占位符替换为描述文本。"""
        # 构建索引映射
        desc_map = {i + 1: img.description for i, img in enumerate(images)}

        # 替换每个页面中的占位符
        for page in loaded.pages:
            text = page.text
            for idx, desc in desc_map.items():
                placeholder = f"[图片:{idx}]"
                if placeholder in text:
                    replacement = f"[图片: {desc}]"
                    text = text.replace(placeholder, replacement)
            page.text = text

        # 更新完整文本
        loaded.text = "\n\n".join(p.text for p in loaded.pages if p.text)

    def _ingest_chunks(
        self, kb: KnowledgeBase, doc: Document, loaded: "LoadedDocument"
    ) -> None:
        """切片并入库。

        父子两段切片（ParentChildSplitter）：
        - chunks 入库（Chroma + BM25）的都是 child（短，~250 字），用于 embedding/BM25 检索；
        - child metadata 里带 parent_id / parent_index / parent_text，
          retriever 在 rerank 后做 parent_collapse（同一 parent 的多个 child 只保留分数最高那个，
          text 用 parent 全文替换）—— 这样 LLM 看到的是完整语义段，而不是孤立的 250 字片段。
        """
        splitter = build_splitter(chunk_size=kb.chunk_size, chunk_overlap=kb.chunk_overlap)
        chunks = splitter.split(loaded)
        if not chunks:
            doc.chunk_count = 0
            doc.parent_count = 0
            kb.chunk_count = len(BM25Store.for_kb(kb.id))
            return

        ids = [str(uuid4()) for _ in chunks]
        texts = [c.text for c in chunks]
        metadatas: list[dict] = []
        for c in chunks:
            md = {
                "doc_id": doc.id,
                "doc_title": doc.title,
                "doc_filename": doc.filename,
                "page": c.page,
                "section": c.section or "",
                # 父子 chunk 元数据（ChildChunk 才有，普通 TextChunk 没有）
                "parent_id": getattr(c, "parent_id", "") or "",
                "parent_index": int(getattr(c, "parent_index", 0) or 0),
                "child_index": int(getattr(c, "child_index", 0) or 0),
                # parent 完整文本：retriever 在 parent_collapse 阶段用它替换 child.text
                "parent_text": getattr(c, "parent_text", "") or "",
            }
            if c.metadata:
                md.update({k: v for k, v in c.metadata.items() if v is not None})
            metadatas.append(md)

        model_name, dim = resolve_embedding(kb)
        embedding = EmbeddingFactory.get(
            model_name=model_name,
            dim=dim,
        )
        store = ChromaStore(
            collection_name=kb.collection_name,
            embedding=embedding,
        )
        store.add(ids=ids, texts=texts, metadatas=metadatas)

        bm25 = BM25Store.for_kb(kb.id)
        bm25.add(
            [
                BM25Doc(chunk_id=i, text=t, metadata=m)
                for i, t, m in zip(ids, texts, metadatas, strict=False)
            ]
        )

        # 统计 parent 数（去重后），比 chunk_count 少（每 parent 有 3-5 child）
        parent_ids = {md.get("parent_id") for md in metadatas if md.get("parent_id")}
        doc.chunk_count = len(chunks)
        doc.parent_count = len(parent_ids) if parent_ids else len(chunks)
        kb.chunk_count = bm25_count = len(bm25)
        logger.info(
            "ingested doc={} kb={} children={} parents={} bm25_total={}",
            doc.id,
            kb.id,
            len(chunks),
            doc.parent_count,
            bm25_count,
        )

    async def delete_document(self, doc: Document) -> None:
        kb_id = doc.knowledge_base_id
        kb = await self.get_kb(kb_id)
        if not kb:
            return

        # 先清外部副作用；SQLAlchemy 一旦 session.delete() 就会级联，这之后对象就过期了。
        ChromaStore(collection_name=kb.collection_name).delete_by_doc_id(doc.id)
        bm25 = BM25Store.for_kb(kb.id)
        bm25.delete_by_doc_id(doc.id)

        file_path = Path(doc.file_path)
        await self.session.delete(doc)
        try:
            await self.session.flush()
        except Exception:
            await self.session.rollback()
            raise

        kb.chunk_count = len(bm25)
        await self._refresh_doc_count(kb)
        await self.session.commit()
        # 文档被删，DocIndex 该 kb 的 doc list 已变化。
        from app.services.doc_index import DocIndex

        try:
            DocIndex.invalidate(kb_id)
        except Exception as e:  # pragma: no cover - cache best-effort
            logger.warning("DocIndex.invalidate failed: {}", e)
        try:
            file_path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("failed to delete source file {}: {}", file_path, e)

    async def retry_document(self, doc: Document) -> Document:
        """重新入库一个失败 / 待入库文档。"""
        kb = await self.get_kb(doc.knowledge_base_id)
        if not kb:
            raise ValueError(f"knowledge base {doc.knowledge_base_id} not found")
        # 清理旧的 chunks
        ChromaStore(collection_name=kb.collection_name).delete_by_doc_id(doc.id)
        BM25Store.for_kb(kb.id).delete_by_doc_id(doc.id)
        doc.chunk_count = 0
        doc.parent_count = 0
        doc.error = None
        doc.status = DocumentStatus.processing
        await self.session.commit()
        await self.session.refresh(doc)

        # 重新跑解析
        try:
            loaded = await asyncio.to_thread(DocumentLoaderFactory.load, doc.file_path)
            if loaded.has_images():
                loaded = await self._process_images(loaded)
            await asyncio.to_thread(self._ingest_chunks, kb, doc, loaded)
            # 重新生成文档级摘要（文件可能已变 / 之前生成失败）
            doc.summary = await self._generate_doc_summary(doc, loaded)
            doc.status = DocumentStatus.ready
        except Exception as e:
            logger.exception("retry ingest failed: {}", e)
            try:
                ChromaStore(collection_name=kb.collection_name).delete_by_doc_id(doc.id)
                bm25 = BM25Store.for_kb(kb.id)
                bm25.delete_by_doc_id(doc.id)
                kb.chunk_count = len(bm25)
            except Exception as cleanup_error:
                logger.warning(
                    "failed to clean partial ingest for doc={}: {}", doc.id, cleanup_error
                )
            doc.chunk_count = 0
            doc.parent_count = 0
            doc.summary = ""
            doc.status = DocumentStatus.failed
            doc.error = str(e)[:2000]
        finally:
            await self._refresh_doc_count(kb)
            await self.session.commit()
            await self.session.refresh(doc)
            from app.services.doc_index import DocIndex

            try:
                DocIndex.invalidate(kb.id)
            except Exception as e:  # pragma: no cover - cache best-effort
                logger.warning("DocIndex.invalidate failed: {}", e)
        return doc

    async def delete_kb(self, kb: KnowledgeBase) -> None:
        from app.models import Document as DocModel

        # 先逐条删除文档
        kb = await self.session.scalar(
            select(KnowledgeBase)
            .options(selectinload(KnowledgeBase.documents))
            .where(KnowledgeBase.id == kb.id)
        )
        if kb is None:
            return

        doc_ids = [d.id for d in kb.documents]

        for doc_id in doc_ids:
            doc = await self.session.get(DocModel, doc_id)
            if doc is None:
                continue
            try:
                p = Path(doc.file_path)
                if p.exists():
                    p.unlink()
            except OSError as e:
                logger.warning("failed to delete source file {}: {}", p, e)
            await self.session.delete(doc)

        try:
            await self.session.flush()
        except Exception:
            await self.session.rollback()
            raise

        ChromaStore(collection_name=kb.collection_name).reset()
        BM25Store.for_kb(kb.id).reset()

        upload_root = get_settings().upload_dir.resolve()
        kb_dir = upload_root / kb.slug
        if kb_dir.exists():
            for child in kb_dir.iterdir():
                try:
                    if child.is_file():
                        child.unlink()
                except OSError as e:
                    logger.warning("failed to delete upload {}: {}", child, e)
            try:
                kb_dir.rmdir()
            except OSError:
                pass

        await self.session.delete(kb)
        await self.session.commit()
        BM25Store._cache.pop(kb.id, None)
        from app.services.doc_index import DocIndex

        try:
            DocIndex.invalidate(kb.id)
        except Exception as e:  # pragma: no cover - cache best-effort
            logger.warning("DocIndex.invalidate failed: {}", e)

    @staticmethod
    def move_upload_to_kb_dir(
        tmp_path: str | Path,
        kb_slug: str,
        original_filename: str | None = None,
    ) -> Path:
        s = get_settings()
        target_dir = s.upload_dir / kb_slug
        target_dir.mkdir(parents=True, exist_ok=True)
        src = Path(tmp_path)
        safe_name = Path(original_filename or "").name
        dest = target_dir / (safe_name or src.name)
        if dest.exists():
            stem, suffix = dest.stem, dest.suffix
            i = 1
            while True:
                candidate = target_dir / f"{stem}_{i}{suffix}"
                if not candidate.exists():
                    dest = candidate
                    break
                i += 1
        shutil.move(str(src), str(dest))
        return dest