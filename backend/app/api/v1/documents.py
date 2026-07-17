"""文档管理路由：上传 / 列表 / 删除。"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path as FsPath
from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Path,
    Query,
    UploadFile,
    status,
)
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.core.deps import get_db
from app.models import Document, KnowledgeBase
from app.schemas.chunk import ChunkDetail, ChunkListItem, ContentSegment, DocumentContent
from app.schemas.document import (
    BatchUploadResponse,
    DocumentListResponse,
    DocumentRead,
    UploadResponse,
)
from app.services.ingest import IngestService
from app.vectorstore import ChromaStore

router = APIRouter(prefix="/knowledge-bases/{kb_id}/documents", tags=["documents"])

KbId = Annotated[int, Path(ge=1)]
DbSession = Annotated[AsyncSession, Depends(get_db)]


async def _get_kb_or_404(session: AsyncSession, kb_id: int) -> KnowledgeBase:
    kb = await session.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="knowledge_base not found")
    return kb


async def _save_upload_to_tmp(file: UploadFile) -> tuple[FsPath, str]:
    settings = get_settings()
    suffix = FsPath(file.filename or "").suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise HTTPException(status_code=415, detail=f"不支持的文件类型: {suffix}")
    tmp = tempfile.NamedTemporaryFile(prefix="upload_", suffix=suffix, delete=False)
    try:
        size = 0
        max_bytes = settings.max_upload_mb * 1024 * 1024
        chunk_size = 1024 * 1024
        while True:
            data = await file.read(chunk_size)
            if not data:
                break
            size += len(data)
            if size > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"文件过大，最大允许 {settings.max_upload_mb} MB",
                )
            tmp.write(data)
        tmp.close()
        return FsPath(tmp.name), suffix
    except Exception:
        if not tmp.closed:
            tmp.close()
        FsPath(tmp.name).unlink(missing_ok=True)
        raise


async def _ingest_async(
    session_factory, kb_id: int, doc_id: int
) -> None:
    """后台处理已经提交的 Document，避免重复建记录和事务锁。"""
    async with session_factory() as session:
        doc: Document | None = None
        try:
            doc = await session.scalar(
                select(Document).where(
                    Document.id == doc_id,
                    Document.knowledge_base_id == kb_id,
                )
            )
            if not doc:
                logger.warning(
                    "ingest_async: doc={} in kb={} disappeared",
                    doc_id,
                    kb_id,
                )
                return
            await IngestService(session).retry_document(doc)
        except Exception as e:
            await session.rollback()
            logger.exception("async ingest failed for kb={} doc={}", kb_id, doc_id)
            if doc is not None:
                try:
                    failed_doc = await session.get(Document, doc_id)
                    if failed_doc is not None:
                        failed_doc.status = "failed"
                        failed_doc.error = str(e)[:2000]
                        await session.commit()
                except Exception:
                    await session.rollback()
                    logger.exception("failed to persist ingest error for doc={}", doc_id)


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    kb_id: KbId,
    session: DbSession,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    status_filter: str | None = Query(default=None, alias="status"),
) -> DocumentListResponse:
    await _get_kb_or_404(session, kb_id)
    stmt = (
        select(Document)
        .where(Document.knowledge_base_id == kb_id)
        .order_by(Document.id.desc())
        .limit(limit)
        .offset(offset)
    )
    if status_filter:
        stmt = stmt.where(Document.status == status_filter)
    res = await session.execute(stmt)
    items = res.scalars().all()
    # total via simple count query
    from sqlalchemy import func

    total = await session.scalar(
        select(func.count(Document.id)).where(Document.knowledge_base_id == kb_id)
    )
    return DocumentListResponse(
        items=[DocumentRead.model_validate(d) for d in items],
        total=int(total or 0),
    )


@router.post("", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    kb_id: KbId,
    background: BackgroundTasks,
    session: DbSession,
    file: UploadFile = File(...),
) -> UploadResponse:
    kb = await _get_kb_or_404(session, kb_id)
    # 提前把不会变的字段缓存下来：commit/refresh 之后再读 mapped 属性
    # 容易在 async session 里触发 MissingGreenlet 懒加载错误。
    kb_id_value: int = kb.id
    kb_slug_value: str = kb.slug
    tmp_path, suffix = await _save_upload_to_tmp(file)
    filename = FsPath(file.filename or "").name or f"upload{suffix}"
    stored_path: FsPath | None = None
    try:
        stored_path = IngestService.move_upload_to_kb_dir(
            tmp_path,
            kb_slug_value,
            filename,
        )
        doc = Document(
            knowledge_base_id=kb_id_value,
            title=FsPath(filename).stem,
            filename=filename,
            file_path=str(stored_path),
            file_ext=suffix,
            file_size=stored_path.stat().st_size,
            mime_type=file.content_type or "",
            status="processing",
        )
        session.add(doc)
        await session.commit()
        await session.refresh(doc)
    except Exception:
        await session.rollback()
        tmp_path.unlink(missing_ok=True)
        if stored_path is not None:
            stored_path.unlink(missing_ok=True)
        raise

    from app.core.db import AsyncSessionLocal

    background.add_task(_ingest_async, AsyncSessionLocal, kb_id_value, doc.id)
    return UploadResponse(
        document=DocumentRead.model_validate(doc),
        accepted=True,
        message="已接收，正在后台入库",
    )


@router.post("/batch", response_model=BatchUploadResponse)
async def upload_documents_batch(
    kb_id: KbId,
    background: BackgroundTasks,
    session: DbSession,
    files: list[UploadFile] = File(...),
) -> BatchUploadResponse:
    kb = await _get_kb_or_404(session, kb_id)
    # 提前把 kb 的字段缓存到本地变量：循环里每次 commit 之后若再读 kb.slug / kb.id，
    # 会触发懒加载（MissingGreenlet，因为 async session 的同步 IO 需要 greenlet 上下文）。
    # slug 和 id 在本次请求里不会变，直接用本地变量最稳。
    kb_id_value: int = kb.id
    kb_slug_value: str = kb.slug
    accepted: list[DocumentRead] = []
    failed: list[dict] = []
    from app.core.db import AsyncSessionLocal

    for f in files:
        tmp_path: FsPath | None = None
        stored_path: FsPath | None = None
        try:
            tmp_path, suffix = await _save_upload_to_tmp(f)
            filename = FsPath(f.filename or "").name or f"upload{suffix}"
            stored_path = IngestService.move_upload_to_kb_dir(
                tmp_path,
                kb_slug_value,
                filename,
            )
            doc = Document(
                knowledge_base_id=kb_id_value,
                title=FsPath(filename).stem,
                filename=filename,
                file_path=str(stored_path),
                file_ext=suffix,
                file_size=stored_path.stat().st_size,
                mime_type=f.content_type or "",
                status="processing",
            )
            session.add(doc)
            await session.commit()
            await session.refresh(doc)
            accepted.append(DocumentRead.model_validate(doc))
            background.add_task(_ingest_async, AsyncSessionLocal, kb_id_value, doc.id)
        except HTTPException as e:
            await session.rollback()
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            if stored_path is not None:
                stored_path.unlink(missing_ok=True)
            failed.append({"filename": f.filename, "error": e.detail})
        except Exception as e:
            await session.rollback()
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            if stored_path is not None:
                stored_path.unlink(missing_ok=True)
            logger.exception("batch upload failed")
            failed.append({"filename": f.filename, "error": str(e)})
    return BatchUploadResponse(documents=accepted, failed=failed)


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    kb_id: KbId,
    doc_id: Annotated[int, Path(ge=1)],
    session: DbSession,
) -> None:
    await _get_kb_or_404(session, kb_id)
    stmt = (
        select(Document)
        .where(Document.id == doc_id, Document.knowledge_base_id == kb_id)
        .options(selectinload(Document.knowledge_base))
    )
    res = await session.execute(stmt)
    doc = res.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="document not found")
    await IngestService(session).delete_document(doc)


@router.post("/{doc_id}/retry", response_model=DocumentRead)
async def retry_document(
    kb_id: KbId,
    doc_id: Annotated[int, Path(ge=1)],
    session: DbSession,
) -> DocumentRead:
    """重新入库失败 / processing 的文档。同步执行；返回时已完成（或再次失败）。"""
    kb = await _get_kb_or_404(session, kb_id)
    kb_slug_value: str = kb.slug
    doc = await session.scalar(
        select(Document).where(
            Document.id == doc_id,
            Document.knowledge_base_id == kb_id,
        )
    )
    if not doc:
        raise HTTPException(status_code=404, detail="document not found")
    if doc.status.value == "ready":
        return DocumentRead.model_validate(doc)

    source_path = FsPath(doc.file_path)
    upload_root = get_settings().upload_dir.resolve()
    if source_path.exists() and not source_path.resolve().is_relative_to(upload_root):
        stored_path = IngestService.move_upload_to_kb_dir(
            source_path,
            kb_slug_value,
            doc.filename,
        )
        doc.file_path = str(stored_path)
        await session.commit()
        await session.refresh(doc)

    updated = await IngestService(session).retry_document(doc)
    return DocumentRead.model_validate(updated)


# ---------- chunks (按文档列切片 / 按 chunk_id 查详情) ----------


@router.get("/{doc_id}/chunks", response_model=list[ChunkListItem])
async def list_document_chunks(
    kb_id: KbId,
    doc_id: Annotated[int, Path(ge=1)],
    session: DbSession,
) -> list[ChunkListItem]:
    """返回该文档在 Chroma 中的全部切片概要，按 chunk_index 升序。"""
    await _get_kb_or_404(session, kb_id)
    doc = await session.get(Document, doc_id)
    if not doc or doc.knowledge_base_id != kb_id:
        raise HTTPException(status_code=404, detail="document not found")

    kb = await session.get(KnowledgeBase, kb_id)
    rows = ChromaStore(collection_name=kb.collection_name).get(where={"doc_id": doc_id})
    # 入库时若存了 metadata.chunk_index 则按它排序；缺则保持 Chroma 返回顺序
    rows.sort(
        key=lambda r: (
            int(r["metadata"].get("chunk_index", 0))
            if str(r["metadata"].get("chunk_index", "")).isdigit()
            else 0
        )
    )
    return [
        ChunkListItem(
            chunk_id=r["id"],
            doc_id=doc_id,
            kb_id=kb_id,
            length=len(r["text"] or ""),
            preview=(r["text"] or "")[:200],
            page=r["metadata"].get("page"),
            section=r["metadata"].get("section"),
            document=doc.title,
            chunk_index=(
                int(r["metadata"]["chunk_index"])
                if str(r["metadata"].get("chunk_index", "")).isdigit()
                else None
            ),
        )
        for r in rows
    ]


@router.get("/{doc_id}/chunks/{chunk_id}", response_model=ChunkDetail)
async def get_document_chunk(
    kb_id: KbId,
    doc_id: Annotated[int, Path(ge=1)],
    chunk_id: str,
    session: DbSession,
) -> ChunkDetail:
    """按 chunk_id 查详情（完整 text + 元数据），用于引用 chip 抽屉。"""
    await _get_kb_or_404(session, kb_id)
    doc = await session.get(Document, doc_id)
    if not doc or doc.knowledge_base_id != kb_id:
        raise HTTPException(status_code=404, detail="document not found")

    kb = await session.get(KnowledgeBase, kb_id)
    rows = ChromaStore(collection_name=kb.collection_name).get(ids=[chunk_id])
    if not rows:
        raise HTTPException(status_code=404, detail="chunk not found")
    r = rows[0]
    return ChunkDetail(
        chunk_id=r["id"],
        doc_id=doc_id,
        kb_id=kb_id,
        text=r["text"] or "",
        page=r["metadata"].get("page"),
        section=r["metadata"].get("section"),
        document=doc.title,
    )


def _chunk_index_key(row: dict) -> int:
    """从 metadata 取 chunk_index 作排序键；缺失/非数字则视为 0。"""
    raw = row["metadata"].get("chunk_index", "")
    return int(raw) if str(raw).isdigit() else 0


@router.get("/{doc_id}/images-debug")
async def debug_document_images(
    kb_id: KbId,
    doc_id: Annotated[int, Path(ge=1)],
    session: DbSession,
) -> dict:
    """调试：对原始文件重新跑一遍解析（不写库），列出每张图的描述长度/前 80 字。

    用于排查「[图片:N] 替换进去的到底是什么」—— 答案对答 RAG、UI 提示词是否生效很关键。
    重 loader 调用本地 CLI / Python 库，单 PDF 通常 1-5 秒；不修改数据库。
    """
    await _get_kb_or_404(session, kb_id)
    doc = await session.get(Document, doc_id)
    if not doc or doc.knowledge_base_id != kb_id:
        raise HTTPException(status_code=404, detail="document not found")

    path = FsPath(doc.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"original file not found: {path}")

    from app.loaders import DocumentLoaderFactory

    settings = get_settings()
    if path.suffix.lower() not in settings.allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"file type {path.suffix} not supported by current loaders",
        )

    try:
        loaded = await asyncio.to_thread(DocumentLoaderFactory.load, path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"loader failed: {e}") from e

    images_meta: list[dict] = []
    for idx, img in enumerate(loaded.all_images(), 1):
        # 1-based 占位符序号，与 ingest 替换时一致
        placeholder = f"[图片:{idx}]"
        images_meta.append(
            {
                "index": idx,
                "placeholder": placeholder,
                "mime_type": img.mime_type,
                "size_bytes": len(img.image_bytes),
                "alt_text": img.alt_text,
                "description": img.description or "",
                "description_chars": len(img.description or ""),
                "description_failed": (img.description or "")
                in ("", "[图片]", "[图片描述生成失败]"),
            }
        )

    # 占位符出现次数（与 loader 抽取的图数对比 → 多/缺 都可能）
    placeholder_counts: dict[str, int] = {}
    for p in loaded.to_pages():
        for line in p.text.splitlines():
            if "[图片:" in line:
                # 简化：每次出现都记一次；具体编号在 images_meta 中查
                placeholder_counts[line.strip()] = (
                    placeholder_counts.get(line.strip(), 0) + 1
                )

    return {
        "doc_id": doc.id,
        "filename": doc.filename,
        "ext": doc.file_ext,
        "loader": type(loaded).__name__ if False else type(loaded).__class__.__name__,
        "has_images": loaded.has_images(),
        "image_count": len(images_meta),
        "settings_image_cap": settings.max_images_per_doc,
        "images": images_meta,
        "page_count": len(loaded.pages),
        "placeholder_lines_in_text": placeholder_counts,
    }


@router.get("/{doc_id}/content", response_model=DocumentContent)
async def get_document_content(
    kb_id: KbId,
    doc_id: Annotated[int, Path(ge=1)],
    session: DbSession,
) -> DocumentContent:
    """「原文」视图：把该文档全部切片按 chunk_index 升序拼成连续全文。

    与 /chunks（切分视图）同源（都读 Chroma），便于前端对照「解析全文 vs 切分边界」。
    """
    await _get_kb_or_404(session, kb_id)
    doc = await session.get(Document, doc_id)
    if not doc or doc.knowledge_base_id != kb_id:
        raise HTTPException(status_code=404, detail="document not found")

    kb = await session.get(KnowledgeBase, kb_id)
    rows = ChromaStore(collection_name=kb.collection_name).get(where={"doc_id": doc_id})
    rows.sort(key=_chunk_index_key)

    segments = [
        ContentSegment(
            chunk_index=(
                int(r["metadata"]["chunk_index"])
                if str(r["metadata"].get("chunk_index", "")).isdigit()
                else None
            ),
            page=r["metadata"].get("page"),
            section=r["metadata"].get("section"),
            text=r["text"] or "",
        )
        for r in rows
    ]
    full_text = "\n\n".join(s.text for s in segments if s.text)
    return DocumentContent(
        doc_id=doc_id,
        kb_id=kb_id,
        title=doc.title,
        segments=segments,
        full_text=full_text,
    )
