#!/usr/bin/env python3
"""把所有 ready 文档用新 pipeline（ParentChildSplitter）重新入库。

用法：
    python -m scripts.reingest_all              # 重做所有 KB 的所有 ready 文档
    python -m scripts.reingest_all --kb 5       # 只重做 KB 5
    python -m scripts.reingest_all --doc 12     # 只重做 doc 12

注意：
- 会删除 Chroma + BM25 中该文档的所有旧 chunk，再重新入库。
- DocIndex 会自动 invalidate（ingest_path 已实现）。
- 状态置为 processing → 跑完置为 ready。失败置为 failed。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# 把 backend 目录加到 sys.path，方便直接 `python -m scripts.reingest_all`
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from sqlalchemy import select  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.db import AsyncSessionLocal  # noqa: E402
from app.models import Document, DocumentStatus, KnowledgeBase  # noqa: E402
from app.services.ingest import IngestService  # noqa: E402


async def reingest(kb_filter: int | None = None, doc_filter: int | None = None) -> None:
    settings = get_settings()
    settings.ensure_dirs()

    async with AsyncSessionLocal() as session:
        stmt = select(Document).where(Document.status == DocumentStatus.ready)
        if kb_filter is not None:
            stmt = stmt.where(Document.knowledge_base_id == kb_filter)
        if doc_filter is not None:
            stmt = stmt.where(Document.id == doc_filter)
        docs = (await session.execute(stmt)).scalars().all()

        if not docs:
            print("no documents to re-ingest")
            return

        print(f"re-ingesting {len(docs)} document(s)...")
        svc = IngestService(session)
        ok = 0
        failed = 0
        for d in docs:
            kb = await session.get(KnowledgeBase, d.knowledge_base_id)
            if not kb:
                print(f"  doc {d.id}: kb {d.knowledge_base_id} not found, skip")
                continue
            print(f"  doc {d.id} ({d.filename}) kb={d.knowledge_base_id}...")
            try:
                await svc.retry_document(d)
                ok += 1
                print(f"    -> done (chunks={d.chunk_count}, parents={d.parent_count})")
            except Exception as e:  # pragma: no cover - defensive
                failed += 1
                print(f"    -> failed: {e}")
        print(f"summary: ok={ok} failed={failed}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-ingest documents with new pipeline")
    parser.add_argument("--kb", type=int, default=None, help="filter by KB id")
    parser.add_argument("--doc", type=int, default=None, help="filter by doc id")
    args = parser.parse_args()
    asyncio.run(reingest(kb_filter=args.kb, doc_filter=args.doc))


if __name__ == "__main__":
    main()