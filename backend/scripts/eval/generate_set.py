#!/usr/bin/env python3
"""从现有文档/chunks 自动生成 (question, doc_id, chunk_ids) 评估三元组。

设计思路：
- 用 LLM 对每个 doc 的代表性 chunk 段反向生成问题 → 避免人工标
- 每个问题保证 ground truth chunk 已知（生成问题的源 chunk）
- 多种 qtype 让评估覆盖检索的不同能力面
  - factual: 单一事实查询
  - lookup: 数字/名词等精确定位
  - reasoning: 需要综合上下文
  - multi-hop: 跨多 chunk 综合

输出：scripts/eval/data/eval_set.jsonl
每行：
  {"qid": "q0001", "question": "...", "kb_id": 2, "doc_id": 5,
   "gold_chunk_ids": ["uuid1", "uuid2"], "qtype": "factual", "source": "auto"}

用法：
  python -m scripts.eval.generate_set                       # 默认 30 题
  python -m scripts.eval.generate_set --per-doc 4           # 每 doc 4 题
  python -m scripts.eval.generate_set --kb 2                # 只对 KB 2
  python -m scripts.eval.generate_set --out data/my_set.jsonl

注意：
- chunk 取自 Chroma（按 chunk_index 排序后的中段）；
- LLM 生成可能失败，会跳过失败 chunk 并 continue；
- 生成的问题质量有限制，作为初版基线，后续可手改 / 补充真实问题。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from collections import defaultdict
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from sqlalchemy import select  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.db import AsyncSessionLocal  # noqa: E402
from app.llm.factory import LLMFactory  # noqa: E402
from app.models import Document, DocumentStatus, KnowledgeBase  # noqa: E402
from app.vectorstore import ChromaStore  # noqa: E402


# LLM 用的提示词：让模型基于给定 chunk 段反向生成 1 个搜索式问题。
# 要求问题「像用户在 KB 里搜的」而不是「像考试题」。
_GEN_QUESTION_PROMPT = """你是检索评估集构造助手。给你一段文档切片，请你站在"想从知识库里查到这个信息的用户"的角度，反向写出 1 个搜索式问题。

要求：
1. 问题必须能由【给定段落】回答（不要超出段落范围）
2. 用自然的口吻提问，像用户在搜索框里输入的查询，不要带"请回答"这种客套
3. 长度 8-30 字（中文）
4. 不要引用段落编号、不要复述原文
5. 输出严格 JSON，键：question / qtype
   - qtype ∈ ["factual", "lookup", "reasoning", "multi_hop"]
   - factual: 直接事实（如"瀑布模型的定义"）
   - lookup: 精确定位（如"瀑布模型的 4 个阶段分别是什么"）
   - reasoning: 综合上下文（如"什么时候不适合用瀑布模型"）
   - multi_hop: 跨段综合（如"对比瀑布模型和增量模型的适用场景"）

段落：
\"\"\"
{chunk_text}
\"\"\"

只输出 JSON，不要其他内容。"""


# multi-hop 提示词：基于同一 doc 的两个相关 chunk 拼成跨段问题
_GEN_MULTIHOP_PROMPT = """你是检索评估集构造助手。给你同一份文档的两个切片，请你站在"想从知识库里查到这个信息的用户"的角度，反向写出 1 个跨段综合问题（用户需要同时看到这两段才能回答）。

要求：
1. 问题必须综合【两段】内容（不能只由任一段单独回答）
2. 用自然的口吻提问
3. 长度 12-40 字（中文）
4. 严格 JSON：question / qtype 两个键

段落 A：
\"\"\"
{chunk_a}
\"\"\"

段落 B：
\"\"\"
{chunk_b}
\"\"\"

只输出 JSON。"""


async def _gen_one_question(text: str) -> dict | None:
    """调用 LLM 生成一个 (question, qtype)。失败返回 None。"""
    raw = await LLMFactory.chat(
        messages=[{"role": "user", "content": _GEN_QUESTION_PROMPT.format(chunk_text=text[:1500])}],
        temperature=0.7,  # 高一点让问题多样化
    )
    # 解析 JSON（容忍前后有非 JSON 噪声）
    s, e = raw.find("{"), raw.rfind("}")
    if s < 0 or e < 0:
        return None
    try:
        data = json.loads(raw[s : e + 1])
    except json.JSONDecodeError:
        return None
    q = data.get("question", "").strip()
    qt = data.get("qtype", "").strip()
    if not q or qt not in ("factual", "lookup", "reasoning", "multi_hop"):
        return None
    return {"question": q, "qtype": qt}


async def _gen_multihop(text_a: str, text_b: str) -> dict | None:
    raw = await LLMFactory.chat(
        messages=[
            {
                "role": "user",
                "content": _GEN_MULTIHOP_PROMPT.format(
                    chunk_a=text_a[:1000], chunk_b=text_b[:1000]
                ),
            }
        ],
        temperature=0.7,
    )
    s, e = raw.find("{"), raw.rfind("}")
    if s < 0 or e < 0:
        return None
    try:
        data = json.loads(raw[s : e + 1])
    except json.JSONDecodeError:
        return None
    q = data.get("question", "").strip()
    if not q:
        return None
    return {"question": q, "qtype": "multi_hop"}


async def _sample_chunks_for_doc(chroma: ChromaStore, doc_id: int, n: int) -> list[dict]:
    """从 doc 中均匀取 n 个 chunk（中段优先；首尾常是封面/章节名）。"""
    all_chunks = chroma.get(where={"doc_id": doc_id})
    if not all_chunks:
        return []
    # 按 chunk_index 排序（缺失则保持原顺序）
    all_chunks.sort(
        key=lambda c: (c.get("metadata", {}).get("chunk_index", 1 << 30))
    )
    if len(all_chunks) <= n:
        return all_chunks
    # 均匀采样：跳过前后 10%
    start = max(0, len(all_chunks) // 10)
    end = min(len(all_chunks), len(all_chunks) - len(all_chunks) // 10)
    pool = all_chunks[start:end]
    step = max(1, len(pool) // n)
    return pool[::step][:n]


async def generate(
    per_doc: int = 3,
    kb_filter: int | None = None,
    out_path: Path | None = None,
) -> list[dict]:
    settings = get_settings()
    settings.ensure_dirs()
    if out_path is None:
        out_path = _BACKEND_DIR / "scripts" / "eval" / "data" / "eval_set.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 选 ready docs；预先把 KB collection_name 读出来，避免 session 关闭后懒加载失败。
    async with AsyncSessionLocal() as session:
        stmt = select(Document).where(Document.status == DocumentStatus.ready)
        if kb_filter is not None:
            stmt = stmt.where(Document.knowledge_base_id == kb_filter)
        # joinedload 把 KB 一起拉进来，关 session 后仍可访问
        from sqlalchemy.orm import joinedload
        stmt = stmt.options(joinedload(Document.knowledge_base))
        docs = (await session.execute(stmt)).scalars().all()
        # 提前固化为 (id, kb_id, kb_collection_name, title) 列表，避免后续访问 detached attr
        doc_meta = [
            {
                "id": d.id,
                "kb_id": d.knowledge_base_id,
                "kb_collection": d.knowledge_base.collection_name if d.knowledge_base else None,
                "title": d.title,
            }
            for d in docs
        ]

    if not doc_meta:
        print("no ready docs found", file=sys.stderr)
        return []

    print(f"generating questions from {len(doc_meta)} docs, {per_doc} per doc...")

    eval_set: list[dict] = []
    counter = 0

    for d in doc_meta:
        kb_collection = d["kb_collection"]
        if not kb_collection:
            continue
        chroma = ChromaStore(collection_name=kb_collection)
        sampled = await _sample_chunks_for_doc(chroma, d["id"], per_doc)
        if not sampled:
            continue

        # 单 chunk → factual/lookup/reasoning
        for c in sampled:
            try:
                q = await _gen_one_question(c["text"])
            except Exception as e:
                print(f"  doc {d['id']} chunk {c['id']}: gen failed: {e}", file=sys.stderr)
                q = None
            if not q:
                continue
            counter += 1
            eval_set.append(
                {
                    "qid": f"q{counter:04d}",
                    "question": q["question"],
                    "kb_id": d["kb_id"],
                    "doc_id": d["id"],
                    "doc_title": d["title"],
                    "gold_chunk_ids": [c["id"]],
                    "qtype": q["qtype"],
                    "source": "auto",
                }
            )

        # 跨 chunk → multi_hop（取采样里两段）
        if len(sampled) >= 2:
            try:
                q = await _gen_multihop(sampled[0]["text"], sampled[-1]["text"])
            except Exception as e:
                q = None
            if q:
                counter += 1
                eval_set.append(
                    {
                        "qid": f"q{counter:04d}",
                        "question": q["question"],
                        "kb_id": d["kb_id"],
                        "doc_id": d["id"],
                        "doc_title": d["title"],
                        "gold_chunk_ids": [sampled[0]["id"], sampled[-1]["id"]],
                        "qtype": "multi_hop",
                        "source": "auto",
                    }
                )

        # 写盘（增量）
        with out_path.open("w", encoding="utf-8") as f:
            for item in eval_set:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nGenerated {len(eval_set)} questions.")
    by_type = defaultdict(int)
    for it in eval_set:
        by_type[it["qtype"]] += 1
    for qt, n in sorted(by_type.items()):
        print(f"  {qt}: {n}")
    print(f"saved to {out_path}")
    return eval_set


def main() -> None:
    p = argparse.ArgumentParser(description="Generate RAG eval set from existing docs")
    p.add_argument("--per-doc", type=int, default=3, help="单 chunk 问题每 doc 数量")
    p.add_argument("--kb", type=int, default=None, help="只对指定 KB")
    p.add_argument("--out", type=Path, default=None, help="输出 jsonl 路径")
    args = p.parse_args()
    asyncio.run(generate(per_doc=args.per_doc, kb_filter=args.kb, out_path=args.out))


if __name__ == "__main__":
    main()