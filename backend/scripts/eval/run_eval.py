#!/usr/bin/env python3
"""检索评估 runner：在多种配置下跑同一份评估集，输出对比表。

设计目标：
- 不修改生产 HybridRetriever；eval 自己实现一份「可拆解的」pipeline，
  让每个组件（vector / bm25 / rerank / collapse / doc_boost / multi_query /
  relevance_filter）独立 toggle，便于看 ablation。
- 每条问题 → 一次 retrieval → 与 gold 比对 → 累加指标。
- 输出：总览表 + 按 qtype 的明细。

用法：
  python -m scripts.eval.run_eval                       # 跑默认 ablation 矩阵
  python -m scripts.eval.run_eval --set data/eval_set.jsonl
  python -m scripts.eval.run_eval --top-k 10
  python -m scripts.eval.run_eval --quick                # 只跑 4 个核心配置

输出格式：markdown 风格表格 + 控制台对齐表。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.core.config import get_settings  # noqa: E402
from app.embeddings.factory import EmbeddingFactory  # noqa: E402
from app.llm.factory import LLMFactory  # noqa: E402
from app.rerankers import LocalReranker  # noqa: E402
from app.services.bm25_store import BM25Store  # noqa: E402
from app.services.doc_index import DocIndex  # noqa: E402
from app.services.retriever import HybridRetriever, RetrievedChunk  # noqa: E402
from app.vectorstore import ChromaStore  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.db import AsyncSessionLocal  # noqa: E402
from app.models import KnowledgeBase  # noqa: E402

# 复用 metrics 模块
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import (  # type: ignore[import-not-found]  # noqa: E402
    RetrievalHit,
    aggregate,
    compare_to_baseline,
    format_row,
)


# ============ EvalConfig 定义 ============


@dataclass
class EvalConfig:
    name: str
    description: str
    use_vector: bool = True
    use_bm25: bool = True
    use_rerank: bool = True
    use_parent_collapse: bool = True
    use_doc_boost: bool = True
    use_multi_query: bool = False
    use_relevance_filter: bool = False
    top_k: int = 10


def default_configs() -> list[EvalConfig]:
    return [
        EvalConfig(
            "vec_only",
            "纯向量检索（top-k 内不 rerank / 不 boost / 不 collapse）",
            use_vector=True,
            use_bm25=False,
            use_rerank=False,
            use_parent_collapse=False,
            use_doc_boost=False,
        ),
        EvalConfig(
            "bm25_only",
            "纯 BM25（不 rerank / 不 boost / 不 collapse）",
            use_vector=False,
            use_bm25=True,
            use_rerank=False,
            use_parent_collapse=False,
            use_doc_boost=False,
        ),
        EvalConfig(
            "hybrid_no_rerank",
            "vector + BM25 + RRF，不 rerank",
            use_rerank=False,
            use_parent_collapse=False,
            use_doc_boost=False,
        ),
        EvalConfig(
            "hybrid_rerank",
            "hybrid + rerank（不 collapse / 不 boost）",
            use_parent_collapse=False,
            use_doc_boost=False,
        ),
        EvalConfig(
            "hybrid_rerank_collapse",
            "hybrid + rerank + parent_collapse（不 boost）",
            use_doc_boost=False,
        ),
        EvalConfig(
            "full",
            "hybrid + rerank + collapse + doc_boost（生产配置）",
        ),
        EvalConfig(
            "full_multiquery",
            "full + multi-query（4 路 retrieve + 末轮 rerank）",
            use_multi_query=True,
        ),
    ]


def quick_configs() -> list[EvalConfig]:
    return [
        EvalConfig("bm25_only", "", use_vector=False, use_bm25=True, use_rerank=False, use_parent_collapse=False, use_doc_boost=False),
        EvalConfig("hybrid_no_rerank", "", use_rerank=False, use_parent_collapse=False, use_doc_boost=False),
        EvalConfig("hybrid_rerank", "", use_parent_collapse=False, use_doc_boost=False),
        EvalConfig("full", ""),
    ]


# ============ 检索 pipeline 拆解 ============


class _EvalRunner:
    """单一 KB 的检索 runner，把每个组件独立出来供 toggle。"""

    def __init__(self, kb_id: int, collection_name: str) -> None:
        self.kb_id = kb_id
        self.collection_name = collection_name
        s = get_settings()
        # 各组件懒加载
        self._chroma: ChromaStore | None = None
        self._bm25: BM25Store | None = None
        self._reranker: LocalReranker | None = None
        self._embedding = None

    @property
    def chroma(self) -> ChromaStore:
        if self._chroma is None:
            self._chroma = ChromaStore(collection_name=self.collection_name)
        return self._chroma

    @property
    def bm25(self) -> BM25Store:
        if self._bm25 is None:
            self._bm25 = BM25Store.for_kb(self.kb_id)
        return self._bm25

    @property
    def reranker(self) -> LocalReranker:
        if self._reranker is None:
            self._reranker = LocalReranker.shared()
        return self._reranker

    async def _vec_query(self, q: str, top_k: int) -> list[RetrievedChunk]:
        import asyncio as _aio
        items = await _aio.to_thread(self.chroma.query, q, top_k)
        return [
            RetrievedChunk(
                id=it.id, text=it.text, metadata=it.metadata,
                vector_score=float(it.score), source="vector",
            )
            for it in items
        ]

    async def _bm25_query(self, q: str, top_k: int) -> list[RetrievedChunk]:
        import asyncio as _aio
        hits = await _aio.to_thread(self.bm25.query, q, top_k)
        out: list[RetrievedChunk] = []
        for h in hits:
            doc, score = h  # BM25Store.query 返回 list[tuple[BM25Doc, float]]
            out.append(
                RetrievedChunk(
                    id=doc.chunk_id,
                    text=doc.text,
                    metadata=doc.metadata,
                    bm25_score=float(score),
                    source="bm25",
                )
            )
        return out

    async def _rerank_chunks(self, q: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        import asyncio as _aio
        texts = [c.text for c in chunks]
        ranked = await _aio.to_thread(self.reranker.rerank, q, texts, len(chunks))
        for idx, score in ranked:
            if 0 <= idx < len(chunks):
                chunks[idx].rerank_score = float(score)
        chunks.sort(key=lambda c: (c.rerank_score or 0.0), reverse=True)
        return chunks

    @staticmethod
    def _rrf_fuse(
        vec: list[RetrievedChunk], bm: list[RetrievedChunk], k_const: int = 60,
    ) -> list[RetrievedChunk]:
        # 借鉴 HybridRetriever._rrf_fuse：按各自原始排名 RRF 融合
        scores: dict[str, RetrievedChunk] = {}
        for c in vec:
            scores[c.id] = RetrievedChunk(
                id=c.id, text=c.text, metadata=c.metadata,
                vector_score=c.vector_score, source="vector",
            )
        for c in bm:
            if c.id in scores:
                scores[c.id].bm25_score = c.bm25_score
                scores[c.id].source = "fused"
            else:
                scores[c.id] = RetrievedChunk(
                    id=c.id, text=c.text, metadata=c.metadata,
                    bm25_score=c.bm25_score, source="bm25",
                )
        vec_rank = {c.id: r for r, c in enumerate(vec)}
        bm_rank = {c.id: r for r, c in enumerate(bm)}
        for cid, chunk in scores.items():
            rrf = 0.0
            if cid in vec_rank:
                rrf += 1.0 / (k_const + vec_rank[cid] + 1)
            if cid in bm_rank:
                rrf += 1.0 / (k_const + bm_rank[cid] + 1)
            chunk.score = rrf
        return sorted(scores.values(), key=lambda c: c.score, reverse=True)

    async def _single_query_retrieve(
        self, q: str, cfg: EvalConfig,
    ) -> list[RetrievedChunk]:
        """跑一次单 query 检索：vec/bm25 + 融合 + 可选 rerank。"""
        s = get_settings()
        # 1. 召回
        tasks = []
        if cfg.use_vector:
            tasks.append(self._vec_query(q, s.rerank_top_k))
        else:
            tasks.append(_noop_async([]))
        if cfg.use_bm25:
            tasks.append(self._bm25_query(q, s.rerank_top_k))
        else:
            tasks.append(_noop_async([]))
        vec_res, bm_res = await asyncio.gather(*tasks)

        # 2. RRF（仅当两个都用；单路直接走 rerank）
        if cfg.use_vector and cfg.use_bm25:
            fused = self._rrf_fuse(vec_res, bm_res)
        elif cfg.use_vector:
            fused = sorted(vec_res, key=lambda c: c.vector_score, reverse=True)
        else:
            fused = sorted(bm_res, key=lambda c: c.bm25_score, reverse=True)

        # 3. Rerank
        if cfg.use_rerank and fused:
            fused = await self._rerank_chunks(q, fused)
        elif fused:
            # 没 rerank 时 score 字段统一映射（保证 doc_boost 等下游能拿到 base score）
            for c in fused:
                if c.score == 0.0:
                    c.score = c.vector_score or c.bm25_score or 0.0
        return fused

    async def retrieve(self, query: str, cfg: EvalConfig) -> list[RetrievedChunk]:
        s = get_settings()

        # 1. 构造 queries（multi-query 走 LLM 改写）
        queries = [query]
        if cfg.use_multi_query:
            try:
                rewrites = await _expand_queries(query, n=3)
                queries = [query] + rewrites
            except Exception as e:
                print(f"  multi-query expand failed: {e}", file=sys.stderr)

        # 2. 每路 retrieve
        chunk_map: dict[str, RetrievedChunk] = {}
        for q in queries:
            chunks = await self._single_query_retrieve(q, cfg)
            for c in chunks:
                ex = chunk_map.get(c.id)
                base = lambda x: x.rerank_score if x.rerank_score is not None else x.score
                if ex is None or base(c) > base(ex):
                    chunk_map[c.id] = c

        # 3. 多路最终一轮 rerank（仿 agent.py 的 multi-query 路径）
        if cfg.use_multi_query and len(queries) > 1 and chunk_map:
            sorted_pre = sorted(
                chunk_map.values(),
                key=lambda c: (c.rerank_score if c.rerank_score is not None else c.score, c.score),
                reverse=True,
            )
            rerank_input_n = min(len(sorted_pre), max(s.rerank_top_k, s.final_top_k * 2))
            rerank_input = sorted_pre[:rerank_input_n]
            try:
                chunks = await self._rerank_chunks(query, rerank_input)
            except Exception as e:
                print(f"  final rerank failed: {e}", file=sys.stderr)
                chunks = sorted_pre
        else:
            chunks = sorted(
                chunk_map.values(),
                key=lambda c: (c.rerank_score if c.rerank_score is not None else c.score, c.score),
                reverse=True,
            )

        # 4. doc boost
        if cfg.use_doc_boost and chunks:
            boost_ids = await _get_boost_doc_ids(self.kb_id, query, top_k=s.doc_index_top_k)
            if boost_ids:
                for c in chunks:
                    md = c.metadata or {}
                    if md.get("doc_id") in boost_ids:
                        base = c.rerank_score if c.rerank_score is not None else c.score
                        if base is not None:
                            c.rerank_score = float(base) * 1.2
                chunks.sort(
                    key=lambda c: (c.rerank_score if c.rerank_score is not None else c.score, c.score),
                    reverse=True,
                )

        # 5. parent collapse
        if cfg.use_parent_collapse:
            # 复用生产实现的 _parent_collapse：单例 HybridRetriever 只是借用其方法
            hr = HybridRetriever(knowledge_base_id=self.kb_id, collection_name=self.collection_name)
            chunks = hr._parent_collapse(chunks)

        # 6. relevance filter（可选）
        if cfg.use_relevance_filter and len(chunks) >= 3:
            chunks = await _relevance_filter(query, chunks)

        return chunks[: cfg.top_k]


async def _noop_async(x):
    return x


# ============ multi-query / doc boost / relevance filter helpers ============


_MQ_PROMPT = """你是搜索改写助手。给定用户原始查询，生成 3 个等价改写，用于在不同表述下命中同一份文档。
要求：
- 改写保持原意，但用词/句式不同（如同义词、换说法、补关键词）
- 不要扩大语义范围、不要改写成不同问题
- 中文输出
- 严格 JSON：queries 键，值为字符串数组（3 个改写）

原始查询：{q}
只输出 JSON。"""


async def _expand_queries(q: str, n: int = 3) -> list[str]:
    raw = await LLMFactory.chat(
        messages=[{"role": "user", "content": _MQ_PROMPT.format(q=q)}],
        temperature=0.5,
    )
    s, e = raw.find("{"), raw.rfind("}")
    if s < 0 or e < 0:
        return []
    try:
        data = json.loads(raw[s : e + 1])
        return [str(x).strip() for x in data.get("queries", []) if str(x).strip()][:n]
    except json.JSONDecodeError:
        return []


async def _get_boost_doc_ids(kb_id: int, query: str, top_k: int) -> set[int]:
    try:
        idx = DocIndex.for_kb(kb_id)
        hits = await idx.query(query, top_k=top_k)
        return {h.doc_id for h in hits}
    except Exception:
        return set()


_REL_PROMPT = """你是相关性评审员。给定用户问题与候选文档片段列表，给每个片段打 relevant (true/false)。
要求：
- 严格判断：片段是否包含回答问题所需的核心信息
- 候选按 [i] 编号（1-based）
- 严格 JSON：verdicts 键，值为对象数组（每项 i + relevant 字段）

问题：{q}

候选：
{candidates}

只输出 JSON。"""


async def _relevance_filter(q: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if len(chunks) < 3:
        return chunks
    MAX = 10
    review = chunks[:MAX]
    ctx = "\n\n".join(f"[{i+1}] {(c.text or '')[:500]}" for i, c in enumerate(review))
    try:
        raw = await LLMFactory.chat(
            messages=[{"role": "user", "content": _REL_PROMPT.format(q=q, candidates=ctx)}],
            temperature=0.0,
        )
        s, e = raw.find("{"), raw.rfind("}")
        if s < 0 or e < 0:
            return chunks
        data = json.loads(raw[s : e + 1])
        verdicts = data.get("verdicts", []) if isinstance(data, dict) else []
        keep = set()
        for v in verdicts:
            if not isinstance(v, dict) or not v.get("relevant"):
                continue
            try:
                idx = int(v.get("i", 0)) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(review):
                keep.add(idx)
        if not keep:
            return chunks[:3]  # 兜底：保留 top-3
        return [review[i] for i in sorted(keep)]
    except Exception:
        return chunks


# ============ Eval 加载与执行 ============


def load_eval_set(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def build_retrieval_hit(question: dict, retrieved: list[RetrievedChunk]) -> RetrievalHit:
    gold_chunk_ids = set(question.get("gold_chunk_ids") or [])
    gold_doc_ids = {int(question["doc_id"])}
    retrieved_ids: list[str] = []
    retrieved_doc_ids: list[int] = []
    for c in retrieved:
        # parent_collapse 后的 id 是 "parent:<pid>"，但 metadata.child_id 才是真 chunk
        md = c.metadata or {}
        chunk_id_real = md.get("child_id") or c.id
        retrieved_ids.append(chunk_id_real)
        # doc_id 兼容 str / int
        try:
            retrieved_doc_ids.append(int(md.get("doc_id")))
        except (TypeError, ValueError):
            pass
    return RetrievalHit(
        retrieved_ids=retrieved_ids,
        gold_chunk_ids=gold_chunk_ids,
        gold_doc_ids=gold_doc_ids,
        retrieved_doc_ids=retrieved_doc_ids,
    )


async def _resolve_collection(kb_id: int) -> str | None:
    async with AsyncSessionLocal() as session:
        kb = await session.get(KnowledgeBase, kb_id)
        return kb.collection_name if kb else None


async def run(
    eval_set_path: Path,
    configs: list[EvalConfig],
    top_k: int = 10,
    only_qtype: str | None = None,
) -> dict:
    items = load_eval_set(eval_set_path)
    if only_qtype:
        items = [it for it in items if it.get("qtype") == only_qtype]
    if not items:
        print("eval set empty", file=sys.stderr)
        return {}

    # 按 KB 分组（不同 KB 用不同 runner；同 KB 的 query 复用同一个 runner 提效）
    by_kb: dict[int, list[dict]] = defaultdict(list)
    for it in items:
        by_kb[it["kb_id"]].append(it)

    runners: dict[int, _EvalRunner] = {}
    for kb_id in by_kb:
        col = await _resolve_collection(kb_id)
        if not col:
            print(f"  KB {kb_id} not found, skip", file=sys.stderr)
            continue
        runners[kb_id] = _EvalRunner(kb_id=kb_id, collection_name=col)

    print(f"eval set: {len(items)} questions across {len(runners)} KB(s)")
    print(f"configs: {[c.name for c in configs]}")
    print(f"top_k={top_k}\n")

    # 对每个 config 跑一遍所有 question
    results: dict[str, list[RetrievalHit]] = {c.name: [] for c in configs}
    by_qtype_results: dict[str, dict[str, list[RetrievalHit]]] = defaultdict(
        lambda: {c.name: [] for c in configs}
    )

    for cfg in configs:
        cfg.top_k = top_k
        t0 = time.perf_counter()
        for it in items:
            runner = runners.get(it["kb_id"])
            if not runner:
                continue
            try:
                retrieved = await runner.retrieve(it["question"], cfg)
            except Exception as e:
                print(f"  [{cfg.name}] q={it['qid']} failed: {e}", file=sys.stderr)
                retrieved = []
            hit = build_retrieval_hit(it, retrieved)
            results[cfg.name].append(hit)
            by_qtype_results[it.get("qtype", "?")][cfg.name].append(hit)
        dt = time.perf_counter() - t0
        print(f"  [{cfg.name}] {len(results[cfg.name])}/{len(items)} done in {dt:.1f}s")

    return {
        "results": results,
        "by_qtype": by_qtype_results,
        "items": items,
    }


def print_report(run_result: dict, configs: list[EvalConfig]) -> None:
    results: dict[str, list[RetrievalHit]] = run_result["results"]
    by_qtype: dict[str, dict[str, list[RetrievalHit]]] = run_result["by_qtype"]

    # 1. 总览表
    print("\n" + "=" * 100)
    print("OVERALL")
    print("=" * 100)
    baseline_metrics = aggregate(results.get(configs[0].name, []))
    baseline_name = configs[0].name
    for c in configs:
        m = aggregate(results.get(c.name, []))
        print(compare_to_baseline(c.name, m, baseline_metrics if c.name != baseline_name else {}))

    # 2. 按 qtype 明细
    qtypes = sorted(by_qtype.keys())
    if qtypes:
        print("\n" + "=" * 100)
        print("BY QUESTION TYPE")
        print("=" * 100)
        for qt in qtypes:
            print(f"\n[{qt}]")
            for c in configs:
                hits = by_qtype[qt].get(c.name, [])
                m = aggregate(hits)
                if m:
                    print(f"  {format_row(c.name, m)}")


def save_report(run_result: dict, configs: list[EvalConfig], out_path: Path) -> None:
    """把指标写成 markdown 报告。"""
    results = run_result["results"]
    by_qtype = run_result["by_qtype"]
    items = run_result["items"]

    lines = ["# Retrieval Evaluation Report", ""]
    lines.append(f"- eval set: `{len(items)}` questions")
    lines.append(f"- configs: {[c.name for c in configs]}")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| config | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr@5 | mrr@10 | hit_doc@5 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in configs:
        m = aggregate(results.get(c.name, []))
        if not m:
            continue
        row = [
            c.name,
            str(m["n"]),
            f"{m.get('recall@1', 0):.3f}",
            f"{m.get('recall@3', 0):.3f}",
            f"{m.get('recall@5', 0):.3f}",
            f"{m.get('recall@10', 0):.3f}",
            f"{m.get('mrr@5', 0):.3f}",
            f"{m.get('mrr@10', 0):.3f}",
            f"{m.get('hit_doc@5', 0):.3f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    qtypes = sorted(by_qtype.keys())
    if qtypes:
        lines.append("## By Question Type")
        lines.append("")
        for qt in qtypes:
            lines.append(f"### {qt}")
            lines.append("")
            lines.append("| config | n | recall@5 | recall@10 | mrr@10 |")
            lines.append("|---|---:|---:|---:|---:|")
            for c in configs:
                m = aggregate(by_qtype[qt].get(c.name, []))
                if not m:
                    continue
                row = [
                    c.name,
                    str(m["n"]),
                    f"{m.get('recall@5', 0):.3f}",
                    f"{m.get('recall@10', 0):.3f}",
                    f"{m.get('mrr@10', 0):.3f}",
                ]
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport saved to {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Run retrieval eval with multiple configs")
    p.add_argument("--set", type=Path, default=_BACKEND_DIR / "scripts" / "eval" / "data" / "eval_set.jsonl")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--quick", action="store_true", help="只跑 4 个核心 config")
    p.add_argument("--qtype", type=str, default=None, help="只看某个 qtype")
    p.add_argument("--report", type=Path, default=None, help="输出 markdown 报告路径")
    args = p.parse_args()

    if not args.set.exists():
        print(f"eval set not found: {args.set}", file=sys.stderr)
        print("先用 python -m scripts.eval.generate_set 生成", file=sys.stderr)
        sys.exit(1)

    configs = quick_configs() if args.quick else default_configs()
    report_path = args.report or args.set.with_suffix(".report.md")

    result = asyncio.run(run(args.set, configs, top_k=args.top_k, only_qtype=args.qtype))
    if result:
        print_report(result, configs)
        save_report(result, configs, report_path)


if __name__ == "__main__":
    main()