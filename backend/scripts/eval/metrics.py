"""检索评估指标。

设计原则：
- gold 可以是 chunk_id（细粒度）或 doc_id（粗粒度）。
- retrieved 用每个 result 的 id（或 metadata.child_id 用于 parent-collapse 后的条目）。
- recall@k：top-k 里是否命中至少一个 gold（chunk 或 doc 都算）
- mrr@k：第一个命中的 gold 排名倒数
- hit_doc@k：top-k 里命中的 doc 是否在 gold doc 集合里（doc-level hit rate）

gold 用集合存放，避免 list 的 O(n) in。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalHit:
    """单次 (query, config) 的检索命中情况。"""

    retrieved_ids: list[str]  # 检索返回的 id 列表（按顺序）
    gold_chunk_ids: set[str]  # 期望命中的 chunk_id
    gold_doc_ids: set[int]  # 期望命中的 doc_id
    # 每条 retrieved 对应的 doc_id（用于 doc-level hit 计算）
    retrieved_doc_ids: list[int]


def _matched_chunk(retrieved_ids: list[str], gold: set[str]) -> int:
    """返回第一个命中的 gold 在 retrieved 列表里的位置（0-indexed）；未命中返回 -1。"""
    for i, rid in enumerate(retrieved_ids):
        if rid in gold:
            return i
    return -1


def _matched_doc(retrieved_doc_ids: list[int], gold: set[int]) -> int:
    for i, d in enumerate(retrieved_doc_ids):
        if d in gold:
            return i
    return -1


def recall_at_k(hit: RetrievalHit, k: int) -> float:
    """top-k 是否命中至少一个 gold chunk 或 doc。"""
    k = min(k, len(hit.retrieved_ids))
    return 1.0 if (
        _matched_chunk(hit.retrieved_ids[:k], hit.gold_chunk_ids) >= 0
        or _matched_doc(hit.retrieved_doc_ids[:k], hit.gold_doc_ids) >= 0
    ) else 0.0


def mrr_at_k(hit: RetrievalHit, k: int) -> float:
    """倒数排名：第一个命中 gold 的位置 k+1 的倒数。0 表示未命中。"""
    k = min(k, len(hit.retrieved_ids))
    pos = _matched_chunk(hit.retrieved_ids[:k], hit.gold_chunk_ids)
    if pos < 0:
        pos = _matched_doc(hit.retrieved_doc_ids[:k], hit.gold_doc_ids)
    return 0.0 if pos < 0 else 1.0 / (pos + 1)


def hit_doc_at_k(hit: RetrievalHit, k: int) -> float:
    """doc-level 命中率：top-k 中至少一个 doc_id 在 gold doc set 里。"""
    k = min(k, len(hit.retrieved_doc_ids))
    return 1.0 if any(d in hit.gold_doc_ids for d in hit.retrieved_doc_ids[:k]) else 0.0


def aggregate(hits: list[RetrievalHit], k_values: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    """聚合一组 hit 的指标。"""
    if not hits:
        return {}
    n = len(hits)
    out = {"n": n}
    for k in k_values:
        out[f"recall@{k}"] = sum(recall_at_k(h, k) for h in hits) / n
        out[f"mrr@{k}"] = sum(mrr_at_k(h, k) for h in hits) / n
        out[f"hit_doc@{k}"] = sum(hit_doc_at_k(h, k) for h in hits) / n
    return out


def format_row(name: str, metrics: dict) -> str:
    """格式化单行评估结果。"""
    if not metrics:
        return f"{name:<32}  (no data)"
    parts = [f"n={metrics['n']:<3}"]
    for k in ("recall@1", "recall@3", "recall@5", "recall@10"):
        if k in metrics:
            parts.append(f"{k}={metrics[k]:.3f}")
    for k in ("mrr@5", "mrr@10"):
        if k in metrics:
            parts.append(f"{k}={metrics[k]:.3f}")
    return f"{name:<32}  " + "  ".join(parts)


def compare_to_baseline(name: str, metrics: dict, baseline: dict) -> str:
    """与 baseline 对比，附 delta。"""
    row = format_row(name, metrics)
    if not metrics or not baseline:
        return row
    deltas = []
    for k in ("recall@5", "recall@10", "mrr@10"):
        if k in metrics and k in baseline and baseline[k] != 0:
            d = metrics[k] - baseline[k]
            sign = "+" if d >= 0 else ""
            deltas.append(f"{k}Δ={sign}{d:.3f}")
    if deltas:
        row += "  (" + "  ".join(deltas) + ")"
    return row