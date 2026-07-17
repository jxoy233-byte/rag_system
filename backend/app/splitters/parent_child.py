"""父子两段切片（Parent-Child Chunking / Small-to-Big Retrieval）。

设计动机：
- 单层固定大小切片（默认 600 字）对中文偏小，易切断完整段落/示例；
  喂给 LLM 时还可能丢上下文。
- 父子两段：parent 较大（保留完整语义段，~800-1200 字），child 较小（精确定位，~200-300 字）。
  检索在 child 上做（召回更准、更快）；最终喂给 LLM 的是 parent 完整文本（上下文更全）。

切分策略（semantic chunking）：
- parent 边界：\\n\\n（段落）、Markdown 标题（# / ## / ###）、表格行（| ... |）连续出现、
  长 --- 分隔线。这些是天然的语义边界。
- child 边界：在 parent 内部按 \\n（短行/列表项）、中英文句末标点（。！？.!?）切。
- 表格 / 代码块 / 列表：尽量作为整体保留，不在中间切；如果超过 parent 上限，按行切但完整保留。
- overlap：parent 间不重叠（parent 是完整段，无需 overlap）；child 间小 overlap 保留上下文连续性。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.loaders.base import LoadedDocument
from app.splitters.factory import TextChunk


@dataclass
class ChildChunk(TextChunk):
    """子块：用于 embedding / BM25 检索的最小粒度。

    继承 TextChunk（text/page/section/metadata）以兼容现有 ingest 流程；
    额外带 parent_id / parent_text 供 retriever 在 rerank 后做 parent_collapse。
    """

    parent_id: str = ""
    parent_index: int = 0
    child_index: int = 0
    parent_text: str = ""


@dataclass
class ParentChunk:
    """父块：最终喂给 LLM 的完整语义段。"""

    parent_id: str
    parent_index: int
    text: str
    page: int | None = None
    section: str | None = None
    children: list[ChildChunk] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# 标题行（Markdown）：# / ## / ### / #### 开头。
# 中文 markdown 经常写 "#标题" 不带空格；放宽到 \s*。
_HEADING_RE = re.compile(r"^(#{1,6})\s*\S")
# 表格行：包含 | 的连续行
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
# 分隔线（Markdown table separator / horizontal rule）
_RULE_LINE_RE = re.compile(r"^\s*[-=*]{3,}\s*$")


def _group_into_parents(pages: list, parent_size: int = 1000) -> list[ParentChunk]:
    """把每页的文本按段落/标题切分成 parent 块。

    切分规则（从前往后扫，逐 block 判断起点）：
    - 遇到标题（# / ## / ### 等）：开新 parent，标题是它的首行
    - 遇到 --- / === 等分隔线：开新 parent
    - 遇到普通段落：归到当前 parent；当前 parent 没有则开新 parent
    - 同一 parent 内连续空行 / 多段落合并；遇到下一标题前一直累积
    - 单个 block 长度 > parent_size * 1.5 时按句末标点二次切（防爆）
    - 累积 cur_len 超过 parent_size 时也 flush（避免累积过大）
    """
    out: list[ParentChunk] = []
    parent_index = 0
    # 单 block 超过这个长度就强制按句子二次切（防爆）。
    hard_cap = int(parent_size * 1.5)

    for page in pages:
        text = page.text or ""
        if not text.strip():
            continue

        # 1. 粗分（按空行）
        raw_blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]

        # 2. 累积模式：cur 是当前正在累积的 parent 内容（str 拼接）
        cur: list[str] = []
        cur_len = 0

        def flush() -> None:
            nonlocal parent_index
            if not cur:
                return
            joined = "\n\n".join(cur).strip()
            if not joined:
                cur.clear()
                cur_len = 0
                return
            if len(joined) <= hard_cap:
                out.append(
                    ParentChunk(
                        parent_id=_parent_id(joined, page.page, page.section, parent_index),
                        parent_index=parent_index,
                        text=joined,
                        page=page.page,
                        section=page.section,
                        metadata=dict(page.metadata or {}),
                    )
                )
                parent_index += 1
            else:
                # 超长 parent 按句子二次切（每段不超过 parent_size）
                for sub in _split_long_block(joined, max_size=parent_size):
                    out.append(
                        ParentChunk(
                            parent_id=_parent_id(sub, page.page, page.section, parent_index),
                            parent_index=parent_index,
                            text=sub,
                            page=page.page,
                            section=page.section,
                            metadata=dict(page.metadata or {}),
                        )
                    )
                    parent_index += 1
            cur.clear()
            cur_len = 0

        for blk in raw_blocks:
            is_heading = bool(_HEADING_RE.match(blk.split("\n", 1)[0]))
            is_rule = bool(_RULE_LINE_RE.match(blk.split("\n", 1)[0]))

            if is_heading or is_rule:
                # 标题/分隔线：先落盘当前 parent，再开新 parent
                flush()
                cur.append(blk)
                cur_len = len(blk)
            else:
                # 普通段落：累积到 parent_size 阈值就 flush
                if cur_len + len(blk) > parent_size and cur:
                    flush()
                cur.append(blk)
                cur_len += len(blk)
        flush()

    return out


def _is_heading_or_rule(text: str) -> bool:
    """首行是否为标题或分隔线（用于"标题 + 内容合并"判断）。"""
    first_line = text.split("\n", 1)[0].strip()
    return bool(_HEADING_RE.match(first_line) or _RULE_LINE_RE.match(first_line))


def _split_long_block(text: str, max_size: int) -> list[str]:
    """把超长段落按句末标点切。优先 \n\n > 。！？ > .!? > ,; > \n > 空格。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_size,
        chunk_overlap=80,
        separators=["\n\n", "。", "！", "？", ". ", "! ", "? ", "\n", ",", ";", " ", ""],
        length_function=len,
    )
    return [s.strip() for s in splitter.split_text(text) if s.strip()]


def _parent_id(text: str, page: int | None, section: str | None, idx: int) -> str:
    """稳定 parent id：page + section + parent_index + 文本前 32 字符的 hash。

    入库阶段生成；后续 rerank / collapse 都靠它对齐 child → parent。
    """
    import hashlib

    h = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"p{idx}_{h}"


def _split_parent_to_children(parent: ParentChunk, child_size: int, child_overlap: int) -> list[ChildChunk]:
    """把 parent 拆成 child：按 \\n（短行/列表项）和句末标点切。"""
    text = parent.text
    if len(text) <= child_size:
        # parent 本身比 child 上限还短 → 单个 child 就是 parent 全文
        return [
            ChildChunk(
                text=text,
                page=parent.page,
                section=parent.section,
                metadata={
                    "parent_id": parent.parent_id,
                    "parent_index": parent.parent_index,
                    "child_index": 0,
                    "is_child": True,
                    **parent.metadata,
                },
                parent_id=parent.parent_id,
                parent_index=parent.parent_index,
                child_index=0,
                parent_text=text,
            )
        ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=child_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )
    raw = [s.strip() for s in splitter.split_text(text) if s.strip()]
    out: list[ChildChunk] = []
    for i, c in enumerate(raw):
        out.append(
            ChildChunk(
                text=c,
                page=parent.page,
                section=parent.section,
                metadata={
                    "parent_id": parent.parent_id,
                    "parent_index": parent.parent_index,
                    "child_index": i,
                    "is_child": True,
                    **parent.metadata,
                },
                parent_id=parent.parent_id,
                parent_index=parent.parent_index,
                child_index=i,
                parent_text=text,
            )
        )
    return out


class ParentChildSplitter:
    """父子两段切片器。

    输出两种视图：
    - `split()`：返回 list[ChildChunk]（每条都带 parent_id/parent_text），兼容现有 ingest 流程
      （下游只关心 .text / .page / .section / .metadata）。parent_text 存在 metadata 里，
      retriever 的 parent_collapse 阶段会用它替换 child.text。
    - `split_to_parents()`：返回 list[ParentChunk]，调试 / 高级用法。
    """

    def __init__(
        self,
        parent_size: int | None = None,
        child_size: int | None = None,
        child_overlap: int | None = None,
    ) -> None:
        from app.core.config import get_settings

        s = get_settings()
        # 默认：parent ~1000 字，child ~250 字，overlap ~50。
        # 之前是单一 chunk_size=600；现在 parent 比之前略大（保完整段），
        # child 比之前小一半（精确定位）。
        self._parent_size = parent_size if parent_size is not None else 1000
        self._child_size = child_size if child_size is not None else 250
        self._child_overlap = (
            child_overlap if child_overlap is not None else max(40, self._child_size // 5)
        )

    @property
    def parent_size(self) -> int:
        return self._parent_size

    @property
    def child_size(self) -> int:
        return self._child_size

    def split(self, doc: LoadedDocument) -> list[ChildChunk]:
        """对外接口：返回 list[ChildChunk]，每条带 parent 信息。"""
        parents = self._split_parents(doc)
        children: list[ChildChunk] = []
        for p in parents:
            p.children = _split_parent_to_children(p, self._child_size, self._child_overlap)
            children.extend(p.children)
        return children

    def split_to_parents(self, doc: LoadedDocument) -> list[ParentChunk]:
        """返回 ParentChunk 视图（含 children）。"""
        parents = self._split_parents(doc)
        for p in parents:
            p.children = _split_parent_to_children(p, self._child_size, self._child_overlap)
        return parents

    def _split_parents(self, doc: LoadedDocument) -> list[ParentChunk]:
        return _group_into_parents(doc.to_pages(), parent_size=self._parent_size)