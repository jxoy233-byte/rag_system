"""LangGraph Agent: classify -> retrieve / web / direct -> generate.

Supports streaming SSE, hybrid retrieval, web search fallback.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from loguru import logger

from app.core.config import get_settings
from app.llm.factory import LLMFactory
from app.schemas.chat import ChatMessage, SourceItem
from app.services.retriever import HybridRetriever, RetrievedChunk
from app.websearch import WebSearchFactory, WebSearchResult


# ===== State =====

Intent = Literal["rag", "web", "direct", "hybrid"]


class AgentState(TypedDict, total=False):
    question: str
    # 经 multi-query / HyDE 改写后的查询列表；retrieve 节点会逐个查询。
    # 默认就是 [question]；扩展失败时回退原值。
    queries: list[str]
    history: list[dict[str, str]]
    # 更早对话的滚动摘要（来自 Conversation.summary）；为空表示无摘要。
    summary: str
    intent: Intent
    kb_id: int | None
    collection_name: str | None
    chunks: list[dict]
    web_results: list[dict]
    answer: str
    sources: list[dict]
    # doc-level 命中：BM25(title+filename+summary) 的 top-K 文档。
    # 用途：① 给 chunk 分数做 soft boost；② 流到前端展示"系统认为哪些文档相关"。
    doc_hits: list[dict]
    latency_ms: int
    used_web: bool
    used_rag: bool
    # 信心闸门标记：当 KB+web 都为空或全不相关时，置 True；
    # astream 据此跳过 LLM 调用、发固定"未找到"回复 + meta.refused=True。
    refused: bool
    error: str | None
    # LangGraph 只保留 schema 里声明的 key；之前 _messages 不在 schema 中
    # 被丢弃，导致 stream_chat 收到空 messages -> LLM 报 2013。
    _messages: list[dict[str, str]]


@dataclass
class AgentEvent:
    type: Literal["intent", "sources", "doc_hits", "token", "meta", "error", "done"]
    payload: dict[str, Any] = field(default_factory=dict)


# ===== Prompt builders =====

CLASSIFY_PROMPT = """You are a query router for a RAG system. Given a user question,
decide which path to take.

CURRENT CONTEXT:
- A knowledge base (KB) of user-uploaded documents is SELECTED right now.
- Prefer "rag" whenever the question could plausibly be answered from the KB.
  The user chose this KB on purpose — even general-knowledge questions should
  try the docs first (the LLM can then say "your docs don't cover this").

Options:
- "rag": question is about user-provided documents, OR could be answered from the KB.
  Default choice whenever a KB is selected. Pick this unless the question clearly
  requires fresh/external info the KB cannot have.
- "web": question asks for fresh/external info (news, current events, real-time
  lookup, prices, weather) that the KB definitely cannot contain.
- "direct": ONLY for casual chitchat / greetings that have nothing to do with the
  user's documents (e.g., "你好", "你是谁", "thanks"). Never pick this for math /
  coding / definitions when a KB is selected.
- "hybrid": clearly needs BOTH local docs AND web (e.g., "compare my notes with
  latest research").

Reply with JSON only: {{"intent": "<rag|web|direct|hybrid>", "reason": "<short>"}}

Question: {question}
"""


ANSWER_WITH_CONTEXT_PROMPT = """You are a helpful assistant answering the user based on the
provided context. Always cite sources using [n] markers tied to the reference list.

CONTEXT (numbered references):
{context}

INSTRUCTIONS:
- Answer in the user's language.
- Be concise and accurate.
- Cite sources inline like [1], [2] matching the reference numbers.
- If context is insufficient, say so honestly.
- Format with markdown when helpful.

USER QUESTION: {question}
"""


ANSWER_DIRECT_PROMPT = """You are a helpful assistant. Answer the user concisely and
accurately in their language. Use markdown when helpful.

QUESTION: {question}
"""


ANSWER_WEB_PROMPT = """You are a helpful assistant. Answer using the web search results
below. Cite sources with [n].

SEARCH RESULTS:
{context}

QUESTION: {question}
"""


ANSWER_HYBRID_PROMPT = """You are a helpful assistant. Combine the local document context
and the web search results to answer the question. Cite both [n]-local and [n]-web.

LOCAL CONTEXT:
{local}

WEB RESULTS:
{web}

QUESTION: {question}
"""


ANSWER_REFUSE_PROMPT = """You are a knowledge-base assistant. The system has already searched
both the user's document knowledge base and the public web, but found no relevant
information for the question below.

Respond in the user's language with a single short sentence like:
"抱歉，知识库中暂未收录与此问题相关的内容。"
or
"Sorry, I could not find relevant information in the knowledge base."

DO NOT answer from your own training knowledge. DO NOT make up facts.
DO NOT cite sources. DO NOT speculate.

QUESTION: {question}
"""


# 滚动摘要：阈值 / 保留窗口
# 总消息数（含本轮新写入的）超过 SUMMARY_TRIGGER_MESSAGES 时，把所有早期 turns 压缩成 summary
SUMMARY_TRIGGER_MESSAGES = 16
# 摘要触发后，仍作为原文 history 注入的最末几轮
SUMMARY_KEEP_RECENT = 6


SUMMARY_UPDATE_PROMPT = """You maintain a running summary of an ongoing conversation between a user
and an AI assistant over a knowledge base.

Existing summary (may be empty):
\"\"\"{existing}\"\"\"

New turns to integrate (oldest first):
{turns}

Produce an updated summary that compresses the conversation while preserving what matters:
- Main topics discussed and any conclusions reached
- Key facts, names, numbers, or decisions established
- Unresolved questions or open threads
- The user's apparent goal or context

Rules:
- Write in Chinese, in the same language as the conversation.
- Aim for ~200-500 Chinese characters as a guideline, but the priority is keeping
  key information; a slightly longer summary is fine if it preserves important details.
- Do NOT pad with filler phrases like "用户和助手讨论了"; be information-dense.
- Merge new information with the existing summary; do NOT repeat unchanged content.
- Output the summary text only, no preamble or labels.
"""


# Multi-query 改写 prompt：把用户问题换成 3 个不同措辞的等价版本。
# 用于提升召回率（用户问法与文档表述不一致时，多版本能多覆盖几条候选）。
QUERY_EXPANSION_PROMPT = """You generate search query rewrites for a RAG system.

Given the user question below, produce exactly 3 alternative phrasings that would
retrieve the same information. Rules:
- Each rewrite must be a complete, standalone question.
- Vary wording, synonyms, and angle; keep the SAME information need.
- Do NOT change the meaning, scope, or add new requirements.
- Reply with JSON only: {{"queries": ["q1", "q2", "q3"]}}

Question: {question}
"""


# HyDE prompt：让 LLM 先「假装回答」，用生成的答案做向量检索。
# 生成的答案会包含更接近文档表述的关键词/概念，对向量召回有帮助。
HYDE_PROMPT = """You are writing a hypothetical answer to a question. This will be used
as a semantic search query against a document collection, so write a short passage
(3-5 sentences) that a high-quality source document might contain.

Rules:
- Write as if you were a relevant section in a textbook or article.
- Include specific terms, names, numbers that would likely appear in the source.
- Answer the question directly; don't hedge or say "it depends".
- Do NOT include phrases like "the document says" or "according to".

Question: {question}
"""


# Self-RAG 风格的 relevance 过滤 prompt。
# 让 LLM 给每个候选 chunk 判 relevant=true/false，过滤掉无关的再喂给生成。
RELEVANCE_CHECK_PROMPT = """You are filtering retrieved document chunks for relevance.

Question: {question}

Chunks (numbered, same order as listed below):
{context}

For each chunk, decide whether it contains information that helps answer the question.
Reply as JSON only:
{{"verdicts": [{{"i": 1, "relevant": true, "reason": "<short>"}}, {{"i": 2, "relevant": false, "reason": "<short>"}}, ...]}}

Rules:
- Mark relevant=true only if the chunk has DIRECTLY useful information (facts, definitions,
  examples, data) that would appear in a good answer.
- Mark relevant=false for: off-topic chunks, wrong domain, vague tangents, or chunks that
  only mention keywords without answering the question.
- Use the same numbering/order as the chunks.
- If a chunk is borderline, prefer relevant=true (don't be too aggressive).
"""


# ===== Agent =====

class RAGAgent:
    """Stateless agent: takes question + kb_id, yields AgentEvents."""

    def __init__(
        self,
        kb_id: int | None,
        collection_name: str | None,
        enable_web: bool = True,
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        self.kb_id = kb_id
        self.collection_name = collection_name
        self.enable_web = enable_web
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self._graph = None

    def build_graph(self):
        """Build LangGraph StateGraph."""
        g = StateGraph(AgentState)

        g.add_node("classify", self._classify_node)
        g.add_node("query_expand", self._query_expand_node)
        g.add_node("retrieve", self._retrieve_node)
        g.add_node("relevance_check", self._relevance_check_node)
        g.add_node("web_search", self._web_search_node)
        g.add_node("build_prompt", self._build_prompt_node)
        g.add_node("generate", self._generate_node)
        g.add_node("direct_answer", self._direct_answer_node)

        g.add_edge(START, "classify")

        # rag / hybrid 先走 query_expand，再 retrieve；web / direct 跳过
        g.add_conditional_edges(
            "classify",
            self._route_after_classify,
            {
                "query_expand": "query_expand",
                "web": "web_search",
                "direct": "direct_answer",
            },
        )
        g.add_edge("query_expand", "retrieve")
        # retrieve 之后：
        # - rag：① KB 命中 → relevance_check；② KB 空 → web_fallback（如果开启 web+fallback）；
        #         ③ KB 命中但 fallback 关 → relevance_check（让 LLM 拒答）
        # - hybrid：先去 web 再回 relevance_check（KB+web 一起过滤）
        g.add_conditional_edges(
            "retrieve",
            self._route_after_retrieve,
            {
                "to_relevance": "relevance_check",
                "web_then_relevance": "web_search",
                "web_fallback": "web_search",
            },
        )
        # web_search 之后：
        # - hybrid（KB 也有结果）：回 relevance_check 过滤
        # - fallback（KB 空）：直接 build_prompt
        # - pure-web：直接 build_prompt
        g.add_conditional_edges(
            "web_search",
            self._route_after_web_search,
            {"to_relevance": "relevance_check", "to_prompt": "build_prompt"},
        )
        # relevance_check 后再次判断：清空 chunks 后是否还要 fallback 到 web
        g.add_conditional_edges(
            "relevance_check",
            self._route_after_relevance_check,
            {"to_prompt": "build_prompt", "web_fallback": "web_search"},
        )
        g.add_edge("build_prompt", "generate")
        g.add_edge("generate", END)
        g.add_edge("direct_answer", END)

        return g.compile()

    @property
    def graph(self):
        if self._graph is None:
            self._graph = self.build_graph()
        return self._graph

    # ----- routing helpers -----

    def _route_after_classify(self, state: AgentState) -> str:
        intent = state.get("intent", "direct")
        if intent in {"rag", "hybrid"}:
            return "query_expand"
        return intent  # "web" or "direct"

    def _route_after_retrieve(self, state: AgentState) -> str:
        intent = state.get("intent")
        chunks = state.get("chunks") or []
        # hybrid 一开始就走 KB+web 双路（KB+web 都有 chunks 时不需要 fallback 单独通道）
        if intent == "hybrid" and self.enable_web:
            return "web_then_relevance"
        # rag：KB 完全空 → 触发 KB→web 兜底（仅在配置允许且 web 可用时）
        if intent == "rag" and not chunks:
            s = get_settings()
            if s.enable_kb_web_fallback and self.enable_web:
                logger.debug("KB empty, falling back to web search")
                return "web_fallback"
        return "to_relevance"

    def _route_after_web_search(self, state: AgentState) -> str:
        intent = state.get("intent")
        chunks = state.get("chunks") or []
        # hybrid：KB+web 都用，需要 relevance_check 过滤无关 KB chunks
        if intent == "hybrid" and chunks:
            return "to_relevance"
        # 其他路径（rag-fallback / pure-web）：直接 build_prompt
        return "to_prompt"

    def _route_after_relevance_check(self, state: AgentState) -> str:
        """relevance_check 清空 chunks 后，决定要不要再 fallback 到 web。

        触发条件：intent 是 rag/hybrid（web 还没跑过）+ chunks 已空 + fallback 开启 + web 可用。
        hybrid 路径下 web 已经先跑过，不该再触发（否则死循环）。
        """
        intent = state.get("intent")
        chunks = state.get("chunks") or []
        if not chunks and intent == "rag" and self.enable_web:
            s = get_settings()
            if s.enable_kb_web_fallback:
                logger.debug("relevance_check cleared all chunks, falling back to web")
                return "web_fallback"
        return "to_prompt"

    # ----- nodes -----

    async def _classify_node(self, state: AgentState) -> AgentState:
        question = state["question"]
        if not self.collection_name:
            # No KB selected -> direct
            state["intent"] = "direct"
            return state

        # KB is selected — this is the default fallback whenever LLM classification
        # can't be parsed, or the whole classify call fails. The user explicitly
        # picked this KB, so we should attempt RAG by default.
        default_intent = "rag"

        try:
            prompt = CLASSIFY_PROMPT.format(question=question)
            raw = await LLMFactory.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            try:
                data = json.loads(self._extract_json(raw))
                # data 可能是 JSON 字符串（如 LLM 直接返回 "intent"）而非对象
                intent = data.get("intent", default_intent) if isinstance(data, dict) else default_intent
                if intent not in {"rag", "web", "direct", "hybrid"}:
                    intent = default_intent
            except Exception:
                # Fallback heuristic — when KB is selected, the safe default is "rag"
                # so we don't silently skip retrieval. Only override to "web" if the
                # question explicitly demands fresh/external info.
                q = question.lower()
                if any(k in q for k in ["搜索", "联网", "最新", "新闻", "今天", "今年", "search", "latest"]):
                    intent = "web"
                elif any(k in q for k in ["你好", "hi", "hello", "你是谁", "thanks", "谢谢"]):
                    intent = "direct"
                else:
                    intent = default_intent
            state["intent"] = intent
        except Exception as e:
            logger.warning("classify failed, fallback to rag: {}", e)
            state["intent"] = default_intent
        return state

    async def _query_expand_node(self, state: AgentState) -> AgentState:
        """Multi-query / HyDE 改写：扩展为多个查询再交给 retrieve 节点。

        - multi_query: 用 LLM 生成 3 个等价改写，并行 retrieve
        - hyde: 用 LLM 生成假设性回答作为额外查询（向量检索命中率更高）
        - 两者同时启用时，multi_query 优先（multi-query 已经包含原 question）
        - 失败/未启用：queries 退化为 [question]
        """
        s = get_settings()
        question = state["question"]
        queries: list[str] = [question]

        if not s.enable_multi_query and not s.enable_hyde:
            state["queries"] = queries
            return state

        try:
            if s.enable_multi_query:
                prompt = QUERY_EXPANSION_PROMPT.format(question=question)
                raw = await LLMFactory.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                data = json.loads(self._extract_json(raw))
                if isinstance(data, dict) and isinstance(data.get("queries"), list):
                    variants = [v for v in data["queries"] if isinstance(v, str) and v.strip()]
                    # 原 question + 3 个改写 = 4 个查询；超过 4 会拖慢 retrieve
                    queries = [question] + variants[:3]
                else:
                    logger.warning("multi-query parse returned non-list, fallback to single")
            elif s.enable_hyde:
                prompt = HYDE_PROMPT.format(question=question)
                hypothetical = await LLMFactory.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                if hypothetical.strip():
                    queries = [question, hypothetical.strip()]
        except Exception as e:
            logger.warning("query expansion failed, fallback to original question: {}", e)

        state["queries"] = queries
        logger.debug("query expansion: {} -> {} queries", question[:30], len(queries))
        return state

    async def _relevance_check_node(self, state: AgentState) -> AgentState:
        """Self-RAG 风格 relevance 过滤：让 LLM 给候选 chunk 打 relevant=true/false，
        过滤掉无关的再喂给生成节点。降低 LLM 拿到无关 chunk 编造答案的概率。
        """
        s = get_settings()
        if not s.enable_relevance_check:
            return state

        chunks = state.get("chunks") or []
        # 候选太少就不浪费 LLM 调用：3 个以下基本都要用
        if len(chunks) < 3:
            return state

        # 限制评审上限：超过 10 个的尾部候选置信度低，且 prompt 会爆长
        MAX_REVIEW = 10
        review = chunks[:MAX_REVIEW]
        ctx = "\n\n".join(
            f"[{i+1}] {(c.get('text') or '')[:500]}" for i, c in enumerate(review)
        )
        try:
            prompt = RELEVANCE_CHECK_PROMPT.format(
                question=state["question"], context=ctx
            )
            raw = await LLMFactory.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            data = json.loads(self._extract_json(raw))
            verdicts = data.get("verdicts", []) if isinstance(data, dict) else []
            keep_indices: set[int] = set()
            for v in verdicts:
                if not isinstance(v, dict) or not v.get("relevant"):
                    continue
                try:
                    idx = int(v.get("i", 0)) - 1
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < len(review):
                    keep_indices.add(idx)

            if not keep_indices:
                # LLM 评判：所有候选都无关 → 清空 chunks，让下游触发 KB→web 兜底或拒绝。
                # 旧行为是保留 top-3，但你的 eval 表明这会让无关 chunk 进 LLM 引发幻觉。
                logger.warning(
                    "relevance check: 0/{} relevant, clearing chunks", len(review),
                )
                state["chunks"] = []
                return state

            # 按原顺序保留 relevant 的 chunk（其余丢弃；尾部未审查的也丢弃）
            kept = [review[i] for i in sorted(keep_indices)]
            state["chunks"] = kept
            logger.debug(
                "relevance check: {}/{} kept", len(kept), len(review)
            )
        except Exception as e:
            logger.warning("relevance check failed, keep all chunks: {}", e)
        return state

    async def _retrieve_node(self, state: AgentState) -> AgentState:
        if not self.collection_name:
            state["chunks"] = []
            return state
        retriever = HybridRetriever(
            knowledge_base_id=self.kb_id or 0,
            collection_name=self.collection_name,
            rerank=True,
            embedding_model=self.embedding_model,
            embedding_dim=self.embedding_dim,
        )
        # 多查询并行检索（来自 query_expand 节点；未启用扩展时是单元素列表）
        # 注意：retriever.retrieve 是 async 函数，直接 gather 它的 coroutine 即可，
        # 不能套 asyncio.to_thread —— 那会给 async 函数返回未 await 的 coroutine 对象，
        # 后续遍历就会报 "'coroutine' object is not iterable"。
        queries = state.get("queries") or [state["question"]]
        is_multi = len(queries) > 1
        # Multi-query 时每路都不 rerank，最后对去重候选统一 rerank 一次。
        # CPU 上 reranker 处理 10×440 字中文 ≈ 0.5-1s；4 路全 rerank ≈ 4×1s = 4s，
        # 改成单次 rerank 能压到 ~1.5s 总耗时。
        per_query_rerank = not is_multi

        # doc-level 预筛选：在 chunk-level 检索前，先用 BM25(title+filename+summary)
        # 找出 top-K 相关文档；对这些文档的 chunk 在最终 rerank 得分上做 soft boost。
        # 失败 / KB 无 doc 都静默 fallback 到不 boost —— 不影响主流程。
        s = get_settings()
        boost_doc_ids: set[int] = set()
        boost_factor = 1.2
        # doc_hits 同时写入 state（供 generate 节点 emit 给前端）
        doc_hits_state: list[dict] = []
        if s.enable_doc_index and self.kb_id is not None:
            try:
                from app.services.doc_index import DocIndex

                idx = DocIndex.for_kb(self.kb_id)
                top_docs = await idx.query(state["question"], top_k=s.doc_index_top_k)
                boost_doc_ids = {d.doc_id for d in top_docs}
                doc_hits_state = [
                    {
                        "doc_id": d.doc_id,
                        "title": d.title,
                        "filename": d.filename,
                        "summary": d.summary,
                        "score": d.score,
                    }
                    for d in top_docs
                ]
                if top_docs:
                    logger.debug(
                        "doc-level top {}: {}",
                        len(top_docs),
                        [(d.doc_id, d.title[:20]) for d in top_docs],
                    )
            except Exception as e:  # pragma: no cover - cache best-effort
                logger.warning("DocIndex query failed: {}", e)

        try:
            results = await asyncio.gather(
                *(
                    retriever.retrieve(q, use_rerank=per_query_rerank)
                    for q in queries
                ),
                return_exceptions=True,
            )
        except Exception as e:
            logger.warning("retrieve gather failed: {}", e)
            results = []

        # 去重 + 保留每 chunk 的最高 rerank_score（不同查询对同一 chunk 的得分不直接可比）
        chunk_map: dict[str, RetrievedChunk] = {}
        for r in results:
            if isinstance(r, Exception):
                logger.warning("retrieve subquery failed: {}", r)
                continue
            for c in r:
                existing = chunk_map.get(c.id)
                if existing is None:
                    chunk_map[c.id] = c
                elif (c.rerank_score or 0.0) > (existing.rerank_score or 0.0):
                    chunk_map[c.id] = c

        # Multi-query：在原始 question 上做一次统一 rerank
        if is_multi and chunk_map:
            # 按当前排序（rerank_score 缺失则用 RRF score）取前 N 名 rerank，
            # 限制 rerank 输入大小，避免 4 路各拿 10 个去重后 40+ 候选全跑 rerank
            sorted_pre = sorted(
                chunk_map.values(),
                key=lambda c: (
                    c.rerank_score if c.rerank_score is not None else c.score,
                    c.score,
                ),
                reverse=True,
            )
            rerank_input_n = min(len(sorted_pre), max(s.rerank_top_k, s.final_top_k * 2))
            rerank_input = sorted_pre[:rerank_input_n]
            try:
                texts = [c.text for c in rerank_input]
                ranked = await asyncio.to_thread(
                    retriever.reranker.rerank,
                    state["question"],
                    texts,
                    len(rerank_input),
                )
                for idx, score in ranked:
                    if 0 <= idx < len(rerank_input):
                        rerank_input[idx].rerank_score = float(score)
                rerank_input.sort(
                    key=lambda c: (c.rerank_score or 0.0), reverse=True
                )
                chunks = rerank_input
            except Exception as e:
                logger.warning("multi-query rerank failed, fall back to RRF order: {}", e)
                chunks = sorted_pre
        else:
            chunks = sorted(
                chunk_map.values(),
                key=lambda c: (
                    c.rerank_score if c.rerank_score is not None else c.score,
                    c.score,
                ),
                reverse=True,
            )

        # doc-level soft boost：对来自 top 文档的 chunk 分数 × boost_factor。
        # 走 rerank_score（如有），没有则用 RRF score。乘前先备份 raw_score 用于 sources 展示。
        if boost_doc_ids:
            for c in chunks:
                md = c.metadata or {}
                base = c.rerank_score if c.rerank_score is not None else c.score
                c.raw_score = base  # 备份 boost 前的原始相关度，展示用
                if md.get("doc_id") in boost_doc_ids and base is not None:
                    c.rerank_score = float(base) * boost_factor
            # 重新排序
            chunks.sort(
                key=lambda c: (c.rerank_score if c.rerank_score is not None else c.score, c.score),
                reverse=True,
            )

        # 按 final_top_k 截断；multi-query 会产生更多候选，这里统一收口
        chunks = chunks[: s.final_top_k]

        state["chunks"] = [
            {
                "id": c.id,
                "text": c.text,
                "metadata": c.metadata,
                "score": c.score,
                "rerank_score": c.rerank_score,
                "raw_score": c.raw_score,
            }
            for c in chunks
        ]
        state["doc_hits"] = doc_hits_state
        state["used_rag"] = bool(chunks)
        return state

    async def _web_search_node(self, state: AgentState) -> AgentState:
        if not self.enable_web:
            state["web_results"] = []
            return state
        client = WebSearchFactory.get()
        if client is None:
            state["web_results"] = []
            return state
        s = get_settings()
        try:
            results: list[WebSearchResult] = await asyncio.to_thread(
                client.search, state["question"], s.max_web_results
            )
        except Exception as e:
            logger.warning("web search failed: {}", e)
            results = []
        state["web_results"] = [
            {"title": r.title, "url": r.url, "snippet": r.snippet, "content": r.content}
            for r in results
        ]
        state["used_web"] = bool(results)
        return state

    async def _build_prompt_node(self, state: AgentState) -> AgentState:
        chunks = state.get("chunks") or []
        web = state.get("web_results") or []
        intent = state.get("intent", "rag")
        question = state["question"]
        summary = state.get("summary") or ""

        # 信心闸门：KB+web 都空 → 直接拒绝，不进 LLM
        if intent in {"rag", "hybrid", "web"} and not chunks and not web:
            state["refused"] = True
            prompt = ANSWER_REFUSE_PROMPT.format(question=question)
            messages: list[dict[str, str]] = [{"role": "system", "content": prompt}]
            if summary:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "下面是本会话之前对话的摘要，用于保留上下文：\n"
                            f"{summary}"
                        ),
                    }
                )
            for h in state.get("history", [])[-SUMMARY_KEEP_RECENT:]:
                messages.append(h)
            messages.append({"role": "user", "content": question})
            state["_messages"] = messages
            logger.info("confidence gate: refused (no chunks, no web)")
            return state

        # rag intent 但 chunks 已空（KB→web 兜底已用 web_results 填上）→ 改用 web prompt
        # 这样 prompt 跟实际上下文类型一致，LLM 不会疑惑"为什么是 rag intent 但没本地结果"
        effective_intent = intent
        if intent == "rag" and not chunks and web:
            effective_intent = "web"

        if effective_intent == "hybrid":
            local_ctx = self._format_chunks(chunks)
            web_ctx = self._format_web(web)
            prompt = ANSWER_HYBRID_PROMPT.format(
                local=local_ctx or "(no local results)",
                web=web_ctx or "(no web results)",
                question=question,
            )
        elif effective_intent == "web":
            web_ctx = self._format_web(web)
            prompt = ANSWER_WEB_PROMPT.format(
                context=web_ctx or "(no web results)",
                question=question,
            )
        else:
            ctx = self._format_chunks(chunks)
            prompt = ANSWER_WITH_CONTEXT_PROMPT.format(
                context=ctx or "(no local results)",
                question=question,
            )

        messages = [{"role": "system", "content": prompt}]
        # 长对话：把 summary 作为 system message 注入；recent messages 原样保留。
        # summary 在 chat.py 里会被更新并存回 Conversation.summary。
        if summary:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "下面是本会话之前对话的摘要，用于保留上下文：\n"
                        f"{summary}"
                    ),
                }
            )
        for h in state.get("history", [])[-SUMMARY_KEEP_RECENT:]:
            messages.append(h)
        messages.append({"role": "user", "content": question})
        state["_messages"] = messages
        return state

    async def _generate_node(self, state: AgentState) -> AgentState:
        messages = state.get("_messages") or []
        chunks = state.get("chunks") or []
        web = state.get("web_results") or []
        # 拒绝回答时无源；sources 留空、前端不渲染引用。
        if state.get("refused"):
            state["sources"] = []
            state["doc_hits"] = []
        else:
            state["sources"] = self._build_sources(chunks, web, kb_id=self.kb_id)
            # doc-level 命中单独存一份（与 chunk-level sources 分离），
            # 前端 "相关文档" 区显示这些，chunk 引用仍走 sources。
            state["doc_hits"] = state.get("doc_hits") or []
        # actual streaming is handled outside (we need token-level events)
        state["answer"] = ""
        return state

    async def _direct_answer_node(self, state: AgentState) -> AgentState:
        prompt = ANSWER_DIRECT_PROMPT.format(question=state["question"])
        messages: list[dict[str, str]] = [{"role": "system", "content": prompt}]
        for h in state.get("history", [])[-6:]:
            messages.append(h)
        messages.append({"role": "user", "content": state["question"]})
        state["_messages"] = messages
        state["sources"] = []
        return state

    # ----- public streaming API -----

    async def astream(
        self,
        question: str,
        history: list[ChatMessage] | None = None,
        summary: str = "",
    ) -> AsyncIterator[AgentEvent]:
        """Yield AgentEvents: intent -> sources -> token* -> meta -> done."""
        t0 = time.time()
        history_dicts = [{"role": m.role, "content": m.content} for m in (history or [])]
        init_state: AgentState = {
            "question": question,
            "history": history_dicts,
            "summary": summary,
            "chunks": [],
            "web_results": [],
            "sources": [],
            "used_web": False,
            "used_rag": False,
        }
        try:
            final_state = await self.graph.ainvoke(init_state)
        except Exception as e:
            logger.exception("graph failed")
            yield AgentEvent(type="error", payload={"message": str(e)})
            yield AgentEvent(type="done", payload={})
            return

        intent = final_state.get("intent", "direct")
        yield AgentEvent(type="intent", payload={"intent": intent})

        sources_payload = final_state.get("sources", [])
        if sources_payload:
            yield AgentEvent(type="sources", payload={"sources": sources_payload})

        # doc-level 命中：相关文档清单 + summary，前端"相关文档"区展示。
        doc_hits_payload = final_state.get("doc_hits") or []
        if doc_hits_payload:
            yield AgentEvent(
                type="doc_hits", payload={"doc_hits": doc_hits_payload}
            )

        messages = final_state.get("_messages") or []
        refused = final_state.get("refused", False)
        full_answer = ""
        if refused:
            # 信心闸门触发：跳过 LLM 调用，直接发固定"未找到"回复。
            # 用 question 的语言不现实（前端不知道用户用什么语言问），
            # 简单给中英双语模板，前端按 user locale 选一个展示。
            full_answer = (
                "抱歉，知识库中暂未收录与此问题相关的内容。"
                " / Sorry, I could not find relevant information in the knowledge base."
            )
            yield AgentEvent(type="token", payload={"content": full_answer})
        else:
            try:
                async for token in LLMFactory.stream_chat(messages):
                    full_answer += token
                    yield AgentEvent(type="token", payload={"content": token})
            except Exception as e:
                logger.warning("stream_chat failed: {}", e)
                yield AgentEvent(type="error", payload={"message": str(e)})

        latency_ms = int((time.time() - t0) * 1000)
        yield AgentEvent(
            type="meta",
            payload={
                "intent": intent,
                "latency_ms": latency_ms,
                "used_web": final_state.get("used_web", False),
                "used_rag": final_state.get("used_rag", False),
                "refused": refused,
                "answer": full_answer,
            },
        )
        yield AgentEvent(type="done", payload={})

    # ----- helpers -----

    @staticmethod
    async def update_summary(existing: str, turns: list[dict[str, str]]) -> str:
        """压缩历史 turns 成 summary。

        turns 是 list[{role, content}]；existing 是当前 summary（可能为空）。
        失败时返回原 existing，保证不破坏已有摘要。
        """
        if not turns:
            return existing
        try:
            formatted = "\n".join(
                f"[{t.get('role', '?')}] {t.get('content', '')[:600]}"
                for t in turns
            )
            prompt = SUMMARY_UPDATE_PROMPT.format(existing=existing, turns=formatted)
            return await LLMFactory.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )
        except Exception as e:
            logger.warning("update_summary failed, keep existing: {}", e)
            return existing

    @staticmethod
    def _extract_json(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            # strip code fence
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            end = text.rfind("```")
            if end >= 0:
                text = text[:end]
        # find first { and last }
        s = text.find("{")
        e = text.rfind("}")
        if s >= 0 and e > s:
            return text[s : e + 1]
        return text

    @staticmethod
    def _format_chunks(chunks: list[dict]) -> str:
        if not chunks:
            return ""
        lines: list[str] = []
        for i, c in enumerate(chunks, 1):
            md = c.get("metadata") or {}
            tag = f"doc={md.get('doc_title', '?')}"
            if md.get("page"):
                tag += f", p={md['page']}"
            if md.get("section"):
                tag += f", sec={md['section']}"
            text = (c.get("text") or "").strip()
            lines.append(f"[{i}] ({tag})\n{text}")
        return "\n\n".join(lines)

    @staticmethod
    def _format_web(web: list[dict]) -> str:
        if not web:
            return ""
        lines: list[str] = []
        for i, w in enumerate(web, 1):
            text = (w.get("content") or w.get("snippet") or "").strip()
            lines.append(f"[{i}] {w.get('title', '')} - {w.get('url', '')}\n{text}")
        return "\n\n".join(lines)

    @staticmethod
    def _build_sources(chunks: list[dict], web: list[dict], kb_id: int | None = None) -> list[dict]:
        sources: list[dict] = []
        for c in chunks:
            md = c.get("metadata") or {}
            # 父子切片下：source_type 用 "parent"（collapsed）；否则 "vector" / "fused" / "bm25"。
            # chunk_id 用 metadata.child_id（真实可查的 child id），保证前端点引用 chip
            # 能拉到该 child 的详情；snippet 用 parent 完整文本的前 300 字。
            child_id = md.get("child_id")
            is_parent = bool(md.get("is_parent")) and bool(child_id)
            sources.append(
                {
                    "kb_id": kb_id,
                    "document": md.get("doc_title") or md.get("doc_filename") or "?",
                    "page": md.get("page"),
                    "chunk_id": child_id or c.get("id"),
                    "snippet": (c.get("text") or "")[:300],
                    "score": c.get("score"),
                    "rerank_score": (
                        c.get("raw_score")
                        if c.get("raw_score") is not None
                        else c.get("rerank_score")
                    ),
                    "doc_id": md.get("doc_id"),
                    "source_type": "parent" if is_parent else "vector",
                }
            )
        for w in web:
            sources.append(
                {
                    "kb_id": kb_id,
                    "document": w.get("title", ""),
                    "url": w.get("url", ""),
                    "snippet": (w.get("snippet") or "")[:300],
                    "source_type": "web",
                }
            )
        return sources


def build_agent(
    kb_id: int | None,
    collection_name: str | None,
    enable_web: bool = True,
    embedding_model: str | None = None,
    embedding_dim: int | None = None,
) -> RAGAgent:
    return RAGAgent(
        kb_id=kb_id,
        collection_name=collection_name,
        enable_web=enable_web,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
    )
