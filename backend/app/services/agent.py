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
    history: list[dict[str, str]]
    intent: Intent
    kb_id: int | None
    collection_name: str | None
    chunks: list[dict]
    web_results: list[dict]
    answer: str
    sources: list[dict]
    latency_ms: int
    used_web: bool
    used_rag: bool
    error: str | None
    # LangGraph 只保留 schema 里声明的 key；之前 _messages 不在 schema 中
    # 被丢弃，导致 stream_chat 收到空 messages -> LLM 报 2013。
    _messages: list[dict[str, str]]


@dataclass
class AgentEvent:
    type: Literal["intent", "sources", "token", "meta", "error", "done"]
    payload: dict[str, Any] = field(default_factory=dict)


# ===== Prompt builders =====

CLASSIFY_PROMPT = """You are a query router for a RAG system. Given a user question,
decide which path to take.

Options:
- "rag": question is about user-provided documents (e.g., "summarize my PDF",
  "what does the manual say about X", "in my notes").
- "web": question asks for fresh/external info (news, current events, lookup).
- "direct": casual chat, chitchat, math, coding, general knowledge not tied to
  any document.
- "hybrid": clearly needs both local docs AND web (e.g., compare my notes with
  latest info).

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
        g.add_node("retrieve", self._retrieve_node)
        g.add_node("web_search", self._web_search_node)
        g.add_node("build_prompt", self._build_prompt_node)
        g.add_node("generate", self._generate_node)
        g.add_node("direct_answer", self._direct_answer_node)

        g.add_edge(START, "classify")

        g.add_conditional_edges(
            "classify",
            self._route_after_classify,
            {
                "rag": "retrieve",
                "web": "web_search",
                "direct": "direct_answer",
                "hybrid": "retrieve",
            },
        )
        g.add_edge("retrieve", "web_search" if False else "build_prompt")
        # hybrid path: retrieve then web
        g.add_conditional_edges(
            "retrieve",
            self._route_after_retrieve,
            {"web_then_prompt": "web_search", "to_prompt": "build_prompt"},
        )
        g.add_edge("web_search", "build_prompt")
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
        return state.get("intent", "direct")

    def _route_after_retrieve(self, state: AgentState) -> str:
        if state.get("intent") == "hybrid" and self.enable_web:
            return "web_then_prompt"
        return "to_prompt"

    # ----- nodes -----

    async def _classify_node(self, state: AgentState) -> AgentState:
        question = state["question"]
        if not self.collection_name:
            # No KB selected -> direct
            state["intent"] = "direct"
            return state

        try:
            prompt = CLASSIFY_PROMPT.format(question=question)
            raw = await LLMFactory.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            intent = "rag"
            try:
                data = json.loads(self._extract_json(raw))
                # data 可能是 JSON 字符串（如 LLM 直接返回 "intent"）而非对象
                intent = data.get("intent", "rag") if isinstance(data, dict) else "rag"
                if intent not in {"rag", "web", "direct", "hybrid"}:
                    intent = "rag"
            except Exception:
                # Fallback heuristic
                q = question.lower()
                if any(k in q for k in ["搜索", "联网", "最新", "新闻", "search"]):
                    intent = "web"
                elif any(k in q for k in ["你好", "hi", "hello", "你是谁", "thanks"]):
                    intent = "direct"
                else:
                    intent = "rag"
            state["intent"] = intent
        except Exception as e:
            logger.warning("classify failed, fallback to direct: {}", e)
            state["intent"] = "direct"
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
        try:
            chunks: list[RetrievedChunk] = await retriever.retrieve(state["question"])
        except Exception as e:
            logger.warning("retrieve failed: {}", e)
            chunks = []
        state["chunks"] = [
            {
                "id": c.id,
                "text": c.text,
                "metadata": c.metadata,
                "score": c.score,
                "rerank_score": c.rerank_score,
            }
            for c in chunks
        ]
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

        if intent == "hybrid":
            local_ctx = self._format_chunks(chunks)
            web_ctx = self._format_web(web)
            prompt = ANSWER_HYBRID_PROMPT.format(
                local=local_ctx or "(no local results)",
                web=web_ctx or "(no web results)",
                question=question,
            )
        elif intent == "web":
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

        messages: list[dict[str, str]] = [{"role": "system", "content": prompt}]
        for h in state.get("history", [])[-6:]:
            messages.append(h)
        messages.append({"role": "user", "content": question})
        state["_messages"] = messages
        return state

    async def _generate_node(self, state: AgentState) -> AgentState:
        messages = state.get("_messages") or []
        chunks = state.get("chunks") or []
        web = state.get("web_results") or []
        state["sources"] = self._build_sources(chunks, web, kb_id=self.kb_id)
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
        self, question: str, history: list[ChatMessage] | None = None
    ) -> AsyncIterator[AgentEvent]:
        """Yield AgentEvents: intent -> sources -> token* -> meta -> done."""
        t0 = time.time()
        history_dicts = [{"role": m.role, "content": m.content} for m in (history or [])]
        init_state: AgentState = {
            "question": question,
            "history": history_dicts,
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

        messages = final_state.get("_messages") or []
        full_answer = ""
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
                "answer": full_answer,
            },
        )
        yield AgentEvent(type="done", payload={})

    # ----- helpers -----

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
            sources.append(
                {
                    "kb_id": kb_id,
                    "document": md.get("doc_title") or md.get("doc_filename") or "?",
                    "page": md.get("page"),
                    "chunk_id": c.get("id"),
                    "snippet": (c.get("text") or "")[:300],
                    "score": c.get("score"),
                    "rerank_score": c.get("rerank_score"),
                    "doc_id": md.get("doc_id"),
                    "source_type": "vector",
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
