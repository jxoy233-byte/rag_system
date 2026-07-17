"""Agent 信心闸门 + KB→Web 兜底测试。

设计：用 mock 替换外部依赖（HybridRetriever / WebSearchFactory / LLMFactory），
直接驱动 RAGAgent.astream()，断言 yield 的事件序列。

覆盖：
1. rag + KB 空 + web 有结果 → 走 web → ANSWER_WEB_PROMPT
2. rag + KB 空 + web 空 → 拒绝（refused=True）+ meta.refused=True
3. rag + KB 有 chunks 但 relevance 全不相关 → 清空 → 触发 web fallback
4. enable_kb_web_fallback=False 时 KB 空 → 直接拒绝，不走 web
5. direct intent → 不进 retrieve / 不进 fallback
6. hybrid intent → KB+web 都跑，走 hybrid prompt（不变化）
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.agent import RAGAgent
from app.services.retriever import RetrievedChunk


# ===== helpers =====


def _make_chunk(chunk_id: str, doc_id: int = 1, text: str = "text") -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        text=text,
        metadata={"doc_id": doc_id, "chunk_index": 0},
        score=0.5,
        source="vector",
    )


async def _collect_events(agent: RAGAgent, question: str = "test q") -> list[Any]:
    out = []
    async for ev in agent.astream(question):
        out.append(ev)
    return out


def _events_by_type(events: list[Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for ev in events:
        out.setdefault(ev.type, []).append(ev)
    return out


# ===== fixture: 替换 LLM / retriever / web / reranker =====


@pytest.fixture
def mock_llm():
    """用 monkeypatch 替换 LLMFactory.chat / stream_chat。

    默认行为：
    - chat 返回「intent=rag + reason=stub」
    - relevance check 全 False（让所有候选都不相关）
    - stream_chat 直接 yield 一个 token
    """

    def _factory(responses: list[str] | None = None):
        # responses: 顺序匹配的 chat 返回值列表
        chat_responses = list(responses or [])

        async def fake_chat(messages, **kwargs):
            if chat_responses:
                return chat_responses.pop(0)
            # 默认：classify → rag
            return json.dumps({"intent": "rag", "reason": "stub"})

        async def fake_stream(messages, **kwargs):
            yield "stub answer"

        return fake_chat, fake_stream, chat_responses

    return _factory


# ===== Test 1: rag + KB 空 + web 有结果 → fallback 到 web =====


@pytest.mark.asyncio
async def test_rag_fallback_to_web_when_kb_empty():
    """KB 检索为空 → 路由到 web_search → 用 ANSWER_WEB_PROMPT。"""
    agent = RAGAgent(kb_id=1, collection_name="kb1", enable_web=True)

    # mock retriever 返回空
    async def fake_retrieve(self, q, **kw):
        return []

    # mock web search 返回 2 条结果（同步函数：_web_search_node 用 asyncio.to_thread 包）
    def fake_web(question, max_results):
        return [
            MagicMock(title="Web Title 1", url="http://a.com", snippet="snip1", content="content1"),
            MagicMock(title="Web Title 2", url="http://b.com", snippet="snip2", content="content2"),
        ]

    chat_calls = []

    async def fake_chat(messages, **kwargs):
        chat_calls.append(messages[0]["content"][:50])
        # 第一次：classify → rag；之后不会再调（relevance_check 不会跑因为 KB 空）
        return json.dumps({"intent": "rag", "reason": "stub"})

    async def fake_stream(messages, **kwargs):
        yield "ok"

    with patch("app.services.agent.HybridRetriever.retrieve", fake_retrieve), \
         patch("app.services.agent.WebSearchFactory.get") as ws_factory, \
         patch("app.services.agent.LLMFactory.chat", fake_chat), \
         patch("app.services.agent.LLMFactory.stream_chat", fake_stream):
        ws_factory.return_value.search = fake_web
        events = await _collect_events(agent)

    by_type = _events_by_type(events)
    # 应有 meta，且 refused=False（因为 web 有结果）
    meta = by_type["meta"][0]
    assert meta.payload["refused"] is False
    assert meta.payload["used_web"] is True
    assert meta.payload["used_rag"] is False
    # 应有 sources（web 类型）
    if "sources" in by_type:
        for s in by_type["sources"][0].payload["sources"]:
            assert s.get("source_type") == "web"


# ===== Test 2: rag + KB 空 + web 空 → 拒绝 =====


@pytest.mark.asyncio
async def test_rag_refuses_when_kb_and_web_empty():
    agent = RAGAgent(kb_id=1, collection_name="kb1", enable_web=True)

    async def fake_retrieve(self, q, **kw):
        return []

    def fake_web(question, max_results):
        return []

    async def fake_chat(messages, **kwargs):
        return json.dumps({"intent": "rag", "reason": "stub"})

    async def fake_stream(messages, **kwargs):
        yield "should not reach"
        pytest.fail("stream_chat should be skipped when refused=True")

    with patch("app.services.agent.HybridRetriever.retrieve", fake_retrieve), \
         patch("app.services.agent.WebSearchFactory.get") as ws_factory, \
         patch("app.services.agent.LLMFactory.chat", fake_chat), \
         patch("app.services.agent.LLMFactory.stream_chat", fake_stream):
        ws_factory.return_value.search = fake_web
        events = await _collect_events(agent, question="kb 里没有的")

    by_type = _events_by_type(events)
    meta = by_type["meta"][0]
    # 闸门触发：meta.refused=True
    assert meta.payload["refused"] is True
    # 不应有 sources / doc_hits 事件
    assert "sources" not in by_type
    assert "doc_hits" not in by_type
    # 应该有 token 事件（固定"未找到"回复）
    assert len(by_type["token"]) >= 1
    full = "".join(t.payload["content"] for t in by_type["token"])
    assert "暂未收录" in full or "knowledge base" in full.lower()


# ===== Test 3: rag + KB 有 chunks 但 relevance 全 False → 走 web fallback =====


@pytest.mark.asyncio
async def test_relevance_check_clears_irrelevant_then_fallbacks_to_web():
    agent = RAGAgent(kb_id=1, collection_name="kb1", enable_web=True)

    # 3 个 chunk，relevance check 全判 False → state.chunks=[] → 触发 fallback
    async def fake_retrieve(self, q, **kw):
        return [
            _make_chunk("c1", text="noise 1"),
            _make_chunk("c2", text="noise 2"),
            _make_chunk("c3", text="noise 3"),
        ]

    def fake_web(question, max_results):
        return [
            MagicMock(title="Fallback Hit", url="http://x.com", snippet="ok", content="ok"),
        ]

    chat_responses = [
        json.dumps({"intent": "rag", "reason": "stub"}),
        # relevance check: 全 False
        json.dumps({"verdicts": [
            {"i": 1, "relevant": False},
            {"i": 2, "relevant": False},
            {"i": 3, "relevant": False},
        ]}),
    ]

    async def fake_chat(messages, **kwargs):
        if not chat_responses:
            return json.dumps({"intent": "rag", "reason": "stub"})
        return chat_responses.pop(0)

    async def fake_stream(messages, **kwargs):
        yield "answer from web fallback"

    with patch("app.services.agent.HybridRetriever.retrieve", fake_retrieve), \
         patch("app.services.agent.WebSearchFactory.get") as ws_factory, \
         patch("app.services.agent.LLMFactory.chat", fake_chat), \
         patch("app.services.agent.LLMFactory.stream_chat", fake_stream):
        ws_factory.ret = fake_web
        ws_factory.return_value.search = fake_web
        events = await _collect_events(agent)

    by_type = _events_by_type(events)
    meta = by_type["meta"][0]
    # 闸门不该触发（web 有结果）
    assert meta.payload["refused"] is False
    assert meta.payload["used_web"] is True
    # 确认 chat 被调了 ≥ 2 次（classify + relevance）
    assert len(chat_responses) == 0  # 两个都被消费


# ===== Test 4: enable_kb_web_fallback=False → KB 空直接拒绝 =====


@pytest.mark.asyncio
async def test_disable_fallback_keeps_empty_kb_no_web():
    """关掉 fallback 配置时，KB 空应该直接拒绝，不调用 web search。"""
    from app.core.config import get_settings

    # 临时改 config 然后 reload
    s = get_settings()
    original = s.enable_kb_web_fallback
    try:
        # 通过环境变量修改（settings 在初始化时已加载，需要 patch 字段）
        s.enable_kb_web_fallback = False

        agent = RAGAgent(kb_id=1, collection_name="kb1", enable_web=True)

        async def fake_retrieve(q, **kw):
            return []

        def fake_web(question, max_results):
            pytest.fail("web search should NOT be called when fallback disabled")

        async def fake_chat(messages, **kwargs):
            return json.dumps({"intent": "rag", "reason": "stub"})

        async def fake_stream(messages, **kwargs):
            yield "should not run"
            pytest.fail("stream_chat should be skipped when refused=True")

        with patch("app.services.agent.HybridRetriever.retrieve", fake_retrieve), \
             patch("app.services.agent.WebSearchFactory.get") as ws_factory, \
             patch("app.services.agent.LLMFactory.chat", fake_chat), \
             patch("app.services.agent.LLMFactory.stream_chat", fake_stream):
            ws_factory.return_value.search = fake_web
            events = await _collect_events(agent)

        by_type = _events_by_type(events)
        meta = by_type["meta"][0]
        assert meta.payload["refused"] is True
        assert meta.payload["used_web"] is False
    finally:
        s.enable_kb_web_fallback = original


# ===== Test 5: direct intent 跳过 retrieve / fallback =====


@pytest.mark.asyncio
async def test_direct_intent_skips_retrieval_and_fallback():
    """无 KB（collection_name=None）→ classify → direct → direct_answer_node。"""
    agent = RAGAgent(kb_id=None, collection_name=None, enable_web=True)

    async def fake_chat(messages, **kwargs):
        pytest.fail("classify LLM should NOT be called when collection_name is None")

    async def fake_stream(messages, **kwargs):
        yield "direct hello"

    with patch("app.services.agent.LLMFactory.chat", fake_chat), \
         patch("app.services.agent.LLMFactory.stream_chat", fake_stream):
        events = await _collect_events(agent, question="你好")

    by_type = _events_by_type(events)
    meta = by_type["meta"][0]
    # direct 不会被闸门拒绝（没走 retrieve）
    assert meta.payload["refused"] is False
    assert meta.payload["intent"] == "direct"


# ===== Test 6: hybrid intent → KB+web 都跑，行为不变 =====


@pytest.mark.asyncio
async def test_hybrid_intent_web_runs_alongside_kb():
    agent = RAGAgent(kb_id=1, collection_name="kb1", enable_web=True)

    async def fake_retrieve(self, q, **kw):
        return [_make_chunk("c1", text="kb hit")]

    def fake_web(question, max_results):
        return [MagicMock(title="W", url="http://w", snippet="ws", content="wc")]

    chat_responses = [
        json.dumps({"intent": "hybrid", "reason": "stub"}),
        # relevance: 1 个 chunk 标 True
        json.dumps({"verdicts": [{"i": 1, "relevant": True}]}),
    ]

    async def fake_chat(messages, **kwargs):
        if not chat_responses:
            return json.dumps({"intent": "hybrid"})
        return chat_responses.pop(0)

    async def fake_stream(messages, **kwargs):
        yield "hybrid answer"

    with patch("app.services.agent.HybridRetriever.retrieve", fake_retrieve), \
         patch("app.services.agent.WebSearchFactory.get") as ws_factory, \
         patch("app.services.agent.LLMFactory.chat", fake_chat), \
         patch("app.services.agent.LLMFactory.stream_chat", fake_stream):
        ws_factory.return_value.search = fake_web
        events = await _collect_events(agent)

    by_type = _events_by_type(events)
    meta = by_type["meta"][0]
    # hybrid 不被闸门拒绝（KB 和 web 都有）
    assert meta.payload["refused"] is False
    assert meta.payload["used_web"] is True
    assert meta.payload["used_rag"] is True


# ===== Test 7: 后端配置默认 enable_kb_web_fallback=True =====


def test_default_config_enables_fallback():
    from app.core.config import get_settings
    s = get_settings()
    assert s.enable_kb_web_fallback is True