"""对话路由：SSE 流式问答 + 会话/消息管理。"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, status
from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sse_starlette.sse import EventSourceResponse

from app.core.config import get_settings
from app.core.deps import get_db
from app.models import Conversation, KnowledgeBase, Message

from app.schemas.chat import (
    ChatMessage as ChatMessageSchema,
    ChatRequest,
    ChatFinalEvent,
    ChatMeta,
    ConversationRead,
    ConversationUpdate,
    DocHitItem,
    MessageRead,
    SourceItem,
)
from app.services.agent import (
    SUMMARY_KEEP_RECENT,
    SUMMARY_TRIGGER_MESSAGES,
    AgentEvent,
    RAGAgent,
    build_agent,
)
from app.services.embedding_resolver import resolve_embedding

router = APIRouter(prefix="/chat", tags=["chat"])

DbSession = Annotated[AsyncSession, Depends(get_db)]


# ---------- helpers ----------


def _safe_json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


async def _maybe_update_summary(
    session: AsyncSession, conv_id: int | None
) -> None:
    """如果会话总消息数超过阈值，把早期 turns 压缩成 summary 写回 Conversation。

    失败不回滚主流程（持久化已成功），仅 log。
    """
    if not conv_id:
        return
    total = await session.scalar(
        select(func.count(Message.id)).where(Message.conversation_id == conv_id)
    )
    if not total or total <= SUMMARY_TRIGGER_MESSAGES:
        return
    conv = await session.get(Conversation, conv_id)
    if not conv:
        return
    # 早期 turns = 全部 - 最近保留窗口
    msgs = (
        await session.execute(
            select(Message)
            .where(Message.conversation_id == conv_id)
            .order_by(Message.id)
        )
    ).scalars().all()
    cutoff = max(0, len(msgs) - SUMMARY_KEEP_RECENT)
    early = [
        {"role": m.role, "content": m.content} for m in msgs[:cutoff]
    ]
    if not early:
        return
    try:
        new_summary = await RAGAgent.update_summary(conv.summary, early)
        if new_summary and new_summary != conv.summary:
            conv.summary = new_summary
            await session.commit()
            logger.info(
                "summary updated for conv={} ({} early turns -> {} chars)",
                conv_id,
                len(early),
                len(new_summary),
            )
    except Exception as e:
        logger.warning("update summary failed for conv={}: {}", conv_id, e)


async def _persist_turn(
    session: AsyncSession,
    conv_id: int | None,
    kb_id: int | None,
    user_content: str,
    assistant_content: str,
    intent: str | None,
    latency_ms: int,
    sources_json: str | None = None,
    doc_hits_json: str | None = None,
) -> tuple[int | None, int | None, int | None]:
    """写入一轮问答；返回 (conversation_id, user_msg_id, assistant_msg_id)。"""
    if conv_id is None:
        title = (user_content[:30] + "...") if len(user_content) > 30 else user_content
        conv = Conversation(knowledge_base_id=kb_id, title=title or "新会话")
        session.add(conv)
        await session.flush()
        conv_id = conv.id
    else:
        conv = await session.get(Conversation, conv_id)
        if not conv:
            raise HTTPException(status_code=404, detail="conversation not found")

    user_msg = Message(
        conversation_id=conv_id,
        role="user",
        content=user_content,
    )
    session.add(user_msg)
    await session.flush()
    assistant_msg = Message(
        conversation_id=conv_id,
        role="assistant",
        content=assistant_content,
        intent=intent,
        latency_ms=latency_ms,
        sources_json=sources_json,
        doc_hits_json=doc_hits_json,
    )
    session.add(assistant_msg)
    await session.flush()
    return conv_id, user_msg.id, assistant_msg.id


# ---------- main chat (SSE) ----------


@router.post("")
async def chat(payload: ChatRequest, session: DbSession):
    settings = get_settings()
    kb: KnowledgeBase | None = None
    if payload.knowledge_base_id is not None:
        kb = await session.get(KnowledgeBase, payload.knowledge_base_id)
        if not kb:
            raise HTTPException(status_code=404, detail="knowledge_base not found")

    async def event_iter():
        embedding_model, embedding_dim = resolve_embedding(kb)
        agent = build_agent(
            kb_id=kb.id if kb else None,
            collection_name=kb.collection_name if kb else None,
            enable_web=payload.enable_web and settings.enable_web_search,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
        )
        t0 = time.time()
        full_answer = ""
        intent: str | None = None
        sources_data: list[dict] = []
        doc_hits_data: list[dict] = []
        error_msg: str | None = None
        meta_refused: bool = False  # 信心闸门标记：从 agent meta 透传到 final event

        history = payload.history or []

        # 拉取已有 summary（如果是续接的会话），注入给 build_prompt 节点
        summary = ""
        if payload.conversation_id:
            existing_conv = await session.get(Conversation, payload.conversation_id)
            if existing_conv:
                summary = existing_conv.summary or ""

        try:
            async for ev in agent.astream(payload.question, history, summary=summary):
                if ev.type == "intent":
                    intent = ev.payload.get("intent")
                    yield {"event": "intent", "data": _safe_json(ev.payload)}
                elif ev.type == "sources":
                    sources_data = ev.payload.get("sources") or []
                    yield {"event": "sources", "data": _safe_json(ev.payload)}
                elif ev.type == "doc_hits":
                    # doc-level 命中：相关文档清单 + summary，前端单独展示。
                    doc_hits_data = ev.payload.get("doc_hits") or []
                    yield {
                        "event": "doc_hits",
                        "data": _safe_json({"doc_hits": doc_hits_data}),
                    }
                elif ev.type == "token":
                    full_answer += ev.payload.get("content", "")
                    yield {
                        "event": "token",
                        "data": _safe_json({"content": ev.payload.get("content", "")}),
                    }
                elif ev.type == "meta":
                    # update latency with full elapsed time
                    latency = int((time.time() - t0) * 1000)
                    # 信心闸门标记：从 agent meta payload 透传到 final event
                    meta_refused = bool(ev.payload.get("refused", False))
                    yield {
                        "event": "meta",
                        "data": _safe_json({"latency_ms": latency, **(ev.payload or {})}),
                    }
                elif ev.type == "error":
                    error_msg = ev.payload.get("message", "")
                    yield {"event": "error", "data": _safe_json(ev.payload)}
                elif ev.type == "done":
                    # 持久化
                    try:
                        conv_id, _, asst_id = await _persist_turn(
                            session=session,
                            conv_id=payload.conversation_id,
                            kb_id=kb.id if kb else None,
                            user_content=payload.question,
                            assistant_content=full_answer,
                            intent=intent,
                            latency_ms=int((time.time() - t0) * 1000),
                            sources_json=json.dumps(sources_data, ensure_ascii=False, default=str)
                            if sources_data
                            else None,
                            doc_hits_json=json.dumps(doc_hits_data, ensure_ascii=False, default=str)
                            if doc_hits_data
                            else None,
                        )
                        await session.commit()
                        # 长对话：消息数超过阈值时压缩早期 turns 成 summary
                        await _maybe_update_summary(session, conv_id)
                        final = ChatFinalEvent(
                            meta=ChatMeta(
                                intent=intent or "direct",
                                latency_ms=int((time.time() - t0) * 1000),
                                used_web=bool(sources_data and any(s.get("source_type") == "web" for s in sources_data)),
                                used_rag=bool(sources_data and any(s.get("source_type") in ("vector", "bm25") for s in sources_data)),
                                conversation_id=conv_id,
                                message_id=asst_id,
                                refused=bool(meta_refused),
                            ),
                            sources=[SourceItem(**s) for s in sources_data],
                            doc_hits=[DocHitItem(**h) for h in doc_hits_data],
                        )
                        yield {"event": "final", "data": final.model_dump_json()}
                    except Exception as e:
                        logger.exception("persist chat turn failed: {}", e)
                        yield {
                            "event": "error",
                            "data": _safe_json({"message": f"persist failed: {e}"}),
                        }
                    yield {"event": "end", "data": "{}"}
        except asyncio.CancelledError:
            logger.info("client disconnected")
            raise
        except Exception as e:
            logger.exception("chat failed")
            yield {"event": "error", "data": _safe_json({"message": str(e)})}
            yield {"event": "end", "data": "{}"}

    return EventSourceResponse(event_iter(), ping=15)


# ---------- non-streaming fallback ----------


@router.post("/sync")
async def chat_sync(payload: ChatRequest, session: DbSession) -> dict:
    """非流式收敛所有事件，返回一次性 JSON。便于调试/测试。"""
    settings = get_settings()
    kb: KnowledgeBase | None = None
    if payload.knowledge_base_id is not None:
        kb = await session.get(KnowledgeBase, payload.knowledge_base_id)
        if not kb:
            raise HTTPException(status_code=404, detail="knowledge_base not found")

    embedding_model, embedding_dim = resolve_embedding(kb)
    agent = build_agent(
        kb_id=kb.id if kb else None,
        collection_name=kb.collection_name if kb else None,
        enable_web=payload.enable_web and settings.enable_web_search,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
    )
    intent: str | None = None
    sources_data: list[dict] = []
    doc_hits_data: list[dict] = []
    full = ""
    err: str | None = None
    t0 = time.time()
    summary = ""
    if payload.conversation_id:
        existing_conv = await session.get(Conversation, payload.conversation_id)
        if existing_conv:
            summary = existing_conv.summary or ""
    async for ev in agent.astream(
        payload.question,
        [h.model_dump() for h in (payload.history or [])],
        summary=summary,
    ):
        if ev.type == "intent":
            intent = ev.payload.get("intent")
        elif ev.type == "sources":
            sources_data = ev.payload.get("sources") or []
        elif ev.type == "doc_hits":
            doc_hits_data = ev.payload.get("doc_hits") or []
        elif ev.type == "token":
            full += ev.payload.get("content", "")
        elif ev.type == "error":
            err = ev.payload.get("message")
    latency = int((time.time() - t0) * 1000)
    resp: dict = {
        "answer": full,
        "intent": intent,
        "sources": sources_data,
        "doc_hits": doc_hits_data,
        "latency_ms": latency,
    }
    if err:
        resp["error"] = err
    try:
        conv_id, _, asst_id = await _persist_turn(
            session=session,
            conv_id=payload.conversation_id,
            kb_id=kb.id if kb else None,
            user_content=payload.question,
            assistant_content=full,
            intent=intent,
            latency_ms=latency,
            sources_json=json.dumps(sources_data, ensure_ascii=False, default=str)
            if sources_data
            else None,
            doc_hits_json=json.dumps(doc_hits_data, ensure_ascii=False, default=str)
            if doc_hits_data
            else None,
        )
        await session.commit()
        await _maybe_update_summary(session, conv_id)
        resp["conversation_id"] = conv_id
        resp["message_id"] = asst_id
    except Exception as e:
        logger.warning("persist failed in sync mode: {}", e)
    return resp


# ---------- conversations ----------


@router.get("/conversations", response_model=list[ConversationRead])
async def list_conversations(
    session: DbSession,
    kb_id: int | None = None,
) -> list[ConversationRead]:
    stmt = select(Conversation).order_by(Conversation.updated_at.desc())
    if kb_id is not None:
        stmt = stmt.where(Conversation.knowledge_base_id == kb_id)
    res = await session.execute(stmt)
    return [ConversationRead.model_validate(c) for c in res.scalars().all()]


@router.get("/conversations/{conv_id}/messages", response_model=list[MessageRead])
async def list_messages(
    conv_id: Annotated[int, Path(ge=1)],
    session: DbSession,
) -> list[MessageRead]:
    res = await session.execute(
        select(Message).where(Message.conversation_id == conv_id).order_by(Message.id)
    )
    out: list[MessageRead] = []
    for m in res.scalars().all():
        item = MessageRead.model_validate(m)
        # sources_json 是持久化的 JSON 字符串（仅助手消息有）；解析回填给前端，
        # 让历史会话重新打开后引用 chip 仍可点击。
        if m.sources_json:
            try:
                raw = json.loads(m.sources_json)
                item.sources = [SourceItem(**s) for s in raw]
            except Exception as e:
                logger.warning("parse sources_json failed for msg={}: {}", m.id, e)
        # doc_hits_json 同样解析回填，让历史消息也能展示"相关文档"区。
        if m.doc_hits_json:
            try:
                raw = json.loads(m.doc_hits_json)
                item.doc_hits = [DocHitItem(**h) for h in raw]
            except Exception as e:
                logger.warning("parse doc_hits_json failed for msg={}: {}", m.id, e)
        out.append(item)
    return out


@router.delete(
    "/conversations/{conv_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def delete_conversation(
    conv_id: Annotated[int, Path(ge=1)],
    session: DbSession,
) -> None:
    conv = await session.scalar(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(Conversation.id == conv_id)
    )
    if not conv:
        raise HTTPException(status_code=404, detail="conversation not found")
    # messages 用 cascade="all, delete-orphan" 自动清理，这里只用显式删除保险一次
    for msg in list(conv.messages):
        await session.delete(msg)
    await session.delete(conv)
    await session.commit()


@router.patch("/conversations/{conv_id}", response_model=ConversationRead)
async def update_conversation(
    conv_id: Annotated[int, Path(ge=1)],
    payload: ConversationUpdate,
    session: DbSession,
) -> ConversationRead:
    conv = await session.get(Conversation, conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="conversation not found")
    if payload.title is not None:
        conv.title = payload.title.strip() or conv.title
    await session.commit()
    await session.refresh(conv)
    return ConversationRead.model_validate(conv)
