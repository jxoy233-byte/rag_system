"""LLM 工厂：基于 LangChain 的 ChatOpenAI 统一接入。

设计：
- 实际推理走 `langchain_openai.ChatOpenAI`，享受 LangChain 的 streaming / LCEL / tool calling
  一致接口；同时因为 OpenAI 兼容协议在国内被广泛使用，只要换 base_url 就能切到
  DeepSeek / 通义千问 / Kimi / Grok / 自部署 vLLM 等任意 provider。
- 对外暴露 `LLMFactory.chat()` / `LLMFactory.stream_chat()`，接口签名与之前保持一致，
  caller（agent.py）一行不动。
- 流式：ChatOpenAI.astream(messages) → AsyncIterator[AIMessageChunk]，逐 chunk yield .content。
- 非流式：ChatOpenAI.ainvoke(messages) → AIMessage，返回 .content。
- mock 模式：USE_MOCK_LLM=true 时走内置占位实现，不调用任何外部 API。

注意：
- langchain-openai 1.x 之后 ChatOpenAI 的 base_url 参数名是 `base_url`（不是 `openai_api_base`）
- ChatOpenAI 接受 list[dict[str, str]] 形式的 messages（role/content），会自动转 BaseMessage
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from langchain_openai import ChatOpenAI

from app.core.config import Settings, get_settings


class _MockLLM:
    """离线/沙箱环境用的占位 LLM：根据用户问题返回简洁的占位回答。"""

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        return (
            f"[mock LLM] 已收到问题：{user[:80]}\n"
            "当前后端运行在离线/沙箱环境，未连接真实 LLM。"
            "请设置 USE_MOCK_LLM=false 并配置可用的 OPENAI_BASE_URL/OPENAI_API_KEY "
            "以启用真实模型。"
        )

    async def stream_chat(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[str]:
        text = await self.chat(messages, **kwargs)
        for ch in text:
            await asyncio.sleep(0.005)
            yield ch


class LLMFactory:
    """统一 LLM 接入（ChatOpenAI + mock）。"""

    # 模型实例缓存：按 (model, temperature, base_url, api_key) 缓存，
    # 避免每次调用都重建。LangChain ChatOpenAI 自身不维护连接池，
    # 每次新建成本不高但有 race 风险，这里串行访问。
    _instances: dict[str, ChatOpenAI] = {}

    @classmethod
    def _cache_key(cls, s: Settings, model: str, temperature: float) -> str:
        return f"{model}|{temperature}|{s.openai_base_url}|{s.openai_api_key[:8]}"

    @classmethod
    def get_chat_model(
        cls,
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> ChatOpenAI:
        s = get_settings()
        m = model or s.llm_model
        t = temperature if temperature is not None else s.llm_temperature
        key = cls._cache_key(s, m, t)
        if key in cls._instances:
            return cls._instances[key]
        chat = ChatOpenAI(
            model=m,
            temperature=t,
            api_key=s.openai_api_key,
            base_url=s.openai_base_url,
            timeout=60.0,
            max_retries=2,
            streaming=True,  # 让 astream() 真正走流式路径而非缓存完整响应
        )
        cls._instances[key] = chat
        return chat

    @classmethod
    def reset(cls) -> None:
        cls._instances.clear()

    @classmethod
    async def chat(
        cls,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        model: str | None = None,
    ) -> str:
        """非流式调用：返回完整文本。"""
        if get_settings().use_mock_llm:
            return await _MockLLM().chat(messages, temperature=temperature, model=model)
        chat = cls.get_chat_model(model=model, temperature=temperature)
        result = await chat.ainvoke(messages)
        return result.content if isinstance(result.content, str) else str(result.content)

    @classmethod
    async def stream_chat(
        cls,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        """流式调用：逐 token yield 增量文本。"""
        if get_settings().use_mock_llm:
            async for ch in _MockLLM().stream_chat(
                messages, temperature=temperature, model=model
            ):
                yield ch
            return
        chat = cls.get_chat_model(model=model, temperature=temperature)
        async for chunk in chat.astream(messages):
            content = chunk.content
            if not content:
                continue
            # content 在某些 provider（如 tool calling）下可能是 list[dict]；这里只取纯文本路径
            if isinstance(content, str):
                yield content
            else:
                # 兜底：把 list 里的文本块拼起来
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        yield part["text"]
                    elif isinstance(part, str):
                        yield part