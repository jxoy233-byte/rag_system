"""LLM 工厂：统一 Chat Completions 接口（OpenAI 兼容 + mock）。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from app.core.config import get_settings


class _MockLLM:
    """离线/沙箱环境用的占位 LLM：根据用户问题返回简洁的占位回答。"""

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        return (
            f"[mock LLM] 已收到问题：{user[:80]}\n"
            "当前后端运行在离线/沙箱环境，未连接真实 LLM。"
            "请设置 USE_MOCK_LLM=false 并配置可用的 OPENAI_BASE_URL/OPENAI_API_KEY "
            "以启用真实模型。"
        )

    async def stream_chat(self, messages: list[dict[str, str]], **kwargs) -> AsyncIterator[str]:
        text = await self.chat(messages, **kwargs)
        for ch in text:
            await asyncio.sleep(0.005)
            yield ch


class LLMFactory:
    _client: AsyncOpenAI | None = None

    @classmethod
    def get_client(cls) -> AsyncOpenAI:
        if cls._client is None:
            s = get_settings()
            cls._client = AsyncOpenAI(api_key=s.openai_api_key, base_url=s.openai_base_url)
        return cls._client

    @classmethod
    def reset(cls) -> None:
        cls._client = None

    @classmethod
    def get_model(cls) -> str:
        return get_settings().llm_model

    @classmethod
    def get_temperature(cls) -> float:
        return get_settings().llm_temperature

    @classmethod
    async def chat(
        cls,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        model: str | None = None,
    ) -> str:
        if get_settings().use_mock_llm:
            return await _MockLLM().chat(messages, temperature=temperature, model=model)
        s = get_settings()
        client = cls.get_client()
        resp = await client.chat.completions.create(
            model=model or s.llm_model,
            temperature=temperature if temperature is not None else s.llm_temperature,
            messages=messages,
        )
        return resp.choices[0].message.content or ""

    @classmethod
    async def stream_chat(
        cls,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        if get_settings().use_mock_llm:
            async for ch in _MockLLM().stream_chat(messages, temperature=temperature, model=model):
                yield ch
            return
        s = get_settings()
        client = cls.get_client()
        stream = await client.chat.completions.create(
            model=model or s.llm_model,
            temperature=temperature if temperature is not None else s.llm_temperature,
            messages=messages,
            stream=True,
        )
        async for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content
            except (IndexError, AttributeError):
                delta = None
            if delta:
                yield delta
