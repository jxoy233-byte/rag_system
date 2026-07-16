"""联网搜索工厂：Tavily 优先，DDG 兜底。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.core.config import get_settings


@dataclass
class WebSearchResult:
     title: str
     url: str
     snippet: str
     content: str = ""


@runtime_checkable
class WebSearch(Protocol):
     def search(self, query: str, max_results: int = 5) -> list[WebSearchResult]: ...


class TavilySearch:
     def __init__(self, api_key: str) -> None:
         self._api_key = api_key
         self._client = None

     def _ensure(self):
         if self._client is None:
             from tavily import TavilyClient

             self._client = TavilyClient(api_key=self._api_key)
         return self._client

     def search(self, query: str, max_results: int = 5) -> list[WebSearchResult]:
         client = self._ensure()
         resp = client.search(query=query, max_results=max_results, search_depth="advanced")
         out: list[WebSearchResult] = []
         for r in resp.get("results", []):
             out.append(
                 WebSearchResult(
                     title=r.get("title", ""),
                     url=r.get("url", ""),
                     snippet=r.get("content", "")[:300],
                     content=r.get("content", ""),
                 )
             )
         return out


class DDGSearch:
     def search(self, query: str, max_results: int = 5) -> list[WebSearchResult]:
         try:
             from duckduckgo_search import DDGS
         except Exception:
             return []
         out: list[WebSearchResult] = []
         with DDGS() as ddgs:
             for r in ddgs.text(query, max_results=max_results):
                 out.append(
                     WebSearchResult(
                         title=r.get("title", ""),
                         url=r.get("href", ""),
                         snippet=r.get("body", "")[:300],
                         content=r.get("body", ""),
                     )
                 )
         return out


class WebSearchFactory:
     _cached: WebSearch | None = None

     @classmethod
     def get(cls) -> WebSearch | None:
         if cls._cached is not None:
             return cls._cached
         s = get_settings()
         if not s.enable_web_search:
             return None
         if s.tavily_api_key:
             try:
                 cls._cached = TavilySearch(s.tavily_api_key)
                 return cls._cached
             except Exception:
                 pass
         cls._cached = DDGSearch()
         return cls._cached
