"""API v1 router."""

from fastapi import APIRouter

from app.api.v1 import chat, documents, knowledge_bases, search

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(knowledge_bases.router)
api_router.include_router(documents.router)
api_router.include_router(chat.router)
api_router.include_router(search.router)
