"""应用配置：基于 pydantic-settings 加载 .env 与环境变量。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BeforeValidator, Field, computed_field
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _split_csv(v):
    """把 .env 中的逗号分隔字符串拆成 list（pydantic-settings 默认会做 JSON 解析，需 NoDecode 配合）。"""
    if isinstance(v, str):
        return [s.strip() for s in v.split(",") if s.strip()]
    return v


CsvList = Annotated[list[str], NoDecode, BeforeValidator(_split_csv)]


class Settings(BaseSettings):
    """全局配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===== Server =====
    host: str = "127.0.0.1"
    port: int = 8765
    log_level: str = "INFO"
    cors_origins: CsvList = Field(
        default_factory=lambda: ["http://localhost:5173", "tauri://localhost"]
    )

    # ===== LLM =====
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.3
    use_mock_llm: bool = False  # 沙箱/离线测试用，不调用外部 LLM

    # ===== Embedding =====
    embedding_provider: Literal["local", "openai", "mock"] = "local"
    openai_embedding_api_key: str = ""
    openai_embedding_base_url: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "BAAI/bge-base-zh-v1.5"
    local_embedding_device: str = "cpu"
    local_embedding_dim: int = 768

    # ===== Reranker =====
    rerank_provider: Literal["local", "none"] = "local"
    # 默认使用 bge-reranker-base（约 280MB）；v2-m3 也是 2.27G 太重，且质量优势在常规
    # 文档场景下与 base 几乎拉不开差距。
    local_rerank_model: str = "BAAI/bge-reranker-base"
    rerank_top_k: int = 20
    final_top_k: int = 5
    # 父子切片下，rerank 后是否做 parent_collapse（同一 parent 的多个 child 合并为 1 个 parent 块）。
    # 关掉会返回多个 child（重复 parent 内容），主要用于调试。
    parent_collapse: bool = True

    # ===== Web Search =====
    tavily_api_key: str = ""
    enable_web_search: bool = True
    max_web_results: int = 5

    # ===== Persistence =====
    data_dir: Path = Path("./data")
    upload_dir: Path = Path("./data/uploads")
    chroma_persist_dir: Path = Path("./data/chroma")
    sqlite_path: Path = Path("./data/metadata.db")
    bm25_index_dir: Path = Path("./data/bm25")

    # ===== Ingest =====
    chunk_size: int = 600
    chunk_overlap: int = 120
    max_upload_mb: int = 50

    # ===== HuggingFace =====
    # 留空走官方；填镜像站（如 https://hf-mirror.com）后下载速度大幅提升。
    # pydantic-settings 不会把未声明的字段写回 os.environ，所以这个字段同时被
    # warmup 主动 export 给 huggingface_hub / sentence_transformers。
    hf_endpoint: str = ""

    # ===== Local model directory =====
    # 本地模型副本的根目录；解析逻辑见 app.core.local_model。
    # 留空时使用工作目录下的 .model/。
    local_model_root: Path = Path("./.model")
    # 显式本地目录覆盖（可选）。设置后优先用该路径，跳过 model_id → 本地路径的推导。
    local_embedding_path: str = ""
    local_rerank_path: str = ""

    # ===== Agent =====
    enable_multi_query: bool = True
    enable_hyde: bool = False
    enable_relevance_check: bool = True
    # doc-level 预筛选：在 chunk-level 检索前先 BM25(title+filename+summary) 找出相关文档，
    # 给这些文档里的 chunk 在最终 rerank 分数上做 soft boost（×1.2）。
    # 关闭后 chunk-level 检索结果不变，只是少了 doc-level 优先级信号。
    enable_doc_index: bool = True
    doc_index_top_k: int = 3
    # KB→Web 自动兜底：KB 检索为空（或 relevance_check 全不相关）时，
    # 自动触发 web search；web 也空则拒绝回答（"暂未收录"）。
    # 关闭后直接拒绝（不消耗 web 配额，但用户拿不到 web 答案）。
    enable_kb_web_fallback: bool = True

    # ===== Image Description =====
    # 留空使用本地 VLM；填写则走 OpenAI 兼容接口
    image_description_api_key: str = ""
    image_description_base_url: str = ""
    image_description_model: str = "gpt-4o-mini"
    # 本地 VLM 模型（无 API Key 时生效）
    local_vlm_model: str = "Qwen/Qwen3-VL-2B-Instruct"
    local_vlm_path: str = ""
    local_vlm_device: str = "auto"
    max_images_per_doc: int = 50

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sqlite_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.sqlite_path}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def allowed_extensions(self) -> set[str]:
        return {
            ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
            ".md", ".txt", ".html", ".htm", ".csv",
        }

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.upload_dir, self.chroma_persist_dir, self.bm25_index_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
