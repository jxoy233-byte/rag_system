# 通用文档问答 RAG 系统

基于 Vue 3 + Tauri 桌面浮窗的通用 RAG 系统，支持：

- **本地混合检索**（向量 + BM25 + RRF + BGE-Reranker）
- **联网搜索**（Tavily 优先，DuckDuckGo 兜底）
- **基础对话**（无外部知识）
- **图片理解**（VLM 描述后并入文本切片）
- 多轮会话、引用溯源、SSE 流式输出

> 设计与阶段划分见 [`agent.md`](./agent.md)。

## 目录

- 一、技术栈与结构
- 二、快速开始
- 三、API 约定
- 四、SSE 事件协议
- 五、环境变量
- 六、开发与调试
- 七、已知风险与回退

## 一、技术栈与结构

| 层 | 技术 |
| --- | --- |
| 后端 | Python 3.11+ · FastAPI · LangChain / LangGraph · SQLAlchemy 2 (async) · Chroma · BM25 (rank-bm25 + jieba) · FlagEmbedding (BGE-M3 / BGE-Reranker-Base) · Tavily / DuckDuckGo |
| 前端 | Vue 3.5 + TS · Vite 6 · Pinia 2 · Vue Router 4 (hash) · Naive UI（按需自动注册） · lucide-vue-next · Markdown-it · ofetch |
| 桌面壳 | Tauri 2.x（全局快捷键、悬浮窗口、托盘、文件拖入） |
| 持久化 | SQLite 元数据 + Chroma 向量库 + Pickle BM25 索引 + 本地上传文件 |
| 图片理解 | 本地 Qwen3-VL-2B-Instruct（默认） / OpenAI 兼容 VLM（可选） |

```
rag_system/
|-- backend/                 # FastAPI 后端
|   |-- app/
|   |   |-- api/v1/          # knowledge_bases / documents / search / chat (SSE)
|   |   |-- core/            # config / logging / db / deps / local_model
|   |   |-- loaders/         # PDF / DOCX / PPTX / MD / TXT / HTML / CSV / XLSX
|   |   |-- splitters/       # Recursive splitter（langchain）
|   |   |-- embeddings/      # local (BGE-M3) / openai / mock 工厂
|   |   |-- vectorstore/     # ChromaStore 封装
|   |   |-- rerankers/       # BGE-Reranker (FlagEmbedding)
|   |   |-- websearch/       # Tavily / DDG 工厂
|   |   |-- llm/             # OpenAI 兼容 Chat/Stream
|   |   |-- services/        # ingest / retriever / agent (LangGraph) / bm25 / image_describer
|   |   |-- models/          # SQLAlchemy ORM
|   |   `-- schemas/         # Pydantic DTO
|   |-- tests/               # pytest-asyncio
|   `-- pyproject.toml
|-- frontend/
|   |-- src/
|   |   |-- api/client.ts    # ofetch 封装 + SSE 解析（带 CRLF→LF 归一化）
|   |   |-- components/      # FloatingBall / ChatPanel / MessageBubble / KnowledgePicker / ConversationList
|   |   |-- views/           # KBView / DocsView / SearchView / SettingsView
|   |   |-- stores/          # Pinia: settings / knowledgeBase / chat
|   |   |-- styles/main.css  # 主题 / Markdown / 滚动条
|   |   `-- router.ts        # hash 模式；App.vue 按 route.name 渲染主面板
|   |-- vite.config.ts
|   |-- tsconfig.json
|   `-- index.html
|-- data/                    # 运行时持久化（gitignore）
|   |-- chroma/
|   |-- bm25/
|   |-- uploads/<kb_slug>/
|   `-- metadata.db
|-- agent.md
`-- README.md
```

### 关键模块职责

| 模块 | 路径 | 职责 |
| --- | --- | --- |
| `IngestService` | `services/ingest.py` | 文档入库主流程：解析 → 图片描述 → 切片 → 向量化 → BM25 + Chroma 写入 |
| `HybridRetriever` | `services/retriever.py` | 向量 + BM25 并行召回 → RRF 融合 → BGE-Reranker |
| `RAGAgent` | `services/agent.py` | LangGraph 状态机：classify → retrieve / web_search / direct_answer → build_prompt → generate，SSE 事件流 |
| `ImageDescriber` | `services/image_describer.py` | 用 VLM 描述 PDF / DOCX 等内嵌图片，文本占位符替换 |
| `BM25Store` | `services/bm25_store.py` | 每知识库一份 pickle 索引，懒加载 + 缓存 + 增量更新 |
| `ChromaStore` | `vectorstore/chroma_store.py` | Chroma collection 封装，支持按 doc_id 删除 |

## 二、快速开始

### 1. 后端

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # 或直接 pip install（pyproject 已列出全部）
cp ../.env.example .env          # 按需修改
python -m app.main               # 或 uvicorn app.main:app --reload --port 8765
```

健康检查：`curl http://127.0.0.1:8765/health` → `{"status":"ok","version":"0.1.0"}`

API 文档：`http://127.0.0.1:8765/docs`

### 2. 前端（浏览器模式）

```bash
cd frontend
npm install
npm run dev                      # http://127.0.0.1:5173
```

Vite 已将 `/api/*` 反代到 `http://127.0.0.1:8765`（可通过 `VITE_DEV_API_TARGET` 覆盖）。

### 3. 桌面浮窗（Tauri，可选）

```bash
cd frontend
npm install
npm run tauri:dev                # 需要本地有 Rust 工具链
```

> Tauri 2.x 安装：参考 https://tauri.app/start/ 。
> Tauri 下窗口支持全局拖入文件上传（见 `App.vue` 的 `dragenter/leave` 计数器）。

## 三、API 约定

所有响应均使用 `KnowledgeBaseRead` / `DocumentRead` / `SourceItem` 等 Pydantic 模型；
前端 `api/client.ts` 中的 `kbApi` / `docApi` / `searchApi` / `streamChat` / `convApi`
是一一对应的封装。

| 路径 | 方法 | 说明 |
| --- | --- | --- |
| /api/v1/knowledge-bases | GET / POST | 列 / 建（POST 会预创建 Chroma collection） |
| /api/v1/knowledge-bases | GET (?q=) | 按名称/描述模糊搜索 |
| /api/v1/knowledge-bases/{id} | GET / PATCH / DELETE | 详情 / 更新 / 删除（级联清向量+文件） |
| /api/v1/knowledge-bases/{id}/stats | GET | 文档与索引统计（doc_total / doc_ready / doc_failed / bm25_chunks / chroma_chunks） |
| /api/v1/knowledge-bases/{id}/documents | GET / POST | 列表 / 上传（POST 立刻返回 202，后台入库） |
| /api/v1/knowledge-bases/{id}/documents/batch | POST | 批量上传（逐文件独立成败） |
| /api/v1/knowledge-bases/{id}/documents/{doc_id} | DELETE | 删除文档级联清理向量与文件 |
| /api/v1/knowledge-bases/{id}/documents/{doc_id}/retry | POST | 重新入库失败/处理中文档（同步） |
| /api/v1/search | POST | 直接调用混合检索，返回 sources + latency |
| /api/v1/chat | POST | **SSE 流式问答**：intent → sources → token* → meta → final → end（可能发 error） |
| /api/v1/chat/sync | POST | 非流式收敛（便于测试） |
| /api/v1/chat/conversations | GET (?kb_id=) | 会话列表（可按 KB 过滤） |
| /api/v1/chat/conversations/{id}/messages | GET | 消息历史 |
| /api/v1/chat/conversations/{id} | PATCH / DELETE | 改标题 / 删除会话 |

### 通用约束

- 上传文件类型：`.pdf .docx .pptx .md .txt .html .htm .csv .xlsx`
- 上传大小上限：`MAX_UPLOAD_MB`（默认 50 MB）
- 上传大文件：解析与切片走 `asyncio.to_thread`，不阻塞事件循环
- 流式问答客户端断开：`CancelledError` 会被记录但不视为错误

## 四、SSE 事件协议

`/api/v1/chat` 使用 `text/event-stream`，事件类型如下：

| event | data 字段 | 触发时机 |
| --- | --- | --- |
| `intent` | `{"intent":"rag\|web\|direct\|hybrid"}` | 意图分类完成 |
| `sources` | `{"sources":[SourceItem, ...]}` | 检索/联网结束 |
| `token` | `{"content":"..."}` | LLM 流式增量（多次） |
| `meta` | `{latency_ms, intent, used_web, used_rag, answer}` | 流结束前（已包含累计答案） |
| `final` | `ChatFinalEvent` JSON | 持久化成功后：含 `meta.conversation_id / message_id` 和 sources |
| `error` | `{"message":"..."}` | 任一阶段失败（其后必发 `end`） |
| `end` | `{}` | 会话结束（无论成功失败） |

示例：

```
event: intent
data: {"intent":"rag"}

event: sources
data: {"sources":[{"document":"x.pdf","page":3,"score":0.83,"source_type":"vector"}]}

event: token
data: {"content":"根据"}

event: meta
data: {"intent":"rag","latency_ms":1823,"used_web":false,"used_rag":true}

event: final
data: {"meta":{"intent":"rag","conversation_id":42,"message_id":99},"sources":[...]}

event: end
data: {}
```

前端解析见 `frontend/src/api/client.ts` 的 `streamChat`：按 `\n\n` 切事件，并把 `\r\n` 归一化为 `\n`，避免漏解析。

## 五、环境变量

复制 `.env.example` 为 `backend/.env`，按需修改。

### LLM（OpenAI 兼容）

| 变量 | 默认 | 用途 |
| --- | --- | --- |
| `OPENAI_API_KEY` | - | LLM 鉴权；留空时 `LLMFactory` 抛错 |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | 兼容 DeepSeek / 通义 / 自建网关 |
| `LLM_MODEL` | `gpt-4o-mini` | 对话模型 |
| `LLM_TEMPERATURE` | `0.3` | 采样温度 |
| `USE_MOCK_LLM` | `false` | 离线/沙箱测试用，启用后跳过真实 LLM 调用 |

### Embedding

| 变量 | 默认 | 用途 |
| --- | --- | --- |
| `EMBEDDING_PROVIDER` | `local` | `local`（BGE-M3） / `openai` / `mock` |
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-m3` | `local` 时生效 |
| `LOCAL_EMBEDDING_DEVICE` | `cpu` | `cpu` / `cuda` |
| `LOCAL_EMBEDDING_DIM` | `1024` | 与 `BGE-M3` 输出维度一致 |
| `OPENAI_EMBEDDING_API_KEY` | - | `openai` 时使用；缺则回落到 `OPENAI_API_KEY` |
| `OPENAI_EMBEDDING_BASE_URL` | - | `openai` 时使用；缺则回落到 `OPENAI_BASE_URL` |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | `openai` 时生效 |

### Reranker

| 变量 | 默认 | 用途 |
| --- | --- | --- |
| `RERANK_PROVIDER` | `local` | `local`（BGE-Reranker）/ `none`（关闭重排序） |
| `LOCAL_RERANK_MODEL` | `BAAI/bge-reranker-base` | 默认 base（约 280MB）；v2-m3 体积过大，常规场景收益不明显 |
| `RERANK_TOP_K` | `20` | 召回阶段的 TopK |
| `FINAL_TOP_K` | `5` | 重排后送入 LLM 的 TopK |

### Web Search

| 变量 | 默认 | 用途 |
| --- | --- | --- |
| `TAVILY_API_KEY` | - | 留空则降级 DuckDuckGo |
| `ENABLE_WEB_SEARCH` | `true` | 全局联网开关；前端开关再 AND 一次 |
| `MAX_WEB_RESULTS` | `5` | 单次联网召回条数 |

### Image Description（VLM）

| 变量 | 默认 | 用途 |
| --- | --- | --- |
| `IMAGE_DESCRIPTION_API_KEY` | - | 留空走本地 VLM；填写走 OpenAI 兼容 |
| `IMAGE_DESCRIPTION_BASE_URL` | - | 同上 |
| `IMAGE_DESCRIPTION_MODEL` | `gpt-4o-mini` | API 模式生效 |
| `LOCAL_VLM_MODEL` | `Qwen/Qwen3-VL-2B-Instruct` | 本地 VLM |
| `LOCAL_VLM_PATH` | - | 本地模型快照路径；非空时跳过下载 |
| `LOCAL_VLM_DEVICE` | `auto` | `auto` / `cpu` / `cuda` |
| `MAX_IMAGES_PER_DOC` | `50` | 单文档最大处理图片数（截断） |

### Server / Persistence / Ingest / Agent

| 变量 | 默认 | 用途 |
| --- | --- | --- |
| `HOST` / `PORT` | `127.0.0.1` / `8765` | 后端监听 |
| `LOG_LEVEL` | `INFO` | loguru 级别 |
| `CORS_ORIGINS` | `http://localhost:5173,tauri://localhost` | 前端来源白名单（`main.py` 还会追加 tauri/localhost:1420/5173 等） |
| `DATA_DIR` | `./data` | 总目录 |
| `UPLOAD_DIR` | `./data/uploads` | 上传文件按 `<kb_slug>/` 分目录存放 |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | 向量库持久化 |
| `SQLITE_PATH` | `./data/metadata.db` | 元数据库 |
| `BM25_INDEX_DIR` | `./data/bm25` | BM25 pickle |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `600` / `120` | 切片参数（知识库创建时也会复制为 KB 字段） |
| `MAX_UPLOAD_MB` | `50` | 上传大小限制 |
| `ENABLE_MULTI_QUERY` | `true` | 是否启用查询改写（保留位，当前实现走 RAG/Web/Direct 三路） |
| `ENABLE_HYDE` | `false` | HyDE 开关（保留位） |

### HuggingFace / 本地模型

| 变量 | 默认 | 用途 |
| --- | --- | --- |
| `HF_ENDPOINT` | - | 镜像站（如 `https://hf-mirror.com`），加速 BGE / VLM 下载 |
| `LOCAL_MODEL_ROOT` | `./.model` | 本地模型副本根目录 |
| `LOCAL_EMBEDDING_PATH` | - | 显式覆盖本地 embedding 路径 |
| `LOCAL_RERANK_PATH` | - | 显式覆盖本地 reranker 路径 |

启动时（`lifespan`）会预热 Reranker，避免首次对话等待下载；预热失败仅记 warning，可懒加载重试。

## 六、开发与调试

### 后端

```bash
cd backend
pytest -q                                # test_loaders / test_splitters / test_bm25 / test_api
ruff check app                           # 静态检查
uvicorn app.main:app --reload --port 8765
```

测试覆盖：

- `test_loaders.py` — PDF/DOCX/MD/TXT/CSV/HTML/XLSX Loader 解析
- `test_splitters.py` — 切片与 metadata 注入
- `test_bm25.py` — BM25 索引增删与查询
- `test_api.py` — FastAPI 路由（`/health` / KB CRUD / chat）

> 首次启动会按需下载 BGE-M3 / Reranker 模型；建议预热或切换 OpenAI Embedding。

### 前端

```bash
cd frontend
npm run typecheck      # vue-tsc --noEmit
npm run build          # 生产构建（含 vue-tsc）
npm run dev            # 开发服务（含 /api 反代）
```

约定：

- Naive UI 走 `unplugin-vue-components` + `NaiveUiResolver` 按需自动注册
- 路由 hash 模式（适配 Tauri file:// 协议）；`App.vue` 根据 `route.name` 决定主面板内容
- `getApiBaseURL()` 优先读 `localStorage.rag.settings.apiBase`，回落到 `VITE_API_BASE` → 默认 `http://127.0.0.1:8765`
- 全局拖入文件：监听 `dragenter/leave/over/drop`，仅当 `dataTransfer.types` 含 `Files` 时触发

## 七、已知风险与回退

| 风险 | 回退方案 |
| --- | --- |
| BGE 本地模型下载慢/失败 | `EMBEDDING_PROVIDER=openai` 切到 OpenAI Embedding；或 `HF_ENDPOINT=https://hf-mirror.com` |
| Tauri 全局快捷键权限被拒 | macOS 系统设置 → 隐私与安全 → 辅助功能 |
| Tauri 编译失败 | 直接用 `npm run dev` 在浏览器中跑通业务，后续补 Tauri |
| Tavily Key 缺失 / 超额 | 自动降级 DuckDuckGo；可在对话中关闭"联网"开关 |
| LangGraph 与 LangChain 版本冲突 | `pyproject.toml` 已锁定基线（langchain≥0.3.7, langgraph≥0.2.45） |
| Chroma 容器被并发重置 | 默认单进程本地持久化；多进程需切换到 Qdrant |
| 上传大文件 OOM | `MAX_UPLOAD_MB` 控制；解析走 `to_thread` 不阻塞事件循环 |
| 图片描述 OOM / 慢 | `MAX_IMAGES_PER_DOC` 截断；用本地 VLM 时 `LOCAL_VLM_DEVICE=cpu` 风险高，建议 GPU |
| Embedding 维度变更 | 知识库已有向量时 `PATCH` 改 embedding 配置会被拒绝（409），需先清空文档 |
| 上传中文文件名乱码 | Linux 下 `python-pptx` 等可能解码异常；目前入口已统一用 `Path(filename).name` 兜底 |
