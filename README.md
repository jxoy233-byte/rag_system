# 通用文档问答 RAG 系统

基于 Vue 3 + Tauri 桌面浮窗的通用 RAG 系统，支持：

- **本地混合检索**（向量 + BM25 + RRF + BGE-Reranker）
- **联网搜索**（Tavily 优先，DuckDuckGo 兜底）
- **基础对话**（无外部知识）
- 多轮会话、引用溯源、SSE 流式输出

> 设计与阶段划分见 [`agent.md`](./agent.md)。

## 目录

- 一、技术栈与结构
- 二、快速开始
- 三、API 约定
- 四、环境变量
- 五、开发与调试
- 六、已知风险与回退

## 一、技术栈与结构

| 层 | 技术 |
| --- | --- |
| 后端 | Python 3.11+ · FastAPI · LangChain/LangGraph · SQLAlchemy 2 (async) · Chroma · BM25 (rank-bm25 + jieba) · FlagEmbedding (BGE-M3 / BGE-Reranker) · Tavily / DuckDuckGo |
| 前端 | Vue 3.5 + TS · Vite 6 · Pinia 2 · Vue Router 4 · Naive UI（按需自动注册） · lucide-vue-next · Markdown-it · ofetch |
| 桌面壳 | Tauri 2.x（全局快捷键、悬浮窗口、托盘） |
| 持久化 | SQLite 元数据 + Chroma 向量库 + Pickle BM25 索引 + 本地上传文件 |

```
rag_system/
|-- backend/                 # FastAPI 后端
|   |-- app/
|   |   |-- api/v1/          # knowledge_bases / documents / search / chat (SSE)
|   |   |-- core/            # config / logging / db / deps
|   |   |-- loaders/         # PDF / DOCX / PPTX / MD / TXT / HTML / CSV / XLSX
|   |   |-- splitters/       # Recursive splitter（langchain）
|   |   |-- embeddings/      # BGE-M3 / OpenAI 兼容
|   |   |-- vectorstore/     # ChromaStore 封装
|   |   |-- rerankers/       # BGE-Reranker (FlagEmbedding / CrossEncoder)
|   |   |-- websearch/       # Tavily / DDG
|   |   |-- llm/             # OpenAI 兼容 Chat/Stream
|   |   |-- services/        # ingest / retriever / agent (LangGraph) / bm25
|   |   |-- models/          # SQLAlchemy ORM
|   |   |-- schemas/         # Pydantic DTO
|   |-- tests/               # pytest-asyncio
|   |-- pyproject.toml
|-- frontend/
|   |-- src/
|   |   |-- api/client.ts    # ofetch 封装 + SSE 解析
|   |   |-- components/      # FloatingBall / ChatPanel / MessageBubble / KnowledgePicker
|   |   |-- stores/          # Pinia: settings / knowledgeBase / chat
|   |   |-- views/           # ChatView / KBView / DocsView / SettingsView
|   |   |-- styles/main.css  # 主题 / Markdown / 滚动条
|   |-- vite.config.ts
|   |-- tsconfig.json
|   |-- index.html
|-- data/                    # 运行时持久化（gitignore）
|   |-- chroma/
|   |-- bm25/
|   |-- uploads/
|   |-- metadata.db
|-- agent.md
```

## 二、快速开始

### 1. 后端

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # 或直接 pip install（pyproject 已列出全部）
cp ../.env.example .env          # 按需修改
python -m app.main               # 或 uvicorn app.main:app --reload --port 8765
```

健康检查：`curl http://127.0.0.1:8765/health` -> `{"status":"ok","version":"0.1.0"}`

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

## 三、API 约定

所有响应均使用 `KnowledgeBaseRead` / `DocumentRead` / `SourceItem` 等 Pydantic 模型；
前端 `api/client.ts` 中的 `kbApi` / `docApi` / `searchApi` / `streamChat` / `convApi`
是一一对应的封装。

| 路径 | 方法 | 说明 |
| --- | --- | --- |
| /api/v1/knowledge-bases | GET / POST | 列 / 建 |
| /api/v1/knowledge-bases/{id} | GET / PATCH / DELETE | 详情 / 更新 / 删除 |
| /api/v1/knowledge-bases/{id}/stats | GET | 文档与索引统计 |
| /api/v1/knowledge-bases/{id}/documents | GET / POST | 列表 / 上传（后台入库） |
| /api/v1/knowledge-bases/{id}/documents/batch | POST | 批量上传 |
| /api/v1/knowledge-bases/{id}/documents/{doc_id} | DELETE | 删除文档级联清理向量 |
| /api/v1/search | POST | 直接调用混合检索 |
| /api/v1/chat | POST | **SSE 流式问答**：intent -> sources -> token* -> meta -> final -> end |
| /api/v1/chat/sync | POST | 非流式收敛（便于测试） |
| /api/v1/chat/conversations | GET | 会话列表（可按 KB 过滤） |
| /api/v1/chat/conversations/{id}/messages | GET | 消息历史 |
| /api/v1/chat/conversations/{id} | DELETE | 删除会话 |

### SSE 事件序列示例

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
data: {"meta":{"intent":"rag","conversation_id":42,"message_id":99},"sources":[…]}

event: end
data: {}
```

前端解析见 `frontend/src/api/client.ts` 的 `streamChat`。

## 四、环境变量

复制 `.env.example` 为 `backend/.env`，按需修改。常用项：

| 变量 | 默认 | 用途 |
| --- | --- | --- |
| OPENAI_API_KEY / OPENAI_BASE_URL | - / openai.com | LLM（兼容 OpenAI 协议，可换 DeepSeek 等） |
| LLM_MODEL | gpt-4o-mini | 对话模型 |
| EMBEDDING_PROVIDER | local | `local`（BGE-M3） 或 `openai` |
| LOCAL_EMBEDDING_MODEL | BAAI/bge-m3 | 本地 Embedding 模型 |
| OPENAI_EMBEDDING_* | - | `EMBEDDING_PROVIDER=openai` 时使用 |
| LOCAL_RERANK_MODEL | BAAI/bge-reranker-v2-m3 | 本地 Reranker |
| RERANK_PROVIDER | local | 可设为 `none` 关闭 |
| TAVILY_API_KEY | - | 留空则降级 DuckDuckGo |
| ENABLE_WEB_SEARCH | true | 全局联网开关 |
| HOST / PORT | 127.0.0.1 / 8765 | 后端监听 |
| CORS_ORIGINS | http://localhost:5173,tauri://localhost | 前端来源白名单 |
| DATA_DIR / CHROMA_PERSIST_DIR / SQLITE_PATH / BM25_INDEX_DIR / UPLOAD_DIR | ./data/* | 持久化路径 |
| CHUNK_SIZE / CHUNK_OVERLAP | 600 / 120 | 切片参数 |
| MAX_UPLOAD_MB | 50 | 上传大小限制 |

## 五、开发与调试

### 后端

```bash
cd backend
pytest -q                                # test_loaders / test_splitters / test_bm25 / test_api
ruff check app                           # 静态检查
uvicorn app.main:app --reload --port 8765
```

> 首次启动会按需下载 BGE-M3 / Reranker 模型；建议预热或切换 OpenAI Embedding。

### 前端

```bash
cd frontend
npm run typecheck      # vue-tsc --noEmit
npm run build          # 生产构建
npm run dev            # 开发服务（含 /api 反代）
```

Naive UI 走按需自动注册，无需手动引入；自定义组件位于 `src/components/`，
会被 `unplugin-vue-components` 全局注册。

## 六、已知风险与回退

| 风险 | 回退方案 |
| --- | --- |
| BGE 本地模型下载慢/失败 | `EMBEDDING_PROVIDER=openai` 切到 OpenAI Embedding |
| Tauri 全局快捷键权限被拒 | macOS 系统设置 -> 隐私与安全 -> 辅助功能 |
| Tauri 编译失败 | 直接用 `npm run dev` 在浏览器中跑通业务，后续补 Tauri |
| Tavily Key 缺失 / 超额 | 自动降级 DuckDuckGo，可在对话中关闭“联网”开关 |
| LangGraph 与 LangChain 版本冲突 | `pyproject.toml` 已锁定基线版本 |
| Chroma 容器被并发重置 | 默认单进程本地持久化；多进程需切换到 Qdrant |
| 上传大文件 OOM | `MAX_UPLOAD_MB` 控制；解析走 to_thread 不阻塞事件循环 |
