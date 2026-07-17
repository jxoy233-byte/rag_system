# Agent.md - 通用文档问答 RAG 系统（桌面浮窗）

> 本文档是项目总纲与执行指南，面向开发者与后续接手的 AI agent。所有架构与阶段决策以本文档为准。
> 日常用户向文档见 [README.md](./README.md)；本文档偏设计与决策记录。

## 一、项目目标

构建一个面向个人/小团队的通用文档问答 RAG 系统，覆盖四种回答路径：

1. **RAG 检索**：基于本地知识库的混合检索问答。
2. **联网搜索**：本地知识不足时调用搜索引擎补全。
3. **基础对话**：无外部知识的常规 LLM 对话。
4. **混合（hybrid）**：同时使用本地文档 + 联网结果。

辅助能力：图片理解（VLM 描述后并入切片）、多轮会话、引用溯源、SSE 流式输出。

产品形态：Vue 3 + Tauri 包装的桌面浮窗应用，点击或全局快捷键唤起，支持文档入库 / 管理 / 问答 / 设置。

## 二、核心质量要求

| 维度 | 要求 |
| --- | --- |
| 检索质量 | 混合检索（向量 + BM25 + RRF + 重排序），目标 Recall@5 ≥ 0.85 |
| 响应延迟 | 流式首 token < 2s（API）/ < 1s（本地模型） |
| 鲁棒性 | 所有外部调用重试 + 降级；失败不卡死，输出可读错误 |
| 可观测 | LangGraph 节点级日志，关键路径 trace_id（保留位） |
| 安全 | API Key 走环境变量；CORS 白名单；上传文件类型与大小校验 |
| 可移植 | 提供本地 dev（pip + npm）；Docker compose 后续补 |

## 三、技术栈

### 后端（Python 3.11+）
- Web 框架：**FastAPI** + Uvicorn（REST + SSE 流式 via `sse-starlette`）
- LLM 编排：**LangChain**（Loader / Splitter / OpenAI 兼容 Chat）
- Agent 编排：**LangGraph**（StateGraph + 路由 + 工具）
- 向量库：**ChromaDB**（默认本地持久化）/ Qdrant（生产可切换）
- 关键词检索：**BM25**（rank_bm25 + jieba 分词，每 KB 一份 pickle）
- Embedding：**BAAI/bge-m3**（本地 FlagEmbedding）/ OpenAI text-embedding-3-small（API）/ mock
- 重排序：**BAAI/bge-reranker-base**（本地，约 280MB）/ 关闭（`RERANK_PROVIDER=none`）
- 联网搜索：**Tavily**（默认）/ DuckDuckGo（备用，无 Key）
- 图片理解：**Qwen3-VL-2B-Instruct**（本地）/ OpenAI 兼容 VLM（可选）
- 元数据存储：SQLite + SQLAlchemy 2（async）+ aiosqlite
- 后台任务：`FastAPI BackgroundTasks`（同进程异步入库；重任务可换 APScheduler / Arq）

### 前端（桌面浮窗）
- 框架：**Vue 3.5+** + TypeScript + Vite 6
- 状态管理：Pinia 2
- 路由：Vue Router 4（**hash 模式**，适配 Tauri `file://`）
- UI 库：**Naive UI**（按需自动注册 via `unplugin-vue-components` + `NaiveUiResolver`）
- 图标：lucide-vue-next
- HTTP：ofetch（轻量 SSE 友好；SSE 自实现 `fetch + ReadableStream`）
- Markdown：markdown-it + highlight.js
- 桌面壳：**Tauri 2.x**（无边框悬浮 + 全局快捷键 + 系统托盘 + 文件拖入）

### 持久化
- 元数据：SQLite（KB / Document / Conversation / Message）
- 向量：Chroma（每 KB 一个 collection）
- 关键词索引：BM25 pickle（每 KB 一份；懒加载 + 内存缓存）
- 文件：`data/uploads/<kb_slug>/<filename>`

## 四、系统架构

### 4.1 整体数据流

```
                                +------------------+
                                |  Tauri Desktop   |
                                |   (Vue3 浮窗)    |
                                +---------+--------+
                                          | HTTP / SSE
                                          v
+------------------+    +--------------------------------+
|  用户上传文档     | -> |  FastAPI (/api/v1/documents)   |
|  / 拖入文件      |    +----------------+---------------+
+------------------+                     |
                                          v
                            +-------------+-------------+
                            |  IngestService (后台)     |
                            +-------------+-------------+
                                          |
                  +-----------------------+-----------------------+
                  v                       v                       v
        +-----------------+      +------------------+    +-----------------+
        | Loader + Splitter|     | ImageDescriber   |    | 元数据 SQLite   |
        | + VLM 描述替换   |     | (本地/OpenAI)     |    |  (KB/Doc/Msg)  |
        +--------+--------+      +--------+---------+    +-----------------+
                 |                         |
                 v                         v
        +-----------------+       +------------------+
        | Embedding (BGE) |       | BM25 (jieba)     |
        +--------+--------+       +--------+---------+
                 |                         |
                 +-------------+-----------+
                               v
                     +------------------+
                     | ChromaDB         |
                     |  collection/库   |
                     +------------------+

                                +------------------+
         用户提问  ------------->| LangGraph Agent  |
                                +--------+---------+
                                         |
                +------------------------+------------------------+
                v                        v                        v
        +---------------+        +----------------+      +----------------+
        | 混合检索 (RAG) |        | Web Search     |      | Direct Answer  |
        | (vec+BM25+RRF |        | (Tavily/DDG)   |      | (纯 LLM)       |
        |  + Rerank)     |        +-------+--------+      +--------+-------+
        +-------+-------+                |                       |
                |                        v                       |
                v                  +-------------+               |
        +-------------------+      | Summarize   |               |
        | build_prompt      |<-----+-------------+               |
        +---------+---------+                                    |
                  |                                              |
                  v                                              v
            +----------------+                          +----------------+
            | generate (SSE) |                          | direct_answer  |
            +--------+-------+                          +--------+-------+
                     |                                           |
                     +-----------------------+-------------------+
                                         v
                                  SSE 流式返回前端
```

### 4.2 LangGraph 工作流

实际节点（`services/agent.py`）：

```
                 +-----------+
        START -> | classify  |   意图分类：rag / web / direct / hybrid
                 +-----+-----+   (无 KB 时直接降级为 direct)
                       |
       +-------+-------+--------+----------+
       v       v                v          v
   +--------+ +--------+   +---------+  +-----------+
   | direct | | retrieve|  | web_    |  | retrieve  |
   | _answer| | (rag)   |  | search  |  | (hybrid)  |
   +---+----+ +---+----+  +----+----+  +-----+-----+
       |          |             |             |
       |          |             v             v
       |          |        (retrieval 完成后再 web_search)
       |          |             |             |
       |          v             v             v
       |     +----------------------+    (no edge)
       |     |  build_prompt        |    (路由到
       |     |  (组装 context)      |     web_search
       |     +----------+-----------+     然后回 build_prompt)
       |                |
       v                v
   +---------+    +---------+
   |  END    |    | generate| -> END
   +---------+    +---------+
```

事件序列：`intent` → `sources`（如有） → `token*` → `meta` → `final`（持久化后） → `end`；任何阶段失败发 `error` 后仍发 `end`。

### 4.3 混合检索流水线（`services/retriever.py`）

```
              +-----------+        +-----------+
   query ---->|  vec q    |        |  bm25 q   |
              | (Chroma)  |        | (pickle)  |
              +-----+-----+        +-----+-----+
                    |   parallel (asyncio.gather)
                    v                      v
              +-----------+        +-----------+
              | top_k=20  |        | top_k=20  |
              +-----+-----+        +-----+-----+
                    \                      /
                     \                    /
                      v                  v
                    +---------------------+
                    |   RRF 融合 (k=60)   |
                    |  score = 1/(k+r)    |
                    +----------+----------+
                               |
                          fused → top_k
                               |
                               v
                    +---------------------+
                    | BGE-Reranker (可选) |
                    +----------+----------+
                               |
                               v
                       final top_k 给 LLM
```

## 五、目录结构

```
rag_system/
|-- agent.md                       # 本文件（设计总纲）
|-- README.md                      # 用户向文档
|-- .env.example
|-- backend/
|   |-- pyproject.toml
|   |-- app/
|   |   |-- main.py                # FastAPI 入口 + lifespan + 预热 reranker
|   |   |-- core/
|   |   |   |-- config.py          # pydantic-settings
|   |   |   |-- logging.py         # loguru
|   |   |   |-- db.py              # async engine + AsyncSessionLocal
|   |   |   |-- deps.py            # FastAPI Depends 注入 AsyncSession
|   |   |   `-- local_model.py     # HF_ENDPOINT 解析 + 本地路径映射
|   |   |-- api/
|   |   |   `-- v1/
|   |   |       |-- knowledge_bases.py   # KB CRUD + stats
|   |   |       |-- documents.py         # 上传 / 列表 / 删除 / 重试
|   |   |       |-- chat.py              # SSE 流式 + 会话管理
|   |   |       `-- search.py            # 直接混合检索
|   |   |-- models/                # KnowledgeBase / Document / Conversation / Message
|   |   |-- schemas/               # Pydantic DTO
|   |   |-- services/
|   |   |   |-- ingest.py          # 入库主流程
|   |   |   |-- retriever.py       # 混合检索
|   |   |   |-- agent.py           # LangGraph
|   |   |   |-- bm25_store.py      # pickle 索引
|   |   |   `-- image_describer.py # VLM 描述
|   |   |-- loaders/               # PDF / DOCX / PPTX / MD / TXT / HTML / CSV / XLSX
|   |   |-- splitters/             # 切片策略
|   |   |-- embeddings/            # Embedding 工厂
|   |   |-- vectorstore/           # Chroma 封装
|   |   |-- rerankers/             # BGE-Reranker
|   |   |-- websearch/             # Tavily / DDG
|   |   `-- llm/                   # LLM 工厂
|   `-- tests/                     # pytest-asyncio
|-- frontend/
|   |-- package.json
|   |-- vite.config.ts
|   |-- index.html
|   |-- src/
|   |   |-- main.ts
|   |   |-- App.vue                # FloatingBall + 主面板分发（按 route.name 切换）
|   |   |-- router.ts              # hash 模式
|   |   |-- views/
|   |   |   |-- KBView.vue         # KB CRUD + 统计
|   |   |   |-- DocsView.vue       # 文档管理
|   |   |   |-- SearchView.vue     # 直接检索
|   |   |   `-- SettingsView.vue
|   |   |-- components/
|   |   |   |-- FloatingBall.vue   # 悬浮球
|   |   |   |-- ChatPanel.vue      # 对话主面板
|   |   |   |-- MessageBubble.vue  # 消息气泡 + Markdown
|   |   |   |-- ConversationList.vue
|   |   |   `-- KnowledgePicker.vue
|   |   |-- stores/                # Pinia: settings / knowledgeBase / chat
|   |   |-- api/                   # ofetch 封装 + SSE 解析
|   |   |-- utils/                 # isTauri 等
|   |   `-- styles/main.css
|   `-- src-tauri/
|       |-- tauri.conf.json
|       `-- src/main.rs            # 全局快捷键 + 悬浮 + 托盘
`-- data/                          # 运行时持久化（gitignore）
     |-- chroma/
     |-- bm25/
     |-- uploads/<kb_slug>/
     `-- metadata.db
```

## 六、核心功能清单

### 6.1 知识库与文档管理
- 知识库（KB）CRUD：创建 / 重命名 / 删除 / 切换（软切换：`settings.kbId`）
- 多格式文档上传：PDF、DOCX、PPTX、MD、TXT、HTML、CSV、XLSX
- 图片理解：Loader 抽出内嵌图片 → VLM 描述 → 替换占位符 `[图片:n]` → 拼回文本切片
- 切片策略：Recursive splitter（langchain），chunk_size / overlap 可在 KB 维度配置
- 元数据保留：`doc_id / doc_title / doc_filename / page / section`
- 入库进度：上传返回 202 + Document 状态 `processing`；`retry` 端点可重试 `failed / processing`
- 文档删除级联清理：Chroma `delete_by_doc_id` + BM25 `delete_by_doc_id` + 删除上传文件

### 6.2 混合检索
- 向量检索：默认 BGE-M3，可切 OpenAI / mock
- 关键词检索：BM25Okapi（中文按字 + jieba 分词）
- 召回并行：`asyncio.gather(vec, bm25)`
- 融合：**Reciprocal Rank Fusion (RRF, k=60)**
- 重排序：Cross-Encoder BGE-Reranker-Base（可关闭）
- 知识库隔离：每个 KB 独立 Chroma collection + BM25 pickle

### 6.3 问答 Agent（LangGraph）
- 意图分类节点：基于 LLM few-shot，路由 `rag / web / direct / hybrid`
- RAG 路径：retrieve → build_prompt → generate，附引用
- 联网路径：web_search → build_prompt → generate
- 混合路径：retrieve → web_search → build_prompt → generate
- 直答路径：direct_answer → END
- 流式输出（SSE）：前端 token 级渲染
- 多轮对话：会话级 memory，最近 6 轮喂入 LLM
- 引用溯源：返回 `kb_id / doc_id / page / chunk_id / score / source_type`

### 6.4 桌面浮窗
- 无边框悬浮球，初始 60px，hover 放大
- 全局快捷键：默认 `Cmd/Ctrl+Shift+Space` 唤起 / 收起
- 主面板：约 800×600 圆角浮窗，毛玻璃背景
- 系统托盘菜单：显示 / 隐藏 / 退出
- 全局文件拖入：`App.vue` 用 dragenter/leave 计数器；`isTauri` 时走 Tauri `invoke` 走原生对话框
- 窗口置顶可切换（`settings.alwaysOnTop`）

### 6.5 设置
- 后端地址（前端 localStorage 持久化）
- 主题（亮 / 暗）
- 联网搜索开关（前端 → 同步到后端 `payload.enable_web`）
- LLM / Embedding / Reranker 选择（后端 `.env` 决定，前端只展示）

## 七、开发阶段

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 0. 准备 | 写 agent.md、目录骨架、依赖清单 | ✅ |
| 1. 后端骨架 | FastAPI 工程、配置/日志/SQLite、健康检查、API 路由占位 | ✅ |
| 2. 前端骨架 | Vue3 + Tauri 浮窗可启动；与后端 `/health` 联通 | ✅ |
| 3. 文档入库 | Loader + Splitter + Embedding + Chroma 分区 + 上传 API + 元数据 | ✅ |
| 4. 混合检索 | 向量 + BM25 + RRF + Rerank | ✅ |
| 5. LangGraph Agent | classify → retrieve/web/direct → generate，SSE | ✅ |
| 6. 前端业务 | 文档管理 UI、聊天 UI、KB 切换、设置、检索页 | ✅ |
| 7. 质量与优化 | 检索评测、错误处理、性能、E2E、Docker | 🟡 部分完成 |

> 第 7 阶段剩余项：评测数据集 / LangSmith 接入 / Docker compose / CI。

## 八、关键设计决策

- **向量库默认 Chroma**：零依赖启动；后续可一行配置切 Qdrant。
- **Embedding 默认 BGE-M3**：本地、可商用、中文强；首次启动自动下载，可在 `EMBEDDING_PROVIDER` 切到 OpenAI。
- **Reranker 用 Base 而非 v2-m3**：默认 `BAAI/bge-reranker-base`（~280MB），普通文档场景与 v2-m3 拉不开差距，但省下 8× 体积。
- **LLM 默认 OpenAI 兼容接口**：DeepSeek / OpenAI / Claude（via proxy）同协议切换；`USE_MOCK_LLM=true` 用于沙箱测试。
- **联网默认 Tavily**：AI 友好摘要；Key 缺失自动降级 DuckDuckGo。
- **Agent 用 LangGraph StateGraph**：节点清晰、可观测、可单步调试；6 个节点（classify / retrieve / web_search / build_prompt / generate / direct_answer）。
- **流式用 SSE 而非 WebSocket**：单向 + 简单 + FastAPI 原生支持；`sse-starlette` 提供 ping keepalive。
- **前端用 Tauri**：包体小、权限细、快捷键原生；hash 模式路由避免 file:// 协议问题。
- **图片描述独立模块**：与切片解耦，失败仅 warning 不阻塞入库（占位符保留）。
- **BM25 + Chroma 双写**：检索时并行召回 + RRF 融合；任一缺失都降级到单路。
- **不做账号系统**：本地工具，简化复杂度。

## 九、环境变量（.env.example）

完整列表见 [README.md#五环境变量](./README.md#五环境变量)，分类速查：

```
LLM:        OPENAI_API_KEY / OPENAI_BASE_URL / LLM_MODEL / LLM_TEMPERATURE / USE_MOCK_LLM
Embedding:  EMBEDDING_PROVIDER / LOCAL_EMBEDDING_* / OPENAI_EMBEDDING_*
Rerank:     RERANK_PROVIDER / LOCAL_RERANK_MODEL / RERANK_TOP_K / FINAL_TOP_K
Web:        TAVILY_API_KEY / ENABLE_WEB_SEARCH / MAX_WEB_RESULTS
VLM:        IMAGE_DESCRIPTION_API_KEY / IMAGE_DESCRIPTION_BASE_URL / IMAGE_DESCRIPTION_MODEL
            LOCAL_VLM_MODEL / LOCAL_VLM_PATH / LOCAL_VLM_DEVICE / MAX_IMAGES_PER_DOC
Server:     HOST / PORT / LOG_LEVEL / CORS_ORIGINS
Persist:    DATA_DIR / UPLOAD_DIR / CHROMA_PERSIST_DIR / SQLITE_PATH / BM25_INDEX_DIR
Ingest:     CHUNK_SIZE / CHUNK_OVERLAP / MAX_UPLOAD_MB
Agent:      ENABLE_MULTI_QUERY / ENABLE_HYDE
HF/Model:   HF_ENDPOINT / LOCAL_MODEL_ROOT / LOCAL_EMBEDDING_PATH / LOCAL_RERANK_PATH
```

## 十、执行待办（首个 Sprint）

- [x] 初始化 backend pyproject + 依赖（fastapi/langchain/langgraph/chroma/flagembedding/sentence-transformers/jieba/tavily/ddg）
- [x] 初始化 frontend Vue3 + Tauri 2.x 骨架（Vite 6、Pinia、Vue Router hash、Naive UI 按需）
- [x] 搭 FastAPI 路由占位与 `/health`
- [x] 写 `loader/splitter/embedding/vectorstore` 抽象
- [x] 实现 PDF/DOCX/PPTX/MD/TXT/HTML/CSV/XLSX 上传与入库
- [x] 实现 BM25 + 向量混合检索 + Rerank（RRF 融合）
- [x] 实现 LangGraph classify → RAG / Web / Direct / Hybrid
- [x] 实现图片理解（VLM）模块 + 文本占位符替换
- [x] 前端浮窗 + 聊天 + 文档管理 + KB 切换 + 设置 + 检索页
- [x] 前端 SSE 解析（CRLF→LF 归一化、按 `\n\n` 切事件）
- [x] 端到端 demo 与 README（浏览器模式）；Tauri 桌面壳持续完善
- [x] 后台任务化文档入库（`BackgroundTasks`）+ 重试端点
- [x] Reranker 启动预热（避免首次对话等待下载）
- [x] Embedding 维度变更时拒绝修改已有向量的 KB

### 已知未完成 / 后续

- 检索评测集与 Recall@5 自动化测试
- LangSmith / OpenTelemetry trace 接入
- Docker compose 一键起
- CI（lint + typecheck + test）
- Tauri 全局快捷键默认绑定（需用户在系统设置授权）
- 查询改写（Multi-Query）/ HyDE 真实实现（目前仅保留开关位）

## 十一、风险与回退

| 风险 | 回退方案 |
| --- | --- |
| BGE 本地模型下载慢 | `EMBEDDING_PROVIDER=openai` 切到 OpenAI Embedding；或 `HF_ENDPOINT=https://hf-mirror.com` |
| Tauri 编译失败（Rust 工具链） | 临时用 Vite dev + 系统浏览器跑通业务，后续补 Tauri |
| Tavily 付费/Key 缺失 | 自动用 DuckDuckGo |
| LangGraph 与 LangChain 版本冲突 | 锁定版本（langchain≥0.3.7, langgraph≥0.2.45） |
| macOS 全局快捷键权限 | 提示用户在系统设置授权 |
| VLM 推理 OOM | `MAX_IMAGES_PER_DOC` 截断；优先 GPU；图片描述失败时保留占位符继续入库 |
| Embedding 维度变更 | 知识库已有向量时 `PATCH` 改 embedding 配置会被 409 拒绝，避免索引错位 |
| 上传大文件 OOM | `MAX_UPLOAD_MB` 控制；解析走 `to_thread` 不阻塞事件循环 |
| Chroma 容器被并发重置 | 默认单进程本地持久化；多进程需切换到 Qdrant |
