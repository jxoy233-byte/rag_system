# Agent.md - 通用文档问答 RAG 系统（桌面浮窗）

> 本文档是项目总纲与执行指南，面向开发者与后续接手的 AI agent。所有架构与阶段决策以本文档为准。

## 一、项目目标

构建一个面向个人/小团队的通用文档问答 RAG 系统，覆盖三种回答路径：

1. **RAG 检索：基于本地知识库的混合检索问答。
2. **联网搜索：本地知识不足时调用搜索引擎补全。
3. **基础对话：无外部知识的常规 LLM 对话。

产品形态：Vue3 + Tauri 包装的桌面浮窗应用，点击或全局快捷键唤起，支持文档入库/管理/问答/设置。

## 二、核心质量要求

| 维度 | 要求 |
| --- | --- |
| 检索质量 | 混合检索（向量 + BM25 + RRF + 重排序），目标 Recall@5 >= 0.85 |
| 响应延迟 | 流式首 token < 2s（API）/ < 1s（本地模型） |
| 鲁棒性 | 所有外部调用重试 + 降级；失败不卡死，输出可读错误 |
| 可观测 | LangGraph 节点级日志，关键路径 trace_id |
| 安全 | API Key 走环境变量；CORS 白名单；上传文件类型与大小校验 |
| 可移植 | 提供 Docker compose 一键起；本地 dev 支持纯 pip + npm |

## 三、技术栈

### 后端（Python 3.12）
- Web 框架：**FastAPI** + Uvicorn（REST + SSE 流式）
- LLM 编排：**LangChain**（Loader/Splitter/Retriever/Chain）
- Agent 编排：**LangGraph**（StateGraph + 路由 + 工具）
- 向量库：**ChromaDB**（默认本地持久化）/ Qdrant（生产可切换）
- 关键词检索：**BM25**（rank_bm25 或 Elasticsearch）
- Embedding：**BAAI/bge-m3**（本地 FlagEmbedding）/ OpenAI text-embedding-3-small（API）
- 重排序：**BAAI/bge-reranker-v2-m3**（本地）/ Cohere Rerank（API）
- 联网搜索：**Tavily**（默认）/ DuckDuckGo（备用，无 Key）
- 元数据存储：SQLite + SQLAlchemy 2.x
- 任务队列：APScheduler（后台入库任务）

### 前端（桌面浮窗）
- 框架：**Vue 3.4+** + TypeScript + Vite 5
- 状态管理：Pinia
- 路由：Vue Router（hash 模式，适配 Tauri）
- UI 库：**Naive UI**（轻量、主题友好）
- 图标：lucide-vue-next
- HTTP：ofetch（轻量 SSE 友好）
- Markdown：markdown-it + highlight.js
- 桌面壳：**Tauri 2.x**（无边框悬浮 + 全局快捷键 + 系统托盘）

## 四、系统架构

```
                                +------------------+
                                |  Tauri Desktop   |
                                |   (Vue3 浮窗)    |
                                +---------+--------+
                                          | HTTP / SSE
                                          v
+------------------+    +--------------------------------+
|  用户上传文档     | -> |  FastAPI (/api/v1/documents)   |
+------------------+    +----------------+---------------+
                                          |
                  +-----------------------+-----------------------+
                  v                       v                       v
        +-----------------+      +------------------+    +-----------------+
        | Loader + Splitter|     | Embedding (BGE)  |    | 元数据 SQLite   |
        +-----------------+      +--------+---------+    +-----------------+
                                          |
                                          v
                                +------------------+
                                | ChromaDB (分区)  |
                                |  collection/库   |
                                +------------------+

                                +------------------+
         用户提问  ------------->| LangGraph Agent  |
                                +--------+---------+
                                         |
                +------------------------+------------------------+
                v                        v                        v
        +---------------+        +----------------+      +----------------+
        | 混合检索 (RAG) |        | Tavily 联网    |      | 直答 (基础对话) |
        +-------+-------+        +-------+--------+      +--------+-------+
                |                        |                       |
                v                        v                       v
        +------------------------------------------------------------+
        |              LLM 生成 (OpenAI / DeepSeek / 本地)             |
        +------------------------------------------------------------+
                                         |
                                         v
                                  SSE 流式返回前端
```

### LangGraph 工作流（草图）

```
                 +-----------+
        START -> | classify  |  意图分类：RAG / Web / Direct
                 +-----+-----+       |
                       |   +---------+---------+
                       v   v                   v
                +-------------+        +-----------+      +-----------+
                | retrieve    |        | web_search|      | direct    |
                | (hybrid)    |        | (Tavily)  |      | answer    |
                +------+------+        +-----+-----+      +-----+-----+
                       |                    |                  |
                       v                    v                  v
                  +---------+         +-----------+        +-----------+
                  | rerank  |         | summarize |        |    END    |
                  +----+----+         +-----+-----+        +-----------+
                       |                   |
                       +---------+---------+
                                 v
                          +-------------+
                          |  generate   |  -> END
                          +-------------+
```

## 五、目录结构

```
rag_system/
|-- agent.md                       # 本文件
|-- README.md
|-- .env.example
|-- backend/
|   |-- pyproject.toml
|   |-- app/
|   |   |-- main.py                # FastAPI 入口
|   |   |-- core/
|   |   |   |-- config.py          # Pydantic Settings
|   |   |   |-- logging.py
|   |   |   |-- lifespan.py
|   |   |   `-- deps.py            # 依赖注入
|   |   |-- api/
|   |   |   `-- v1/
|   |   |       |-- knowledge_bases.py
|   |   |       |-- documents.py
|   |   |       |-- chat.py        # SSE 流式
|   |   |       `-- search.py
|   |   |-- models/                # SQLAlchemy ORM
|   |   |-- schemas/               # Pydantic DTO
|   |   |-- services/
|   |   |   |-- ingest.py          # 入库编排
|   |   |   |-- retriever.py       # 混合检索
|   |   |   `-- agent.py           # LangGraph
|   |   |-- loaders/               # 多格式文档解析
|   |   |-- splitters/             # 切片策略
|   |   |-- embeddings/            # Embedding 工厂
|   |   |-- vectorstore/           # Chroma 封装
|   |   |-- rerankers/             # 重排序
|   |   |-- websearch/             # Tavily/DDG
|   |   `-- llm/                   # LLM 工厂
|   `-- tests/
|-- frontend/
|   |-- package.json
|   |-- vite.config.ts
|   |-- index.html
|   |-- src/
|   |   |-- main.ts
|   |   |-- App.vue
|   |   |-- views/
|   |   |   |-- ChatView.vue       # 主问答
|   |   |   |-- DocsView.vue       # 文档管理
|   |   |   |-- KBView.vue         # 知识库管理
|   |   |   `-- SettingsView.vue
|   |   |-- components/
|   |   |   |-- FloatingBall.vue   # 悬浮球
|   |   |   |-- ChatPanel.vue
|   |   |   |-- SourceList.vue
|   |   |   `-- ...
|   |   |-- stores/                # Pinia
|   |   |-- api/                   # ofetch 封装
|   |   `-- utils/
|   `-- src-tauri/
|       |-- Cargo.toml
|       |-- tauri.conf.json
|       `-- src/main.rs            # 全局快捷键 + 悬浮
`-- data/                          # 本地持久化（gitignore）
     |-- chroma/
     `-- metadata.db
```

## 六、核心功能清单

### 6.1 知识库与文档管理
- 知识库（KB）CRUD：创建/重命名/删除/切换
- 多格式文档上传：PDF、DOCX、PPTX、MD、TXT、HTML、CSV、图片（OCR 可选）
- 切片策略：按段落 + 滑窗，重叠 100-200 token
- 元数据保留：来源、页码、切片序号、标题层级
- 入库进度可观测：日志 + 任务状态
- 文档删除级联清理向量

### 6.2 混合检索
- 向量检索：默认 BGE-M3
- 关键词检索：BM25Okapi（中文按字 + jieba 分词）
- 融合：**Reciprocal Rank Fusion (RRF)**
- 重排序：Cross-Encoder BGE-Reranker
- 查询改写：Multi-Query（生成 3 个变体检索后合并）
- 上下文压缩：LLM 抽取相关句子

### 6.3 问答 Agent（LangGraph）
- 意图分类节点：基于 LLM 的 few-shot 路由
- RAG 路径：检索 -> 重排 -> 生成，附带引用
- 联网路径：Tavily 搜索 -> 摘要 -> 生成
- 混合路径：本地 + 联网融合
- 直答路径：纯对话
- 流式输出（SSE）：前端 token 级渲染
- 多轮对话：会话级 memory
- 引用溯源：返回 doc_id/page/chunk_id

### 6.4 桌面浮窗
- 无边框悬浮球，初始 60px，hover 放大
- 全局快捷键：`Cmd/Ctrl+Shift+Space` 唤起/收起
- 展开后：800x600 圆角浮窗，毛玻璃背景
- 系统托盘菜单：显示/隐藏/退出
- 拖拽移动，位置记忆
- 窗口置顶可切换

### 6.5 设置
- LLM 服务商切换（OpenAI / DeepSeek / 自定义 base_url）
- API Key 填写（前端加密缓存）
- 联网搜索开关与 Key
- Embedding 模型选择
- 主题（亮/暗）

## 七、开发阶段

| 阶段 | 内容 | 验收 |
| --- | --- | --- |
| 0. 准备 | 写 agent.md、目录骨架、依赖清单 | 仓库结构可读 |
| 1. 后端骨架 | FastAPI 工程、配置/日志/SQLite、健康检查、API 路由占位 | `/health` 200；CORS OK |
| 2. 前端骨架 | Vue3 + Tauri 浮窗可启动；与后端 `/health` 联通 | 悬浮球可点开窗口 |
| 3. 文档入库 | Loader + Splitter + Embedding + Chroma 分区 + 上传 API + 元数据 | 上传 PDF 可见 chunks |
| 4. 混合检索 | 向量 + BM25 + RRF + Rerank | 单测命中相关切片 |
| 5. LangGraph Agent | classify -> retrieve/web/direct -> generate，SSE | 三路径可走通 |
| 6. 前端业务 | 文档管理 UI、聊天 UI、KB 切换、设置 | 端到端 demo |
| 7. 质量与优化 | 检索评测、错误处理、性能、E2E、Docker | 评测报告 + 通过测试 |

## 八、关键设计决策

- **向量库默认 Chroma**：零依赖启动；后续可一行配置切 Qdrant。
- **Embedding 默认 BGE-M3**：本地、可商用、中文强；首次启动自动下载。
- **LLM 默认 OpenAI 兼容接口：DeepSeek/OpenAI/Claude（via proxy）同协议切换。
- **联网默认 Tavily**：AI 友好摘要；Key 缺失自动降级 DuckDuckGo。
- **Agent 用 LangGraph StateGraph**：节点清晰、可观测、可单步调试。
- **流式用 SSE 而非 WebSocket**：单向 + 简单 + FastAPI 原生支持。
- **前端用 Tauri**：包体小、权限细、快捷键原生。
- **不做账号系统：本地工具，简化复杂度。

## 九、环境变量（.env.example）

```env
# LLM
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

# Embedding（可选：留空则用本地 BGE-M3）
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# 联网搜索
TAVILY_API_KEY=

# 服务
HOST=127.0.0.1
PORT=8765
LOG_LEVEL=INFO

# 持久化
DATA_DIR=./data
CHROMA_PERSIST_DIR=./data/chroma
SQLITE_PATH=./data/metadata.db

# 检索
RERANK_TOP_K=20
FINAL_TOP_K=5
```

## 十、执行待办（首个 Sprint）

- [x] 初始化 backend pyproject + 依赖（fastapi/langchain/langgraph/chroma/flagembedding 等）
- [x] 初始化 frontend Vue3 + Tauri 2.x 骨架
- [x] 搭 FastAPI 路由占位与 `/health`
- [x] 写 `loader/splitter/embedding/vectorstore` 抽象
- [x] 实现 PDF/DOCX/MD/TXT 上传与入库
- [x] 实现 BM25 + 向量混合检索 + Rerank
- [x] 实现 LangGraph classify -> RAG/Web/Direct
- [x] 前端浮窗 + 聊天 + 文档管理 + 设置
- [x] 端到端 demo 与 README（浏览器模式）；Tauri 桌面壳后续补

### 当前会话增量（2026-07-01）

- 修复 `app/loaders/factory.py` 的循环引用 bug（导入了不存在的 `csv_loader` / `html_loader`，实际类都在 `markdown_loader.py`）
- 补齐 4 个 API 路由：`api/v1/{knowledge_bases,documents,search,chat}.py`；chat 包含 SSE 与对话/消息持久化
- 新增 `app/core/deps.py`：FastAPI Depends 注入 `AsyncSession`
- 前端 Pinia stores：`settings`（主题/后端地址/KB 选择/持久化） / `knowledgeBase` / `chat`（SSE 解析）
- 前端组件：`FloatingBall` / `ChatPanel` / `MessageBubble` / `KnowledgePicker`
- 前端视图：`ChatView` / `KBView`（KB CRUD + 统计） / `SettingsView`（含 /health 探测）
- 前端样式：`styles/main.css`（主题变量 + Markdown 样式 + 滚动条 + 引用 chip）
- 前端配置：`vite.config.ts`（alias + /api 反代 + AutoImport + Components 按需）、`tsconfig.json`（`@/*` 别名）、`index.html`
- 编写根目录 `README.md`（架构 / 快速开始 / API 约定 / SSE 事件示例 / 环境变量 / 风险回退）

> 本轮未实际安装 backend venv 与前端 `node_modules`；运行性验证交由后续手动 `pip install -e .` 与 `npm install` 后执行 `pytest` / `npm run typecheck` / `npm run dev`。

## 十一、风险与回退

| 风险 | 回退方案 |
| --- | --- |
| BGE 本地模型下载慢 | 切换 OpenAI Embedding |
| Tauri 编译失败（Rust 工具链） | 临时用 Vite dev + 系统浏览器跑通业务，后续补 Tauri |
| Tavily 付费/Key 缺失 | 自动用 DuckDuckGo |
| LangGraph 与 LangChain 版本冲突 | 锁定版本（langchain>=0.3, langgraph>=0.2） |
| macOS 全局快捷键权限 | 提示用户在系统设置授权 |
