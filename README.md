# 通用文档问答 RAG 系统

基于 Vue 3 + Tauri 桌面浮窗的通用 RAG 系统，支持：

- **本地混合检索**（向量 + BM25 + RRF + BGE-Reranker）
- **联网搜索**（Tavily 优先，DuckDuckGo 兜底）
- **基础对话**（无外部知识）
- **图片理解**（VLM 描述后并入文本切片）
- 多轮会话、引用溯源、SSE 流式输出

> 设计与阶段划分见 [`agent.md`](./agent.md)。

## 目录

- [〇、产品结构 & 决策流程图（whiteboards）](#〇产品结构--决策流程图whiteboards)
- 一、技术栈与结构
- 二、快速开始
- 三、API 约定
- 四、SSE 事件协议
- 五、引用 chip & 切片预览
- 六、环境变量
- 七、开发与调试
- 八、已知风险与回退
- 九、检索信心闸门 & KB→Web 自动兜底
- 十、检索评估（ablation & 调优记录）

---

## 〇、产品结构 & 决策流程图（whiteboards）

架构图与决策流程图集中在 `.whiteboards/`（SVG 源文件，可直接拖入飞书文档）：

| 图 | 内容 | 看点 |
| --- | --- | --- |
| [`01_product_structure.svg`](./.whiteboards/01_product_structure.svg) | 5 层产品结构：用户/Tauri → 前端 → FastAPI → 服务层 → 模型与存储 | RAGAgent 块标注了 KB→Web 兜底 + confidence gate 两个关键能力 |
| [`02_decision_flow.svg`](./.whiteboards/02_decision_flow.svg) | LangGraph 完整决策流 | 9 个节点 + 4 个条件路由：classify → query_expand → retrieve / web → relevance_check → build_prompt → generate；含 KB→Web 兜底与 confidence gate 短路 |
| [`03_core_logic.svg`](./.whiteboards/03_core_logic.svg) | 端到端数据流：写入轨 → 存储层 → 检索放大 → 生成轨 → 不变量 | 重点看 DocIndex soft boost、relevance_check、confidence gate 短路 |

> **飞书导入**：把对应 `.svg` 文件直接拖到飞书文档即可，矢量缩放无损。
> **本地转 PNG**（如需嵌 README）：`brew install cairo && pip install cairosvg`，然后 `python -c "import cairosvg; cairosvg.svg2png(url='01_product_structure.svg', write_to='01.png', output_width=2400)"`。

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
|   |   |-- loaders/         # PDF / DOCX / PPTX / MD / TXT / HTML / CSV / XLSX + 旧版 .doc/.ppt/.xls
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
| `IngestService` | `services/ingest.py` | 文档入库主流程：解析 → 图片描述 → 切片 → 向量化 → BM25 + Chroma + DocIndex 写入 |
| `HybridRetriever` | `services/retriever.py` | 向量 + BM25 并行召回 → RRF 融合 → BGE-Reranker；可选 parent_collapse 合并子切片 |
| `RAGAgent` | `services/agent.py` | LangGraph 状态机：classify → query_expand → retrieve / web_search → relevance_check → build_prompt → generate；含 KB→Web 兜底与 confidence gate |
| `DocIndex` | `services/doc_index.py` | doc-level 索引（BM25 on title+filename+summary），用于 chunk soft boost 与"相关文档"展示 |
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
| /api/v1/knowledge-bases/{id}/documents/{doc_id}/chunks | GET | **列出该文档的切片**（`ChunkListItem[]`，按 `chunk_index` 升序；用于「切片预览」抽屉） |
| /api/v1/knowledge-bases/{id}/documents/{doc_id}/chunks/{chunk_id} | GET | **按 chunk_id 查详情**（`ChunkDetail`，含完整 text + 元数据；用于「引用 chip」抽屉） |
| /api/v1/knowledge-bases/{id}/documents/{doc_id}/content | GET | **该文档的解析后全文**（`DocumentContent`，含 `segments[]` 与 `full_text`；用于「原文」视图） |
| /api/v1/search | POST | 直接调用混合检索，返回 sources + latency |
| /api/v1/chat | POST | **SSE 流式问答**：intent → sources → token* → meta → final → end（可能发 error） |
| /api/v1/chat/sync | POST | 非流式收敛（便于测试） |
| /api/v1/chat/conversations | GET (?kb_id=) | 会话列表（可按 KB 过滤） |
| /api/v1/chat/conversations/{id}/messages | GET | 消息历史 |
| /api/v1/chat/conversations/{id} | PATCH / DELETE | 改标题 / 删除会话 |

### 通用约束

- 上传文件类型：

  | 类别 | 扩展名 | 解析方式 |
  | --- | --- | --- |
  | 现代 Office | `.docx` `.pptx` `.xlsx` | `python-docx` / `python-pptx` / `openpyxl` |
  | **旧版 Office** | `.doc` `.ppt` `.xls` | `.doc` → `antiword`；`.ppt` → `catppt`（catdoc）；`.xls` → `xlrd>=2.0.1` |
  | 文本 | `.md` `.txt` `.log` `.html` `.htm` `.csv` | 内置纯 Python 解析 |
  | 文档 | `.pdf` | `pypdf` + `pdfplumber` |

  > **旧版 Office 前置依赖**：
  > - macOS：`brew install antiword catdoc`（antiword 解析 .doc，catdoc 含 catppt 解析 .ppt）
  > - Debian/Ubuntu：`apt-get install antiword catdoc`
  > - 若不安装，`.doc` / `.ppt` 上传会成功但后台入库失败，`doc.error` 含 `brew install antiword` 提示，装好后调 `/retry` 即可
  > - `.xls` 依赖 `pyproject.toml` 中的 `xlrd>=2.0.1,<3.0`，随 backend 一起装
- 上传大小上限：`MAX_UPLOAD_MB`（默认 50 MB）
- 上传大文件：解析与切片走 `asyncio.to_thread`，不阻塞事件循环
- 流式问答客户端断开：`CancelledError` 会被记录但不视为错误

## 四、SSE 事件协议

`/api/v1/chat` 使用 `text/event-stream`，事件类型如下：

| event | data 字段 | 触发时机 |
| --- | --- | --- |
| `intent` | `{"intent":"rag\|web\|direct\|hybrid"}` | 意图分类完成 |
| `sources` | `{"sources":[SourceItem, ...]}` | chunk-level 检索/联网结束 |
| `doc_hits` | `{"doc_hits":[DocHitItem, ...]}` | doc-level 命中（BM25 title+filename+summary top-K），前端「相关文档」区 |
| `token` | `{"content":"..."}` | LLM 流式增量（多次） |
| `meta` | `{latency_ms, intent, used_web, used_rag, refused, answer}` | 流结束前；`refused=true` 表示 confidence gate 触发（KB+web 都空），跳过 LLM 直接发固定回复 |
| `final` | `ChatFinalEvent` JSON | 持久化成功后：含 `meta.conversation_id / message_id / refused` 和 sources |
| `error` | `{"message":"..."}` | 任一阶段失败（其后必发 `end`） |
| `end` | `{}` | 会话结束（无论成功失败） |

示例：

```
event: intent
data: {"intent":"rag"}

event: sources
data: {"sources":[{"document":"x.pdf","page":3,"score":0.83,"source_type":"vector"}]}

event: doc_hits
data: {"doc_hits":[{"doc_id":7,"title":"性能优化手册","filename":"perf.pdf","summary":"..."}]}

event: token
data: {"content":"根据"}

event: meta
data: {"intent":"rag","latency_ms":1823,"used_web":false,"used_rag":true,"refused":false}

event: final
data: {"meta":{"intent":"rag","conversation_id":42,"message_id":99,"refused":false},"sources":[...]}

event: end
data: {}
```

前端解析见 `frontend/src/api/client.ts` 的 `streamChat`：按 `\n\n` 切事件，并把 `\r\n` 归一化为 `\n`，避免漏解析。

## 五、引用 chip & 切片预览

### 引用 chip → chunk 详情

聊天答案中 LLM 输出的 `[n]` 标记会被 markdown-it 转成**可点击的引用 chip**（见 `MessageBubble.vue` 的 inline rule）。
点击后在右侧滑出 `AppDrawer`，展示该 chunk 的**完整原文**（不限长）、`page / section / score / rerank_score`，以及「跳转到文档」按钮。

```
事件：         click .cite-chip[data-cite-idx="N"]
前端：         MessageBubble.onMdClick  → emit('openChunk', { kbId, docId, chunkId, source })
              ChatPanel.openChunk      → chunkApi.getDetail(kbId, docId, chunkId)
后端：         GET /api/v1/knowledge-bases/{id}/documents/{doc_id}/chunks/{chunk_id}
              → ChunkDetail { chunk_id, doc_id, kb_id, text, page, section, score, ... }
```

边界：

- 引用 chip **任意来源都点击有反馈**：
  - 有 `chunk_id` 的 RAG 来源 → 拉 `ChunkDetail` 完整原文
  - BM25 / Web 来源 → 用 `source.snippet` 拼一个 `ChunkDetail` 兜底（web 含 `url` 时抽屉底栏额外出现「打开外链」按钮）
  - 历史会话重新打开时也有效——`GET /chat/conversations/{id}/messages` 会从 `Message.sources_json` 回填 `sources[]`
- 抽屉宽 480px；底栏提供「跳转到文档」按钮（`router.push('/docs/'+kbId+'?doc='+docId)`，与 `?doc=` 高亮机制联动）

### 切片预览 / 原文视图（按文档行点击）

`DocsView` 的每个文档行**整行可点**；点击后右侧抽屉可在两种视图间切换：

| 视图 | 来源 | 用途 |
| --- | --- | --- |
| **切分** | `GET .../chunks` | 看每个切片的序号 / 页码 / 章节 / 长度 / 200 字预览 |
| **原文** | `GET .../content` | 把全部切片按 `chunk_index` 顺序拼成连续全文（与「切分」边界对照，看解析/切分效果） |

```
切分源数据：   ChunkListItem[] { chunk_id, length, preview[:200], page, section, chunk_index, ... }
原文源数据：   DocumentContent { doc_id, kb_id, title, segments[], full_text }
              full_text = 各段 text 以 \n\n 拼接（上限 = 该文档所有切片）
```

- 「原文」懒加载：首次切到 tab 时才请求，断网/失败时显示错误状态；切回文档重置
- 处理中 / 失败 / 解析失败时 `chunks` 为空数组 → 切分视图显示「暂无切片」；原文视图显示「该文档暂无解析内容」
- 抽取 `chunk_index` 入库时由 `IngestService._ingest_chunks` 写入 metadata；缺则按 Chroma 返回顺序展示

### 共享抽屉组件

两个用例共用 `frontend/src/components/AppDrawer.vue`：naive-ui `n-drawer` 的薄封装，统一宽度（480/520）、主题色、关闭行为、footer 槽位。

## 六、环境变量

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
| `EMBEDDING_PROVIDER` | `local` | `local`（bge-base-zh-v1.5） / `openai` / `mock` |
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-base-zh-v1.5` | `local` 时生效 |
| `LOCAL_EMBEDDING_DEVICE` | `cpu` | `cpu` / `cuda` |
| `LOCAL_EMBEDDING_DIM` | `768` | 与 `bge-base-zh-v1.5` 输出维度一致 |
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

提示词 `IMAGE_DESCRIBE_PROMPT`（`services/image_describer.py`）兼顾 RAG 可检索性 + 适度长度（200-300 字）。要求覆盖：

1. 图像类型（照片/流程图/示意图/图表/表格/截图/公式）
2. **图中可见文字逐条抄录（OCR）** ← 关键信息
3. 图表/表格的**坐标轴 / 图例 / 关键数据点 / 趋势** ← 关键信息
4. 主要对象 / 实体及其布局
5. 该图在文档里的语义作用

生成上限：OpenAI `max_tokens=768`、LocalVLM `max_new_tokens=512`。图片最长边 512px（VLM 解析速度与 OCR 清晰度平衡；想更清晰可手动上调 `max_pixels`）。

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
| `ENABLE_MULTI_QUERY` | `true` | multi-query 改写：把 1 个问题扩展成 3 个等价版本并行检索，提高召回率 |
| `ENABLE_HYDE` | `false` | HyDE 开关（保留位） |
| `ENABLE_RELEVANCE_CHECK` | `true` | Self-RAG：LLM 给 top-10 chunk 判 relevant=true/false，过滤无关 chunk；全不相关时清空并触发 KB→Web 兜底 |
| `ENABLE_DOC_INDEX` | `true` | doc-level 预筛选：BM25(title+filename+summary) 找 top-K 相关文档，对这些文档里的 chunk 在 rerank 分数上 ×1.2 soft boost |
| `DOC_INDEX_TOP_K` | `3` | doc-level 召回条数 |
| `ENABLE_KB_WEB_FALLBACK` | `true` | KB→Web 自动兜底开关：KB 检索为空（或 relevance 全不相关）时自动触发 web search；web 也空 → confidence gate 直接拒绝 |

### HuggingFace / 本地模型

| 变量 | 默认 | 用途 |
| --- | --- | --- |
| `HF_ENDPOINT` | - | 镜像站（如 `https://hf-mirror.com`），加速 BGE / VLM 下载 |
| `LOCAL_MODEL_ROOT` | `./.model` | 本地模型副本根目录 |
| `LOCAL_EMBEDDING_PATH` | - | 显式覆盖本地 embedding 路径 |
| `LOCAL_RERANK_PATH` | - | 显式覆盖本地 reranker 路径 |

启动时（`lifespan`）会预热 Reranker，避免首次对话等待下载；预热失败仅记 warning，可懒加载重试。

## 七、开发与调试

### 后端

```bash
cd backend
pytest -q                                # test_loaders / test_splitters / test_bm25 / test_api
ruff check app                           # 静态检查
uvicorn app.main:app --reload --port 8765
```

测试覆盖：

- `test_loaders.py` — PDF/DOCX/MD/TXT/CSV/HTML/XLSX Loader 解析
- `test_legacy_loaders.py` — 旧版 Office Loader：扩展名注册、antiword/catppt 缺失提示、xlrd 缺失提示
- `test_chunk_endpoints.py` — `ChromaStore.get()` + ChunkDetail/ChunkListItem schema + 路由注册
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

## 八、已知风险与回退

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
| `.doc` / `.ppt` 上传后 `doc.status=failed` | 缺 `antiword` / `catppt`；按错误提示安装后调 `/retry`。`.xls` 失败一般因 `xlrd` 未装，pip install 即可 |

## 九、检索信心闸门 & KB→Web 自动兜底

**问题**：用户问 KB 完全没有覆盖的问题时，旧版会拼 `(no local results)` 强行让 LLM 答，导致幻觉。rerank 引入后还把无关 chunk 推上去进一步污染。

**方案**（对应 [02_decision_flow.svg](./.whiteboards/02_decision_flow.svg)）：

```
classify ─► query_expand ─► retrieve ─┬─► chunks 非空 ─► relevance_check ─► build_prompt
                                     │                                │
                                     └─► chunks 为空 ─► web_search (KB→Web 兜底)
                                                              │
                                                              ▼
                                                      build_prompt
                                                              │
                                                       ┌──────┴──────┐
                                                       │ KB+web 都空？ │
                                                       └──────┬──────┘
                                                       是 │   │ 否
                                                          ▼   ▼
                                              refused=True   按意图选 prompt
                                              跳过 LLM           │
                                              发固定回复          ▼
                                                  │         generate
                                                  ▼             │
                                                END  ◄─────────┘

另：intent=web ─► web_search ─► build_prompt ─► generate ─► END
    intent=direct ─► direct_answer ─► END（不受闸门影响）
    hybrid：retrieve + web_search 并行（KB+web 双路，不是兜底）
```

### 决策矩阵

| intent | KB chunks | KB 全不相关（relevance） | web | 行为 |
| --- | --- | --- | --- | --- |
| `rag` | 0 | – | 有结果 | 走 ANSWER_WEB_PROMPT，sources 全是 web |
| `rag` | 0 | – | 0 | **refused**：发「知识库暂未收录」+ `meta.refused=true` |
| `rag` | N | 是 | 有结果 | 走 ANSWER_WEB_PROMPT |
| `rag` | N | 是 | 0 | **refused** |
| `rag` | N | 否 | – | ANSWER_WITH_CONTEXT_PROMPT + LLM |
| `web` | 跳过 | – | – | 直接 web_search |
| `hybrid` | 0 | – | 有结果 | ANSWER_WEB_PROMPT |
| `hybrid` | 0 | – | 0 | **refused** |
| `hybrid` | N | 是 | 有结果 | ANSWER_HYBRID_PROMPT（chunks 被 relevance 清空，但 web 还在） |
| `hybrid` | N | 否 | – | ANSWER_HYBRID_PROMPT（KB+web 一起用） |
| `direct` | 跳过 | – | – | ANSWER_DIRECT_PROMPT，不受闸门影响 |

### 关键代码位置

| 行为 | 文件:行 |
| --- | --- |
| AgentState 加 `refused: bool` | `backend/app/services/agent.py:54` |
| `ANSWER_REFUSE_PROMPT` | `backend/app/services/agent.py:143-156` |
| `_route_after_retrieve`（KB→Web 兜底路由） | `agent.py:339-351` |
| `_route_after_relevance_check`（relevance 清空后再兜底） | `agent.py:362-375` |
| `_build_prompt_node` 信心闸门分支 | `agent.py:711-730` |
| `astream` 跳过 LLM 调用 | `agent.py:851-859` |
| `ChatMeta.refused` schema | `backend/app/schemas/chat.py:46` |
| `ChatMessage.refused` 前端类型 | `frontend/src/types.ts:84` |
| `MessageBubble` 柔和渲染分支 | `frontend/src/components/MessageBubble.vue:146-149` |
| 配置开关 `enable_kb_web_fallback` | `backend/app/core/config.py:111` |
| 测试 | `backend/tests/test_agent.py`（7 个 case 覆盖所有分支） |

### 设计要点

1. **KB→Web 单向兜底**：web→KB 无意义（web 永远有结果），不引入 web_fallback 反向路径
2. **prompt 跟实际上下文一致**：rag intent 但 chunks 为空、web 有结果时，临时改用 ANSWER_WEB_PROMPT，LLM 不会困惑
3. **relevance_check 全不相关 → 触发兜底**：旧版保留 top-3 让 LLM 强行答；新版清空 chunks 后让路由决定兜底或拒绝
4. **hybrid 不触发 fallback 死循环**：hybrid 一开始就跑过 web_search，relevance_check 后若 chunks 空也只走 `to_prompt`（拒绝），不会再跳回 web
5. **refused 透传到前端**：sse `meta.refused` + `ChatFinalEvent.meta.refused`；前端 `MessageBubble` 用柔和灰色（区别于红色 error）渲染「暂未收录」

### 关掉兜底 = 直接拒绝

```bash
# .env 中关闭 KB→Web 自动兜底
ENABLE_KB_WEB_FALLBACK=false
```

效果：KB 空时直接 `refused=True`，不消耗 web 配额；用户拿到的就是「知识库暂未收录相关内容」。

## 十、检索评估（ablation & 调优记录）

评估脚本 `backend/scripts/eval/run_eval.py` 在 `backend/scripts/eval/data/eval_set.jsonl` 上跑 7 种配置，记录 recall@k / mrr@k / hit_doc@k。

```bash
cd backend && .venv/bin/python -m scripts.eval.run_eval --top-k 10
```

输出到 `scripts/eval/data/eval_set.report.md`，**每次调检索参数前先跑一次，作为对照基线**。

### 已知调优结论

| 调优 | 结论 |
| --- | --- |
| 启用 `rerank` | 召回不变（rerank 不改 set），mrr 反而降 5-7pp：rerank 把"半相关"的错配 chunk 推到 top，挤掉正确答案 → LLM 拿到错误上下文幻觉 |
| 启用 `parent_collapse` | 召回基本不变；让 LLM 看到完整 parent 上下文，缓解"片段孤岛"，但本身不涨 mrr |
| 启用 `multi_query` | 召回微涨（多版本覆盖一些边角）；mrr 持平；CPU 上多路+rerank 慢 10×，收益不抵成本 |
| 启用 `ENABLE_RELEVANCE_CHECK` | 不影响召回；降低"半相关"chunk 进入 LLM 的概率，间接减少幻觉；全不相关时触发 KB→Web 兜底 |
| 启用 `ENABLE_DOC_INDEX` | 给来自相关文档的 chunk ×1.2 soft boost；微涨 doc-level 命中率 |

> **当前结论**（最近一次跑，36 题 7 配置）：`vec_only` 在合成集上 mrr 仍领先 `full` 7pp，rerank 不是越多越好。但 `full` 是 production 默认（提供 doc-level soft boost + relevance 过滤 + KB→Web 兜底），幻觉率更低，体验更好。
> 详见 `backend/scripts/eval/data/eval_set.report.md`。
