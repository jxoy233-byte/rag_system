export interface KnowledgeBase {
  id: number
  name: string
  slug: string
  description: string | null
  collection_name: string
  embedding_model: string | null
  embedding_dim: number | null
  chunk_size: number
  chunk_overlap: number
  doc_count: number
  chunk_count: number
  created_at: string
  updated_at: string
}

export type DocumentStatus = 'pending' | 'processing' | 'ready' | 'failed'

export interface Document {
  id: number
  knowledge_base_id: number
  title: string
  filename: string
  file_ext: string
  file_size: number
  status: DocumentStatus
  error: string | null
  chunk_count: number
  // parent chunk 数（父子切片下的"父块"数）。child 数 = chunk_count。
  parent_count?: number
  // 入库时由 LLM 生成的文档摘要（~150-300 字）。
  // 文档列表行下方展示，方便快速浏览每份文档覆盖什么内容。
  summary?: string
  created_at: string
  updated_at: string
}

export interface Source {
  kb_id?: number | null
  document: string
  page?: number | null
  chunk_id?: string | null
  snippet: string
  score?: number | null
  rerank_score?: number | null
  doc_id?: number | null
  source_type: 'vector' | 'bm25' | 'web'
  url?: string | null
}

// doc-level 命中：BM25(title+filename+summary) 出来的 top-K 文档。
// 前端"相关文档"区显示这些，每条带 summary 摘要，让用户看到
// 「系统判断这几份文档相关，理由是它们覆盖了 X」。
export interface DocHit {
  doc_id: number
  title: string
  filename: string
  summary: string
  score?: number | null
}

export interface ChatMeta {
  intent: string
  latency_ms: number
  used_web: boolean
  used_rag: boolean
  conversation_id: number | null
  message_id: number | null
  // 信心闸门：检索+web 都失败 / 全不相关 时为 true。
  // 前端据此把消息渲染成「暂未收录」柔和提示（区别于 error 的红色警示）。
  refused?: boolean
}

export interface ChatMessage {
  id?: number
  role: 'user' | 'assistant' | 'system'
  content: string
  sources?: Source[]
  doc_hits?: DocHit[]
  intent?: string
  latency_ms?: number
  thinking?: boolean
  error?: string
  refused?: boolean
}
