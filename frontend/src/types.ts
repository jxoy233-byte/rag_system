export interface KnowledgeBase {
  id: number
  name: string
  slug: string
  description: string | null
  collection_name: string
  embedding_model: string
  embedding_dim: number
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

export interface ChatMeta {
  intent: string
  latency_ms: number
  used_web: boolean
  used_rag: boolean
  conversation_id: number | null
  message_id: number | null
}

export interface ChatMessage {
  id?: number
  role: 'user' | 'assistant' | 'system'
  content: string
  sources?: Source[]
  intent?: string
  latency_ms?: number
  thinking?: boolean
  error?: string
}
