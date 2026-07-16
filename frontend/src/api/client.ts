import { ofetch } from 'ofetch'

const env = (import.meta as ImportMeta & { env: Record<string, string | undefined> }).env
const defaultBaseURL: string = env.VITE_API_BASE || 'http://127.0.0.1:8765'
const settingsKey = 'rag.settings'

export function getApiBaseURL(): string {
  if (typeof window === 'undefined') return defaultBaseURL
  try {
    const saved = JSON.parse(localStorage.getItem(settingsKey) || '{}') as { apiBase?: string }
    return saved.apiBase?.trim().replace(/\/$/, '') || defaultBaseURL
  } catch {
    return defaultBaseURL
  }
}

export function api<T = unknown>(request: string, options: Record<string, any> = {}) {
  return ofetch<T>(request, {
    baseURL: getApiBaseURL(),
    timeout: 120000,
    retry: 1,
    ...options,
  })
}

export const kbApi = {
  list: () => api<unknown[]>('/api/v1/knowledge-bases'),
  create: (body: unknown) => api('/api/v1/knowledge-bases', { method: 'POST', body: body as any }),
  get: (id: number) => api(`/api/v1/knowledge-bases/${id}`),
  update: (id: number, body: unknown) => api(`/api/v1/knowledge-bases/${id}`, { method: 'PATCH', body: body as any }),
  remove: (id: number) => api(`/api/v1/knowledge-bases/${id}`, { method: 'DELETE' }),
  stats: (id: number) => api(`/api/v1/knowledge-bases/${id}/stats`),
}

export const docApi = {
  list: (kbId: number, limit = 100, offset = 0) =>
    api(`/api/v1/knowledge-bases/${kbId}/documents?limit=${limit}&offset=${offset}`),
  upload: (kbId: number, file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    return api(`/api/v1/knowledge-bases/${kbId}/documents`, { method: 'POST', body: fd })
  },
  uploadBatch: (kbId: number, files: File[]) => {
    const fd = new FormData()
    files.forEach((f) => fd.append('files', f))
    return api(`/api/v1/knowledge-bases/${kbId}/documents/batch`, { method: 'POST', body: fd })
  },
  remove: (kbId: number, docId: number) =>
    api(`/api/v1/knowledge-bases/${kbId}/documents/${docId}`, { method: 'DELETE' }),
  retry: (kbId: number, docId: number) =>
    api(`/api/v1/knowledge-bases/${kbId}/documents/${docId}/retry`, { method: 'POST' }),
}

export const searchApi = {
  search: (body: { query: string; knowledge_base_id?: number; top_k?: number; use_rerank?: boolean }) =>
    api('/api/v1/search', { method: 'POST', body }),
}

export interface ChatEvent {
  event: string
  data: string
}

export async function* streamChat(
  body: {
    question: string
    knowledge_base_id?: number | null
    conversation_id?: number | null
    history?: { role: string; content: string }[]
    enable_web?: boolean
  },
): AsyncGenerator<ChatEvent> {
  const res = await fetch(getApiBaseURL() + '/api/v1/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok || !res.body) {
    throw new Error('HTTP ' + res.status)
  }
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    // 归一化 CRLF -> LF：sse_starlette 按 SSE 规范用 \r\n\r\n 分隔事件，
    // 但很多前端实现只识别 \n\n。统一在收包时把 \r\n 转成 \n，避免漏解析。
    buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n')
    let idx
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const block = buf.slice(0, idx)
      buf = buf.slice(idx + 2)
      let event = 'message'
      let data = ''
      for (const line of block.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) data += line.slice(5).trim()
      }
      if (data) yield { event, data }
    }
  }
}

export const convApi = {
  list: (kbId?: number | null) => {
    const q = kbId ? '?kb_id=' + kbId : ''
    return api('/api/v1/chat/conversations' + q)
  },
  messages: (id: number) => api('/api/v1/chat/conversations/' + id + '/messages'),
  update: (id: number, body: { title: string }) =>
    api('/api/v1/chat/conversations/' + id, { method: 'PATCH', body }),
  remove: (id: number) => api('/api/v1/chat/conversations/' + id, { method: 'DELETE' }),
}
