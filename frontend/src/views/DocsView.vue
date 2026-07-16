<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount, computed, watch, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { Upload, FileText, Trash2, RefreshCw, Search, ArrowLeft, AlertCircle, CheckCircle2, Clock, X, RotateCcw } from 'lucide-vue-next'
import { docApi, kbApi } from '@/api/client'
import { isTauri } from '@/utils'
import { invoke } from '@tauri-apps/api/core'
import { getCurrentWebviewWindow } from '@tauri-apps/api/webviewWindow'
import type { UnlistenFn } from '@tauri-apps/api/event'
import type { Document, KnowledgeBase } from '@/types'
import { formatBytes, formatDate } from '@/utils'

const route = useRoute()
const router = useRouter()
const kbId = computed(() => Number(route.params.kbId))
const highlightDocId = computed(() => {
  const v = route.query.doc
  return v ? Number(v) : null
})

function backToChat() {
  router.push('/')
}

function backToKb() {
  router.push('/kb')
}

const docs = ref<Document[]>([])
const kb = ref<KnowledgeBase | null>(null)
const loading = ref(false)
const uploading = ref(false)
const error = ref<string | null>(null)
const search = ref('')
const dragOver = ref(false)
const rowRefs = new Map<number, HTMLElement>()
let fetching = false
let pollTimer: number | null = null
let unlistenDragDrop: UnlistenFn | null = null

function setRowRef(id: number, el: HTMLElement | null) {
  if (el) rowRefs.set(id, el)
  else rowRefs.delete(id)
}

async function scrollToHighlighted(id: number | null) {
  if (id == null) return
  await nextTick()
  const el = rowRefs.get(id)
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' })
}

watch(highlightDocId, scrollToHighlighted, { immediate: true })

const filtered = computed(() => {
  const q = search.value.trim().toLowerCase()
  if (!q) return docs.value
  return docs.value.filter((d) => d.title.toLowerCase().includes(q) || d.filename.toLowerCase().includes(q))
})

async function fetchDocs(showLoading = true) {
  if (!kbId.value || fetching) return
  fetching = true
  if (showLoading) loading.value = true
  error.value = null
  try {
    const res = (await docApi.list(kbId.value, 500)) as { items: Document[]; total: number }
    docs.value = res.items
    await scrollToHighlighted(highlightDocId.value)
  } catch (e: any) {
    error.value = e?.data?.detail || e?.message || String(e)
  } finally {
    loading.value = false
    fetching = false
  }
}

async function upload(files: FileList | null) {
  if (!files || files.length === 0 || !kbId.value) return
  uploading.value = true
  error.value = null
  try {
    let uploadError: string | null = null
    if (files.length === 1) {
      await docApi.upload(kbId.value, files[0])
    } else {
      const arr = Array.from(files)
      const result = (await docApi.uploadBatch(kbId.value, arr)) as {
        failed?: { filename?: string; error: string }[]
      }
      if (result.failed?.length) {
        uploadError = result.failed
          .map((item) => `${item.filename || '文件'}: ${item.error}`)
          .join('\n')
      }
    }
    await fetchDocs(false)
    error.value = uploadError
  } catch (e: any) {
    error.value = e?.data?.detail || e?.message || String(e)
  } finally {
    uploading.value = false
  }
}

// 从文件路径上传（Rust 端文件拖拽事件）
async function uploadFromPaths(paths: string[]) {
  if (!paths || paths.length === 0 || !kbId.value) return
  uploading.value = true
  error.value = null
  try {
    // 读取文件内容并构建 File 对象
    const files: File[] = []
    for (const path of paths) {
      try {
        // 使用 Tauri 文件系统 API 读取文件
        const { readFile } = await import('@tauri-apps/plugin-fs')
        const contents = await readFile(path)
        const filename = path.split('/').pop() || 'unknown'
        const mime = getMimeType(filename)
        const file = new File([contents], filename, { type: mime })
        files.push(file)
      } catch (e) {
        console.error(`Failed to read file ${path}:`, e)
      }
    }

    if (files.length === 0) return

    // 调用上传 API
    let uploadError: string | null = null
    if (files.length === 1) {
      await docApi.upload(kbId.value, files[0])
    } else {
      const result = (await docApi.uploadBatch(kbId.value, files)) as {
        failed?: { filename?: string; error: string }[]
      }
      if (result.failed?.length) {
        uploadError = result.failed
          .map((item) => `${item.filename || '文件'}: ${item.error}`)
          .join('\n')
      }
    }
    await fetchDocs(false)
    error.value = uploadError
  } catch (e: any) {
    error.value = e?.data?.detail || e?.message || String(e)
  } finally {
    uploading.value = false
  }
}

function getMimeType(filename: string): string {
  const ext = filename.split('.').pop()?.toLowerCase() || ''
  const mimeMap: Record<string, string> = {
    pdf: 'application/pdf',
    docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    pptx: 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    xlsx: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    md: 'text/markdown',
    txt: 'text/plain',
    html: 'text/html',
    htm: 'text/html',
    csv: 'text/csv',
  }
  return mimeMap[ext] || 'application/octet-stream'
}

function onFileInput(e: Event) {
  const input = e.target as HTMLInputElement
  upload(input.files)
  input.value = ''
}

const confirmingDeleteId = ref<number | null>(null)
let confirmDeleteTimer: ReturnType<typeof setTimeout> | null = null

function cancelDeleteConfirm() {
  if (confirmDeleteTimer) {
    clearTimeout(confirmDeleteTimer)
    confirmDeleteTimer = null
  }
  confirmingDeleteId.value = null
}

async function removeDoc(doc: Document) {
  if (confirmingDeleteId.value !== doc.id) {
    cancelDeleteConfirm()
    confirmingDeleteId.value = doc.id
    confirmDeleteTimer = setTimeout(cancelDeleteConfirm, 3000)
    return
  }
  cancelDeleteConfirm()
  error.value = null
  try {
    await docApi.remove(kbId.value, doc.id)
    docs.value = docs.value.filter((d) => d.id !== doc.id)
  } catch (e: any) {
    error.value = e?.data?.detail || e?.message || String(e)
  }
}

const retrying = ref<number | null>(null)
async function retryDoc(doc: Document) {
  if (retrying.value !== null) return
  retrying.value = doc.id
  try {
    const updated = (await docApi.retry(kbId.value, doc.id)) as Document
    docs.value = docs.value.map((d) => (d.id === updated.id ? updated : d))
  } catch (e: any) {
    error.value = e?.data?.detail || e?.message || String(e)
  } finally {
    retrying.value = null
  }
}

function onDrop(e: DragEvent) {
  e.preventDefault()
  dragOver.value = false
  if (e.dataTransfer?.files) upload(e.dataTransfer.files)
}

function statusIcon(s: string) {
  if (s === 'ready') return CheckCircle2
  if (s === 'failed') return AlertCircle
  return Clock
}
function statusColor(s: string) {
  if (s === 'ready') return 'success'
  if (s === 'failed') return 'danger'
  return 'pending'
}

onMounted(async () => {
  // 提升窗口层级 + 监听文件拖拽事件
  if (isTauri()) {
    try {
      await invoke('set_window_on_top', { onTop: true })

      // 使用 Tauri 原生文件拖拽 API
      const webview = getCurrentWebviewWindow()
      unlistenDragDrop = await webview.onDragDropEvent((event) => {
        if (event.payload.type === 'over') {
          dragOver.value = true
        } else if (event.payload.type === 'drop') {
          dragOver.value = false
          const paths = event.payload.paths
          if (paths && paths.length > 0) {
            uploadFromPaths(paths)
          }
        } else if (event.payload.type === 'leave') {
          dragOver.value = false
        }
      })
    } catch (e) {
      console.warn('set_window_on_top / onDragDropEvent failed:', e)
    }
  }

  try {
    kb.value = (await kbApi.get(kbId.value)) as KnowledgeBase
  } catch (e: any) {
    error.value = e?.data?.detail || e?.message || String(e)
  }
  await fetchDocs()
  pollTimer = window.setInterval(() => {
    if (docs.value.some((doc) => doc.status === 'processing')) {
      fetchDocs(false)
    }
  }, 3000)
})

onBeforeUnmount(() => {
  // 恢复窗口层级 + 清理监听
  if (isTauri()) {
    invoke('set_window_on_top', { onTop: false }).catch((e) =>
      console.warn('set_window_on_top(false) failed:', e)
    )
    unlistenDragDrop?.()
  }
  if (pollTimer !== null) window.clearInterval(pollTimer)
})
</script>

<template>
  <div class="docs-view">
    <header>
      <button class="back" @click="backToKb" title="返回知识库"><ArrowLeft :size="16" /></button>
      <h2>{{ kb ? `${kb.name} · 文档管理` : '文档管理' }}</h2>
      <div class="grow"></div>
      <button class="back" @click="backToChat" title="返回聊天"><X :size="16" /></button>
    </header>

    <div
      class="dropzone"
      :class="{ over: dragOver, uploading }"
      @dragover.prevent="dragOver = true"
      @dragleave="dragOver = false"
      @drop="onDrop"
    >
      <Upload :size="32" />
      <p>{{ uploading ? '正在上传…' : '拖拽文件到此处，或点击选择' }}</p>
      <input
        type="file"
        multiple
        accept=".pdf,.docx,.pptx,.md,.txt,.html,.htm,.csv,.xlsx"
        @change="onFileInput"
        :disabled="uploading"
      />
    </div>

    <div class="toolbar">
      <div class="search">
        <Search :size="14" />
        <input v-model="search" placeholder="按名称搜索…" />
      </div>
      <button class="icon-btn" @click="fetchDocs()" title="刷新"><RefreshCw :size="16" /></button>
    </div>

    <div class="list scrollbar">
      <div v-if="error" class="api-error">{{ error }}</div>
      <div v-if="loading" class="empty">加载中…</div>
      <div v-else-if="filtered.length === 0" class="empty">暂无文档</div>
      <div
        v-for="d in filtered"
        :key="d.id"
        class="row"
        :class="{ highlight: highlightDocId === d.id }"
        :ref="(el) => setRowRef(d.id, el as HTMLElement | null)"
      >
        <div class="info">
          <FileText :size="20" />
          <div>
            <div class="title">{{ d.title }}</div>
            <div class="meta">
              <span>{{ d.filename }}</span>
              <span>{{ formatBytes(d.file_size) }}</span>
              <span>{{ d.chunk_count }} chunks</span>
              <span>{{ formatDate(d.created_at) }}</span>
            </div>
            <div v-if="d.error" class="error">{{ d.error }}</div>
          </div>
        </div>
        <div class="actions">
          <span class="status" :class="statusColor(d.status)">
            <component :is="statusIcon(d.status)" :size="14" />
            {{ d.status }}
          </span>
          <button
            v-if="d.status === 'failed' || d.status === 'processing'"
            class="icon-btn"
            :disabled="retrying === d.id"
            @click.stop="retryDoc(d)"
            title="重新入库"
          >
            <RotateCcw :size="13" :class="{ spin: retrying === d.id }" />
          </button>
          <button
            class="icon-btn danger"
            :class="{ confirming: confirmingDeleteId === d.id }"
            @click.stop="removeDoc(d)"
            :title="confirmingDeleteId === d.id ? '再次点击确认删除' : '删除'"
          >
            <template v-if="confirmingDeleteId === d.id">确定？</template>
            <Trash2 v-else :size="14" />
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.docs-view {
  position: fixed; inset: 16px;
  background: var(--bg);
  border-radius: 16px;
  box-shadow: var(--shadow);
  display: flex; flex-direction: column;
  overflow: hidden;
}
header {
  display: flex; align-items: center; gap: 12px;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
}
header h2 { margin: 0; font-size: 16px; }
.grow { flex: 1; }
.back { background: transparent; border: none; cursor: pointer; color: var(--text-soft); padding: 4px; border-radius: 4px; }
.back:hover { background: var(--bg-soft); color: var(--text); }
.dropzone {
  margin: 16px;
  border: 2px dashed var(--border);
  border-radius: 12px;
  padding: 28px;
  text-align: center;
  color: var(--text-soft);
  cursor: pointer;
  position: relative;
  transition: all 0.2s;
}
.dropzone:hover, .dropzone.over { border-color: var(--primary); color: var(--primary); background: var(--primary-soft); }
.dropzone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
.dropzone.uploading { pointer-events: none; }
.dropzone p { margin: 8px 0 0; font-size: 13px; }
.toolbar {
  padding: 0 16px 8px;
  display: flex; gap: 8px; align-items: center;
}
.search {
  flex: 1;
  display: flex; align-items: center; gap: 6px;
  padding: 6px 10px;
  background: var(--bg-soft);
  border: 1px solid var(--border);
  border-radius: 8px;
}
.search input { flex: 1; border: none; outline: none; background: transparent; color: var(--text); font-size: 13px; }
.icon-btn { background: transparent; border: none; cursor: pointer; color: var(--text-soft); padding: 6px; border-radius: 6px; }
.icon-btn:hover { background: var(--bg-soft); color: var(--text); }
.icon-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.icon-btn.danger:hover { color: var(--danger); }
.icon-btn.danger.confirming {
  background: var(--danger);
  color: #fff;
  padding: 2px 8px;
  font-size: 12px;
}
.spin { animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.list { flex: 1; overflow-y: auto; padding: 8px 16px 16px; }
.row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 12px; border: 1px solid var(--border); border-radius: 10px; margin-bottom: 8px;
  transition: all 0.25s;
}
.row.highlight {
  border-color: var(--primary);
  background: var(--primary-soft);
  animation: flash 1.4s ease-out;
}
@keyframes flash {
  0% { box-shadow: 0 0 0 4px var(--primary-soft); }
  100% { box-shadow: 0 0 0 0 transparent; }
}
.info { display: flex; align-items: center; gap: 12px; flex: 1; min-width: 0; }
.info > div { flex: 1; min-width: 0; }
.title { font-size: 14px; font-weight: 500; margin-bottom: 2px; }
.meta { display: flex; gap: 10px; font-size: 11px; color: var(--text-soft); flex-wrap: wrap; }
.api-error {
  padding: 8px 10px;
  margin-bottom: 8px;
  border-radius: 6px;
  background: #fee2e2;
  color: #991b1b;
  font-size: 12px;
  white-space: pre-line;
}
.error { color: var(--danger); font-size: 12px; margin-top: 4px; }
.actions { display: flex; align-items: center; gap: 8px; }
.status { display: flex; align-items: center; gap: 4px; font-size: 11px; padding: 3px 8px; border-radius: 4px; }
.status.success { background: #d1fae5; color: #065f46; }
.status.danger { background: #fee2e2; color: #991b1b; }
.status.pending { background: #fef3c7; color: #92400e; }
.empty { padding: 30px; text-align: center; color: var(--text-soft); font-size: 13px; }
</style>
