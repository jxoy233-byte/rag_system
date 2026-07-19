<script setup lang="ts">
import { ref, computed, nextTick, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import {
  Send,
  X,
  Minus,
  Settings,
  Database,
  FileText,
  Trash2,
  Globe,
  Moon,
  Sun,
  Square,
  History,
  Search,
  Upload,
} from 'lucide-vue-next'
import { useChatStore } from '@/stores/chat'
import { useKnowledgeBaseStore } from '@/stores/knowledgeBase'
import { useSettingsStore } from '@/stores/settings'
import { isTauri } from '@/utils'
import { invoke } from '@tauri-apps/api/core'
import { chunkApi, type ChunkDetail } from '@/api/client'
import type { Source } from '@/types'
import KnowledgePicker from './KnowledgePicker.vue'
import MessageBubble from './MessageBubble.vue'
import ConversationList from './ConversationList.vue'
import AppDrawer from './AppDrawer.vue'

const chat = useChatStore()
const kbStore = useKnowledgeBaseStore()
const settings = useSettingsStore()
const router = useRouter()

const emit = defineEmits<{
  (e: 'close'): void
  (e: 'minimize'): void
}>()

const input = ref('')
const scrollerRef = ref<HTMLElement | null>(null)
const inputRef = ref<HTMLTextAreaElement | null>(null)
const showHistory = ref(false)
const dragOver = ref(false)

// 引用 chip 抽屉状态
const chunkDrawerOpen = ref(false)
const chunkDetail = ref<ChunkDetail | null>(null)
const chunkLoading = ref(false)
const chunkError = ref<string | null>(null)
// 记一下最近一次点击的 payload，给抽屉 footer 用（外链/路由跳转）
const lastChunkPayload = ref<{
  kbId: number | null
  docId: number | null
  chunkId: string
  source: Source
} | null>(null)
const payloadForFooter = computed(() => lastChunkPayload.value)

const placeholder = computed(() =>
  settings.currentKbId
    ? '向知识库提问，或补全上下文…'
    : '直接对话、问文档，或选择知识库后再问',
)

function scrollToBottom(smooth = true) {
  nextTick(() => {
    if (scrollerRef.value) {
      scrollerRef.value.scrollTo({
        top: scrollerRef.value.scrollHeight,
        behavior: smooth ? 'smooth' : 'auto',
      })
    }
  })
}

async function send() {
  const q = input.value.trim()
  if (!q || chat.streaming) return
  input.value = ''
  autoSize()
  await chat.send({
    question: q,
    knowledgeBaseId: settings.currentKbId,
    enableWeb: settings.enableWeb,
  })
  scrollToBottom()
}

function autoSize() {
  const ta = inputRef.value
  if (!ta) return
  ta.style.height = 'auto'
  ta.style.height = Math.min(120, ta.scrollHeight) + 'px'
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    send()
  }
}

function newChat() {
  chat.reset()
}

function gotoKbManage() {
  router.push('/kb')
}

function gotoKbEntry() {
  // 单入口：有当前 KB → 进文档页；没有 → 进 KB 列表
  if (settings.currentKbId) {
    router.push(`/docs/${settings.currentKbId}`)
  } else {
    gotoKbManage()
  }
}

function gotoSearch() {
  router.push('/search')
}

function jumpToSource(payload: { kbId: number; docId: number }) {
  router.push(`/docs/${payload.kbId}?doc=${payload.docId}`)
}

async function openChunk(payload: {
  kbId: number | null
  docId: number | null
  chunkId: string
  source: Source
}) {
  chunkDrawerOpen.value = true
  lastChunkPayload.value = payload
  chunkDetail.value = null
  chunkError.value = null
  chunkLoading.value = true

  const src = payload.source
  // 兜底：bm25/web 来源或越界时，ChatPanel 拿不到 chunk_id，直接用 source.snippet
  // 拼一个「摘要视图」，保证任何引用点击都有反馈，不静默失败。
  const fallback = (): ChunkDetail => ({
    chunk_id: payload.chunkId || '',
    doc_id: payload.docId ?? 0,
    kb_id: payload.kbId ?? 0,
    text: src.snippet,
    page: src.page ?? null,
    section: null,
    score: src.score ?? null,
    rerank_score: src.rerank_score ?? null,
    document: src.document || src.url || null,
    source_type: src.source_type,
  })

  if (!payload.chunkId || payload.kbId == null || payload.docId == null) {
    chunkDetail.value = fallback()
    chunkLoading.value = false
    return
  }

  try {
    chunkDetail.value = await chunkApi.getDetail(payload.kbId, payload.docId, payload.chunkId)
  } catch (e: any) {
    // 后端拉失败也不应该只显示 error——用 snippet 兜底，至少让用户看到摘要
    chunkDetail.value = fallback()
    chunkError.value = e?.message || '加载引用详情失败（已显示摘要）'
  } finally {
    chunkLoading.value = false
  }
}

function gotoChunkDoc() {
  if (!chunkDetail.value) return
  const { kb_id, doc_id } = chunkDetail.value
  if (!kb_id || !doc_id) return
  router.push(`/docs/${kb_id}?doc=${doc_id}`)
  chunkDrawerOpen.value = false
}

function gotoSettings() {
  router.push('/settings')
}

function toggleWeb() {
  settings.setEnableWeb(!settings.enableWeb)
}

function toggleTheme() {
  settings.toggleTheme()
}

async function minimizeWindow() {
  // 收起成浮球（窗口缩成 80×80，仍可见可点），而不是彻底隐藏窗口。
  // 这样用户点浮球就能再展开，不至于只能依赖 ⌘⇧Space / 托盘。
  emit('close')
}

// 拖拽：调 Rust 的 start_window_drag 命令
function handleHeaderMouseDown(e: MouseEvent) {
  const target = e.target as HTMLElement | null
  // 按钮 / 输入 / 链接 / 拾取器放过，让原生 click 生效
  if (target?.closest('button, input, textarea, select, a, [role="button"], .kb-picker, .trigger, .menu')) {
    return
  }
  if (!isTauri()) return
  e.preventDefault()
  // 走 Rust 的命令（同步发起，不要 await，否则 mousedown 已经处理完）
  invoke('start_window_drag').catch((err) => console.warn('start_window_drag failed:', err))
}

onMounted(() => {
  scrollToBottom(false)
  if (kbStore.items.length === 0) kbStore.fetch()
})

onUnmounted(() => {
  dragOver.value = false
})

// 拖拽上传：检测拖拽文件 -> 跳转到 DocsView 上传
function onDragOver(e: DragEvent) {
  e.preventDefault()
  if (e.dataTransfer?.types?.includes('Files')) {
    dragOver.value = true
  }
}

function onDragLeave(e: DragEvent) {
  e.preventDefault()
  // 只有真正离开面板时才关闭 dragOver
  const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
  const x = e.clientX
  const y = e.clientY
  if (x < rect.left || x > rect.right || y < rect.top || y > rect.bottom) {
    dragOver.value = false
  }
}

async function onDropFiles(e: DragEvent) {
  e.preventDefault()
  dragOver.value = false

  // 直接跳转到文档页面
  if (settings.currentKbId) {
    router.push({ name: 'docs', params: { id: settings.currentKbId } })
  } else {
    router.push({ name: 'knowledgeBases' })
  }
}
</script>

<template>
  <div
    class="panel"
    :class="settings.themeClass"
    @dragover.prevent="onDragOver"
    @dragleave="onDragLeave"
    @drop="onDropFiles"
  >
    <!-- 拖拽上传提示遮罩 -->
    <div v-if="dragOver" class="drag-overlay">
      <Upload :size="48" />
      <p>放开鼠标，进入文档管理上传</p>
    </div>

    <header class="bar" data-tauri-drag-region @mousedown="handleHeaderMouseDown">
      <KnowledgePicker />
      <div class="grow"></div>
      <button
        class="icon-btn"
        :class="{ active: settings.enableWeb }"
        @click="toggleWeb"
        title="联网搜索"
      >
        <Globe :size="15" />
      </button>
      <button
        class="icon-btn"
        @click="gotoKbEntry"
        :title="settings.currentKbId ? '当前知识库的文档' : '知识库管理'"
      >
        <component :is="settings.currentKbId ? FileText : Database" :size="15" />
      </button>
      <button class="icon-btn" @click="gotoSearch" title="独立检索">
        <Search :size="15" />
      </button>
      <button class="icon-btn" @click="showHistory = true" title="历史会话">
        <History :size="15" />
      </button>
      <button class="icon-btn" @click="toggleTheme" title="主题">
        <Sun v-if="settings.theme === 'dark'" :size="15" />
        <Moon v-else :size="15" />
      </button>
      <button class="icon-btn" @click="gotoSettings" title="设置">
        <Settings :size="15" />
      </button>
      <button class="icon-btn" @click="newChat" title="新会话">
        <Trash2 :size="15" />
      </button>
      <button class="icon-btn" @click="minimizeWindow" title="最小化" v-if="isTauri()">
        <Minus :size="15" />
      </button>
      <button class="icon-btn" @click="emit('close')" title="收起">
        <X :size="15" />
      </button>
    </header>

 <div class="scroll scrollbar" ref="scrollerRef">
   <div v-if="!chat.hasMessages" class="empty">
     <div class="empty-icon">✨</div>
     <h3>开始一段新的对话</h3>
     <p>可以使用文档问答、联网搜索或直接对话</p>
     <div class="examples">
       <button
         v-for="t in [
           '总结我知识库里的核心要点',
           '请联网搜索最近的 AI 行业新闻',
           '用 Python 写一个 HTTP 服务器',
         ]"
         :key="t"
         @click="input = t; send()"
       >
         {{ t }}
       </button>
     </div>
   </div>
   <MessageBubble
     v-for="(m, i) in chat.messages"
     :key="i"
     :index="i"
     :message="m"
     :sources="m.sources"
     :doc-hits="m.doc_hits"
     @jump-source="jumpToSource"
     @open-chunk="openChunk"
   />
   <div v-if="chat.error && !chat.streaming" class="global-error">
     {{ chat.error }}
   </div>
 </div>

    <footer class="composer">
      <textarea
        ref="inputRef"
        rows="1"
        v-model="input"
        :placeholder="placeholder"
        @keydown="onKeydown"
        @input="autoSize"
        :disabled="chat.streaming"
      ></textarea>
      <button
        v-if="chat.streaming"
        class="send stop"
        @click="chat.abort()"
        title="停止"
      >
        <Square :size="14" />
      </button>
      <button
        v-else
        class="send"
        :disabled="!input.trim()"
        @click="send"
        title="发送"
      >
        <Send :size="14" />
      </button>
    </footer>

    <ConversationList :open="showHistory" @close="showHistory = false" />

    <!-- 引用 chip 抽屉：显示 chunk 完整内容 + 跳转文档 -->
    <AppDrawer
      :open="chunkDrawerOpen"
      :title="chunkDetail?.document || '引用详情'"
      :width="480"
      @update:open="chunkDrawerOpen = $event"
    >
      <div v-if="chunkLoading" class="chunk-state">加载中…</div>
      <div v-else-if="chunkError && !chunkDetail" class="chunk-state chunk-error">{{ chunkError }}</div>
      <div v-else-if="chunkDetail" class="chunk-detail">
        <div class="chunk-meta">
          <span v-if="chunkDetail.page != null">p.{{ chunkDetail.page }}</span>
          <span v-if="chunkDetail.section">§{{ chunkDetail.section }}</span>
          <span v-if="chunkDetail.score != null">· 相关度 {{ Math.round(chunkDetail.score * 100) }}%</span>
          <span v-if="chunkDetail.rerank_score != null">· 重排 {{ Math.round(chunkDetail.rerank_score * 100) }}%</span>
          <span v-if="chunkDetail.source_type === 'web'" class="web-badge">· 网络</span>
        </div>
        <pre class="chunk-text">{{ chunkDetail.text }}</pre>
        <div v-if="chunkError" class="chunk-fallback-hint">{{ chunkError }}</div>
      </div>
      <template #footer>
        <div class="drawer-actions">
          <a
            v-if="payloadForFooter && payloadForFooter.source.url"
            :href="payloadForFooter.source.url"
            target="_blank"
            rel="noopener"
            class="btn-primary"
          >打开外链</a>
          <button class="btn-primary" :disabled="!chunkDetail || !chunkDetail.doc_id" @click="gotoChunkDoc">
            跳转到文档
          </button>
        </div>
      </template>
    </AppDrawer>
  </div>
</template>

<style scoped>
.panel {
  position: fixed;
  inset: 16px;
  display: flex;
  flex-direction: column;
  background: var(--bg-overlay);
  -webkit-backdrop-filter: saturate(160%) blur(20px);
  backdrop-filter: saturate(160%) blur(20px);
  border: 1px solid var(--border);
  border-radius: 16px;
  box-shadow: var(--shadow);
  overflow: hidden;
}
.bar {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-overlay);
  /* 拖拽区只限 header：
     - data-tauri-drag-region：Tauri 原生识别
     - -webkit-app-region: drag：webkit 兜底
     子元素全部 no-drag，保证按钮可点 */
  -webkit-app-region: drag;
  cursor: grab;
}
.bar:active {
  cursor: grabbing;
}
/* 排除 header 内所有交互元素，确保它们可以正常点击 */
.bar button,
.bar input,
.bar textarea,
.bar select,
.bar a,
.bar [role='button'],
.bar .kb-picker,
.bar :deep(*) {
  -webkit-app-region: no-drag;
}
.grow { flex: 1; }
.icon-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 6px;
  border: none;
  background: transparent;
  color: var(--text-soft);
  flex-shrink: 0;
}
.icon-btn:hover {
  background: var(--bg-soft);
  color: var(--text);
}
.icon-btn.active {
  background: var(--primary-soft);
  color: var(--primary);
}
.scroll {
  flex: 1;
  overflow-y: auto;
  padding: 18px 18px 0;
}
.empty {
  text-align: center;
  padding: 60px 20px;
  color: var(--text-soft);
}
.empty-icon { font-size: 36px; margin-bottom: 8px; }
.empty h3 { margin: 0 0 4px; color: var(--text); font-size: 16px; }
.empty p { margin: 0 0 18px; font-size: 13px; }
.examples {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-width: 420px;
  margin: 0 auto;
}
.examples button {
  text-align: left;
  padding: 8px 12px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg);
  color: var(--text);
  font-size: 13px;
  cursor: pointer;
  transition: all 0.15s;
}
.examples button:hover {
  border-color: var(--primary);
  background: var(--primary-soft);
  color: var(--primary);
}
.global-error {
  padding: 10px 12px;
  background: #fee2e2;
  color: #991b1b;
  border-radius: 8px;
  font-size: 13px;
  margin: 14px 0;
}
.composer {
  display: flex;
  align-items: flex-end;
  gap: 10px;
  padding: 12px 14px;
  border-top: 1px solid var(--border);
  background: var(--bg-overlay);
}
.composer textarea {
  flex: 1;
  resize: none;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px 12px;
  background: var(--bg);
  color: var(--text);
  font-size: 14px;
  line-height: 1.5;
  outline: none;
  max-height: 120px;
}
.composer textarea:focus {
  border-color: var(--primary);
  box-shadow: 0 0 0 2px var(--primary-soft);
}
.send {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border-radius: 10px;
  border: none;
  background: var(--primary);
  color: var(--primary-fg);
  flex-shrink: 0;
}
.send:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
.send.stop {
  background: var(--danger);
}
.drag-overlay {
  position: absolute;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  z-index: 100;
  border-radius: 12px;
  color: #fff;
  pointer-events: none;
}
.drag-overlay p {
  font-size: 14px;
  opacity: 0.9;
}

/* ===== 引用 chip 抽屉 ===== */
.chunk-state {
  padding: 40px 0;
  text-align: center;
  color: var(--text-soft);
  font-size: 13px;
}
.chunk-state.chunk-error {
  color: var(--danger);
}
.chunk-detail {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.chunk-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  font-size: 12px;
  color: var(--text-soft);
  padding-bottom: 10px;
  border-bottom: 1px dashed var(--border);
}
.chunk-text {
  margin: 0;
  padding: 12px;
  background: var(--bg-soft);
  border-radius: 8px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 13px;
  line-height: 1.6;
  color: var(--text);
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 60vh;
  overflow: auto;
}
.web-badge {
  background: #fef3c7;
  color: #92400e;
  padding: 1px 6px;
  border-radius: 4px;
  font-size: 11px;
}
.chunk-fallback-hint {
  font-size: 11px;
  color: var(--text-soft);
  border-top: 1px dashed var(--border);
  padding-top: 8px;
}
.drawer-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}
.btn-primary {
  padding: 6px 14px;
  border: none;
  border-radius: 6px;
  background: var(--primary);
  color: var(--primary-fg);
  font-size: 13px;
  cursor: pointer;
}
.btn-primary:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
.btn-primary:hover:not(:disabled) {
  filter: brightness(0.95);
}
</style>
