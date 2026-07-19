<script setup lang="ts">
import MarkdownIt from 'markdown-it'
import { computed, ref } from 'vue'
import {
  User,
  Bot,
  Loader2,
  AlertCircle,
  ExternalLink,
  FileText,
  Globe,
  Library,
  Copy,
  RotateCcw,
  Trash2,
  Check,
  SearchX,
} from 'lucide-vue-next'
import { useChatStore } from '@/stores/chat'
import type { ChatMessage, DocHit, Source } from '@/types'

const props = defineProps<{
  message: ChatMessage
  sources?: Source[]
  docHits?: DocHit[]
  index: number
}>()

const emit = defineEmits<{
  (e: 'jumpSource', payload: { kbId: number; docId: number }): void
  (
    e: 'openChunk',
    payload: {
      kbId: number | null
      docId: number | null
      chunkId: string
      source: Source
    },
  ): void
}>()

const chat = useChatStore()

const md = new MarkdownIt({ html: false, linkify: true, breaks: true })

// 把答案里 [n] 引用标记转成可点击 chip（<sup class="cite-chip" data-cite-idx="N">）。
// markdown-it 的 inline rule 在 render 阶段触发，对流式 token 反复重渲也安全。
md.inline.ruler.after('emphasis', 'cite', (state, silent) => {
  const start = state.pos
  const m = state.src.slice(start).match(/^\[(\d+)\]/)
  if (!m) return false
  if (!silent) {
    const tok = state.push('html_inline', '', 0)
    tok.content = `<sup class="cite-chip" data-cite-idx="${m[1]}">${m[1]}</sup>`
  }
  state.pos += m[0].length
  return true
})

const html = computed(() => md.render(props.message.content || ''))
const isUser = computed(() => props.message.role === 'user')
const isAssistant = computed(() => props.message.role === 'assistant')
const hasError = computed(() => !!props.message.error)
const isRefused = computed(() => !!props.message.refused && !hasError.value)
const thinking = computed(
  () => !!props.message.thinking && !props.message.content && !hasError.value && !isRefused.value,
)
const isLast = computed(() => props.index === chat.messages.length - 1)
const isLastAssistant = computed(() => isAssistant.value && isLast.value)
const copied = ref(false)

async function copy() {
  try {
    await navigator.clipboard.writeText(props.message.content || '')
    copied.value = true
    setTimeout(() => (copied.value = false), 1200)
  } catch (e) {
    console.warn('copy failed:', e)
  }
}

async function regenerate() {
  let userIdx = -1
  for (let i = props.index - 1; i >= 0; i--) {
    if (chat.messages[i].role === 'user') {
      userIdx = i
      break
    }
  }
  if (userIdx < 0) return
  await chat.regenerateFrom(userIdx)
}

function removeMsg() {
  chat.removeMessage(props.index)
}

function onSourceClick(s: Source) {
  if (s.source_type === 'web' || !s.doc_id) return
  const kbId = (s as any).kb_id as number | undefined
  if (!kbId) return
  emit('jumpSource', { kbId, docId: s.doc_id })
}

// 引用 chip 点击 → 通知父组件打开 chunk 详情抽屉。
// 用 .md-body 上的事件委托：v-html 多次重渲 chip 时 onClick 不会丢。
// 兜底：没有 chunk_id（bm25/web 来源）或编号越界也 emit，
// 由 ChatPanel 用 snippet 兜底展示，避免「点了没反应」。
function onMdClick(e: MouseEvent) {
  const target = e.target as HTMLElement | null
  const chip = target?.closest('.cite-chip') as HTMLElement | null
  if (!chip) return
  const idxStr = chip.dataset.citeIdx
  if (!idxStr) return
  const idx = Number(idxStr)
  if (!Number.isFinite(idx) || idx < 1) return
  const source = props.sources?.[idx - 1]
  if (!source) return
  const kbId = (source as any).kb_id as number | null
  const chunkId = (source.chunk_id as string | null) ?? ''
  const docId = (source.doc_id as number | null) ?? null
  emit('openChunk', {
    kbId,
    docId,
    chunkId,
    source,
  })
}
</script>

<template>
  <div class="bubble-row" :class="{ user: isUser }">
    <div class="avatar" :class="{ user: isUser }">
      <User v-if="isUser" :size="16" />
      <Bot v-else :size="16" />
    </div>
    <div class="bubble" :class="{ user: isUser, error: hasError, refused: isRefused }">
      <div v-if="thinking" class="thinking">
        <Loader2 :size="14" class="spin" />
        <span>正在思考…</span>
      </div>
      <div v-else-if="hasError" class="error">
        <AlertCircle :size="14" />
        <span>{{ message.error }}</span>
      </div>
      <div v-else-if="isRefused" class="refused">
        <SearchX :size="14" />
        <span>{{ message.content || '知识库中暂未收录与此问题相关的内容。' }}</span>
      </div>
      <div v-else class="md-body" v-html="html" @click="onMdClick"></div>

      <div v-if="message.intent && isAssistant" class="meta">
        <span class="intent" :class="`intent-${message.intent}`">{{ message.intent }}</span>
        <span v-if="message.latency_ms">· {{ message.latency_ms }} ms</span>
      </div>

      <!-- 相关文档：doc-level BM25(title+filename+summary) 的 top 命中。
           每条展示 title + summary 摘要，让用户看到「系统为什么认为这些文档相关」。
           独立于 chunk-level 引用（下方 sources），用于解释 RAG 的判断依据。 -->
      <div v-if="docHits && docHits.length" class="doc-hits">
        <div class="dh-title">
          <Library :size="13" />
          <span>相关文档 ({{ docHits.length }})</span>
        </div>
        <ul>
          <li v-for="(d, i) in docHits" :key="d.doc_id" class="doc-hit">
            <div class="dh-head">
              <FileText :size="12" />
              <span class="dh-title-text">{{ d.title || d.filename }}</span>
              <span v-if="d.filename && d.title && d.filename !== d.title" class="dh-filename">
                {{ d.filename }}
              </span>
            </div>
            <div v-if="d.summary" class="dh-summary">{{ d.summary }}</div>
          </li>
        </ul>
      </div>

      <div v-if="sources && sources.length" class="sources">
        <div class="src-title">引用 ({{ sources.length }})</div>
        <ul>
          <li
            v-for="(s, i) in sources"
            :key="i"
            :class="{ clickable: s.source_type !== 'web' && s.doc_id }"
            @click="onSourceClick(s)"
          >
            <FileText v-if="s.source_type !== 'web'" :size="12" />
            <Globe v-else :size="12" />
            <a
              v-if="s.url"
              :href="s.url"
              target="_blank"
              rel="noopener"
              class="src-label"
              @click.stop
            >{{ s.document || s.url }}</a>
            <span v-else class="src-label">{{ s.document }}</span>
            <span v-if="s.page != null" class="src-meta">· p.{{ s.page }}</span>
            <span v-if="(s.rerank_score ?? s.score) != null" class="src-meta">
              · 相关度 {{ Math.min(100, Math.round((s.rerank_score ?? s.score ?? 0) * 100)) }}%
            </span>
            <a
              v-if="s.url"
              :href="s.url"
              target="_blank"
              rel="noopener"
              class="src-open"
              title="打开"
              @click.stop
            >
              <ExternalLink :size="11" />
            </a>
          </li>
        </ul>
      </div>
    </div>

    <div class="actions" v-if="!thinking">
      <button class="act" @click="copy" :title="copied ? '已复制' : '复制'">
        <Check v-if="copied" :size="12" />
        <Copy v-else :size="12" />
      </button>
      <button
        v-if="isLastAssistant && !chat.streaming"
        class="act"
        @click="regenerate"
        title="重新生成"
      >
        <RotateCcw :size="12" />
      </button>
      <button class="act danger" @click="removeMsg" title="删除">
        <Trash2 :size="12" />
      </button>
    </div>
  </div>
</template>

<style scoped>
.bubble-row {
  display: flex;
  gap: 10px;
  align-items: flex-start;
  margin-bottom: 18px;
  position: relative;
}
.bubble-row.user {
  flex-direction: row-reverse;
}
.actions {
  display: flex;
  gap: 4px;
  align-self: center;
  opacity: 0;
  transition: opacity 0.12s;
  pointer-events: none;
}
.bubble-row:hover .actions {
  opacity: 1;
  pointer-events: auto;
}
.act {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg);
  color: var(--text-soft);
  cursor: pointer;
}
.act:hover {
  background: var(--bg-soft);
  color: var(--text);
}
.act.danger:hover {
  color: var(--danger);
  border-color: var(--danger);
}
.bubble-row.user .actions {
  flex-direction: row-reverse;
}
.avatar {
  width: 28px;
  height: 28px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--bg-soft);
  color: var(--text-soft);
  flex-shrink: 0;
}
.avatar.user {
  background: var(--primary);
  color: var(--primary-fg);
}
.bubble {
  max-width: calc(100% - 50px);
  padding: 10px 14px;
  border-radius: 12px;
  background: var(--bg-soft);
  color: var(--text);
  font-size: 14px;
  line-height: 1.6;
}
.bubble.user {
  background: var(--primary);
  color: var(--primary-fg);
}
.bubble.error {
  background: #fee2e2;
  color: #991b1b;
}
.bubble.refused {
  background: #f3f4f6;
  color: #6b7280;
}
.thinking {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--text-soft);
  font-size: 13px;
}
.spin {
  animation: spin 1s linear infinite;
}
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
.error {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
}
.refused {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
}
.refused :deep(svg) {
  flex-shrink: 0;
}
.meta {
  margin-top: 8px;
  font-size: 11px;
  color: var(--text-soft);
  display: flex;
  align-items: center;
  gap: 4px;
}
.intent {
  display: inline-block;
  padding: 1px 6px;
  border-radius: 4px;
  background: var(--border);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.intent-rag { background: #dbeafe; color: #1e40af; }
.intent-web { background: #fef3c7; color: #92400e; }
.intent-direct { background: #f3f4f6; color: #4b5563; }
.intent-hybrid { background: #ede9fe; color: #5b21b6; }

/* doc-level 命中：相关文档清单 + summary */
.doc-hits {
  margin-top: 10px;
  padding: 8px 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: rgba(99, 102, 241, 0.04);
}
.dh-title {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  color: var(--text-soft);
  margin-bottom: 6px;
  font-weight: 500;
}
.doc-hits ul {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.doc-hit {
  font-size: 12px;
  color: var(--text);
  line-height: 1.5;
}
.dh-head {
  display: flex;
  align-items: baseline;
  gap: 6px;
  flex-wrap: wrap;
}
.dh-title-text {
  font-weight: 500;
  color: var(--text);
}
.dh-filename {
  font-size: 11px;
  color: var(--text-soft);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
.dh-summary {
  margin-top: 2px;
  padding-left: 18px;
  color: var(--text-soft);
  font-size: 11.5px;
  line-height: 1.55;
  border-left: 2px solid rgba(99, 102, 241, 0.25);
}

.sources {
  margin-top: 10px;
  padding-top: 8px;
  border-top: 1px dashed var(--border);
}
.src-title {
  font-size: 11px;
  color: var(--text-soft);
  margin-bottom: 4px;
}
.sources ul {
  list-style: none;
  padding: 0;
  margin: 0;
}
.sources li {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  color: var(--text-soft);
  margin-bottom: 2px;
}
.src-label {
  color: var(--text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 240px;
}
.src-meta {
  color: var(--text-soft);
  font-size: 11px;
}
.src-open {
  margin-left: auto;
  color: var(--text-soft);
}
.src-open:hover {
  color: var(--primary);
}
</style>
