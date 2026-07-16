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
  Copy,
  RotateCcw,
  Trash2,
  Check,
} from 'lucide-vue-next'
import { useChatStore } from '@/stores/chat'
import type { ChatMessage, Source } from '@/types'

const props = defineProps<{
  message: ChatMessage
  sources?: Source[]
  index: number
}>()

const emit = defineEmits<{
  (e: 'jumpSource', payload: { kbId: number; docId: number }): void
}>()

const chat = useChatStore()

const md = new MarkdownIt({ html: false, linkify: true, breaks: true })

const html = computed(() => md.render(props.message.content || ''))
const isUser = computed(() => props.message.role === 'user')
const isAssistant = computed(() => props.message.role === 'assistant')
const hasError = computed(() => !!props.message.error)
const thinking = computed(
  () => !!props.message.thinking && !props.message.content && !hasError.value,
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
</script>

<template>
  <div class="bubble-row" :class="{ user: isUser }">
    <div class="avatar" :class="{ user: isUser }">
      <User v-if="isUser" :size="16" />
      <Bot v-else :size="16" />
    </div>
    <div class="bubble" :class="{ user: isUser, error: hasError }">
      <div v-if="thinking" class="thinking">
        <Loader2 :size="14" class="spin" />
        <span>正在思考…</span>
      </div>
      <div v-else-if="hasError" class="error">
        <AlertCircle :size="14" />
        <span>{{ message.error }}</span>
      </div>
      <div v-else class="md-body" v-html="html"></div>

      <div v-if="message.intent && isAssistant" class="meta">
        <span class="intent" :class="`intent-${message.intent}`">{{ message.intent }}</span>
        <span v-if="message.latency_ms">· {{ message.latency_ms }} ms</span>
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
            <span v-if="s.score != null" class="src-meta">
              · {{ Math.round((s.score ?? 0) * 100) }}%
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
