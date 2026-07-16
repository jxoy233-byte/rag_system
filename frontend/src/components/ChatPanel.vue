<script setup lang="ts">
import { ref, computed, nextTick, onMounted } from 'vue'
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
} from 'lucide-vue-next'
import { useChatStore } from '@/stores/chat'
import { useKnowledgeBaseStore } from '@/stores/knowledgeBase'
import { useSettingsStore } from '@/stores/settings'
import { isTauri } from '@/utils'
import { invoke } from '@tauri-apps/api/core'
import KnowledgePicker from './KnowledgePicker.vue'
import MessageBubble from './MessageBubble.vue'
import ConversationList from './ConversationList.vue'

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
</script>

<template>
  <div class="panel" :class="settings.themeClass">
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
     @jump-source="jumpToSource"
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
</style>
