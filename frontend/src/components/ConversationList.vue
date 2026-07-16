<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import {
  MessageSquare,
  Plus,
  RefreshCw,
  Pencil,
  Trash2,
  Check,
  X,
} from 'lucide-vue-next'
import { useChatStore } from '@/stores/chat'
import { useSettingsStore } from '@/stores/settings'
import { formatDate } from '@/utils'

const props = defineProps<{ open: boolean }>()
const emit = defineEmits<{
  (e: 'close'): void
}>()

const chat = useChatStore()
const settings = useSettingsStore()
const router = useRouter()

const editingId = ref<number | null>(null)
const editingTitle = ref('')
// 二次确认态：避免误删。点击删除按钮第一次变成"确定？"，再点才真删；
// 3 秒内未确认或列表关闭都自动取消。
const confirmingId = ref<number | null>(null)
let confirmTimer: ReturnType<typeof setTimeout> | null = null

watch(
  () => props.open,
  (v) => {
    if (v) {
      chat.fetchConversations(settings.currentKbId)
    } else {
      cancelConfirm()
    }
  },
)

async function switchTo(id: number) {
  if (chat.conversationId === id) {
    emit('close')
    return
  }
  await chat.loadConversation(id)
  emit('close')
}

function startNew() {
  chat.reset()
  emit('close')
}

function startEdit(id: number, title: string, e: Event) {
  e.stopPropagation()
  editingId.value = id
  editingTitle.value = title
}

async function commitEdit() {
  if (editingId.value == null) return
  const id = editingId.value
  const t = editingTitle.value.trim()
  if (t) {
    try {
      await chat.renameConversation(id, t)
    } catch (e) {
      console.warn('rename failed:', e)
    }
  }
  editingId.value = null
}

function cancelEdit(e: Event) {
  e.stopPropagation()
  editingId.value = null
}

function cancelConfirm() {
  if (confirmTimer) {
    clearTimeout(confirmTimer)
    confirmTimer = null
  }
  confirmingId.value = null
}

function askConfirm(id: number, e: Event) {
  e.stopPropagation()
  cancelConfirm()
  confirmingId.value = id
  confirmTimer = setTimeout(cancelConfirm, 3000)
}

async function remove(id: number, e: Event) {
  e.stopPropagation()
  if (confirmingId.value !== id) {
    askConfirm(id, e)
    return
  }
  cancelConfirm()
  try {
    await chat.deleteConversation(id)
  } catch (err: any) {
    const msg = err?.data?.detail || err?.message || String(err)
    // Tauri 2.x 默认禁用 window.alert，这里只在 web 环境 fallback
    if (typeof window !== 'undefined' && !('__TAURI_INTERNALS__' in window)) {
      window.alert('删除会话失败：' + msg)
    }
    console.error('delete conversation failed:', err)
  }
}

function gotoKb() {
  emit('close')
  router.push('/kb')
}

onMounted(() => {
  if (props.open) chat.fetchConversations(settings.currentKbId)
})

onUnmounted(() => {
  cancelConfirm()
})
</script>

<template>
  <div v-if="open" class="mask" @click.self="emit('close')">
    <aside class="drawer">
      <header>
        <button class="back" @click="emit('close')" title="关闭">
          <X :size="16" />
        </button>
        <h3>
          <MessageSquare :size="16" />
          历史会话
        </h3>
        <button class="icon" @click="chat.fetchConversations(settings.currentKbId)" title="刷新">
          <RefreshCw :size="14" />
        </button>
      </header>

      <button class="new-btn" @click="startNew">
        <Plus :size="14" /> 新建会话
      </button>

      <div class="list scrollbar">
        <div v-if="chat.conversationsLoading" class="empty">加载中…</div>
        <div v-else-if="chat.conversations.length === 0" class="empty">
          还没有会话
        </div>
        <div
          v-for="c in chat.conversations"
          :key="c.id"
          class="item"
          :class="{ active: chat.conversationId === c.id, editing: editingId === c.id }"
          @click="switchTo(c.id)"
        >
          <template v-if="editingId === c.id">
            <input
              v-model="editingTitle"
              class="rename-input"
              @keydown.enter="commitEdit"
              @keydown.esc="cancelEdit"
              @click.stop
            />
            <button class="icon" @click.stop="commitEdit" title="保存">
              <Check :size="13" />
            </button>
            <button class="icon" @click.stop="cancelEdit" title="取消">
              <X :size="13" />
            </button>
          </template>
          <template v-else>
            <div class="title">{{ c.title || '新会话' }}</div>
            <div class="meta">{{ formatDate(c.updated_at) }}</div>
            <div class="ops">
              <button class="icon" @click.stop="startEdit(c.id, c.title, $event)" title="重命名">
                <Pencil :size="12" />
              </button>
              <button
                class="icon danger"
                :class="{ confirming: confirmingId === c.id }"
                @click.stop="remove(c.id, $event)"
                :title="confirmingId === c.id ? '再次点击确认删除' : '删除'"
              >
                <template v-if="confirmingId === c.id">确定？</template>
                <Trash2 v-else :size="12" />
              </button>
            </div>
          </template>
        </div>
      </div>

      <footer>
        <button class="link" @click="gotoKb">
          <MessageSquare :size="12" /> 知识库管理
        </button>
      </footer>
    </aside>
  </div>
</template>

<style scoped>
.mask {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.25);
  display: flex;
  z-index: 200;
}
.drawer {
  width: 280px;
  max-width: 80%;
  background: var(--bg);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  box-shadow: var(--shadow);
  animation: slidein 0.2s ease-out;
}
@keyframes slidein {
  from { transform: translateX(-100%); }
  to { transform: translateX(0); }
}
header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px;
  border-bottom: 1px solid var(--border);
}
header h3 {
  margin: 0;
  font-size: 14px;
  display: flex;
  align-items: center;
  gap: 6px;
  flex: 1;
}
.back, .icon {
  background: transparent;
  border: none;
  cursor: pointer;
  color: var(--text-soft);
  padding: 4px;
  border-radius: 4px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}
.back:hover, .icon:hover {
  background: var(--bg-soft);
  color: var(--text);
}
.icon.danger:hover {
  color: var(--danger);
}
.icon.danger.confirming {
  background: var(--danger);
  color: #fff;
  padding: 2px 6px;
  font-size: 11px;
  animation: pulse 0.8s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.05); }
}
.new-btn {
  margin: 8px 12px 4px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 7px 12px;
  border-radius: 8px;
  background: var(--primary);
  color: var(--primary-fg);
  border: none;
  cursor: pointer;
  font-size: 13px;
  justify-content: center;
}
.new-btn:hover { opacity: 0.92; }
.list {
  flex: 1;
  overflow-y: auto;
  padding: 6px 8px;
}
.empty {
  padding: 30px 16px;
  text-align: center;
  font-size: 13px;
  color: var(--text-soft);
}
.item {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 8px 10px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 13px;
  position: relative;
}
.item:hover {
  background: var(--bg-soft);
}
.item.active {
  background: var(--primary-soft);
  color: var(--primary);
}
.item.editing {
  flex-direction: row;
  align-items: center;
  gap: 4px;
  cursor: default;
}
.title {
  font-weight: 500;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.meta {
  font-size: 11px;
  color: var(--text-soft);
}
.ops {
  position: absolute;
  right: 6px;
  top: 50%;
  transform: translateY(-50%);
  display: none;
  gap: 2px;
  background: var(--bg);
  padding: 2px;
  border-radius: 4px;
}
.item:hover .ops,
.item.active .ops {
  display: flex;
}
.rename-input {
  flex: 1;
  border: 1px solid var(--primary);
  border-radius: 4px;
  padding: 3px 6px;
  font-size: 13px;
  background: var(--bg);
  color: var(--text);
  outline: none;
}
footer {
  border-top: 1px solid var(--border);
  padding: 8px;
}
.link {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  background: transparent;
  border: none;
  color: var(--primary);
  cursor: pointer;
  padding: 4px 6px;
  border-radius: 4px;
}
.link:hover {
  background: var(--primary-soft);
}
</style>