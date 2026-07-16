<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import { Database, Plus, ChevronDown, Check } from 'lucide-vue-next'
import { useKnowledgeBaseStore } from '@/stores/knowledgeBase'
import { useSettingsStore } from '@/stores/settings'
import { useRouter } from 'vue-router'

const settings = useSettingsStore()
const store = useKnowledgeBaseStore()
const router = useRouter()

const open = ref(false)
const rootRef = ref<HTMLElement | null>(null)
const current = computed(() => store.getById(settings.currentKbId))

function toggle() {
  open.value = !open.value
}

function pick(id: number | null) {
  settings.setKb(id)
  open.value = false
}

function goManage() {
  router.push('/kb')
  open.value = false
}

async function refresh() {
  await store.fetch()
  if (settings.currentKbId !== null && !store.getById(settings.currentKbId)) {
    settings.setKb(null)
  }
}

function onDocClick(e: MouseEvent) {
  if (!rootRef.value) return
  if (!rootRef.value.contains(e.target as Node)) open.value = false
}

onMounted(async () => {
  document.addEventListener('click', onDocClick)
  if (!store.loaded) await refresh()
})

onBeforeUnmount(() => {
  document.removeEventListener('click', onDocClick)
})
</script>

<template>
  <div class="kb-picker" ref="rootRef">
    <button class="trigger" @click="toggle">
      <Database :size="14" />
      <span class="label">{{ current ? current.name : '未选择知识库' }}</span>
      <span v-if="current" class="count">{{ current.chunk_count }} chunks</span>
      <ChevronDown :size="14" :class="{ open: open }" />
    </button>
    <div v-if="open" class="menu">
      <div class="menu-header">
        <span>知识库</span>
        <button class="link" @click="refresh">刷新</button>
      </div>
      <div
        class="item all"
        :class="{ active: settings.currentKbId == null }"
        @click="pick(null)"
      >
        <span class="name">不使用</span>
        <Check v-if="settings.currentKbId == null" :size="14" />
      </div>
      <div
        v-for="kb in store.items"
        :key="kb.id"
        class="item"
        :class="{ active: settings.currentKbId === kb.id }"
        @click="pick(kb.id)"
      >
        <span class="name">{{ kb.name }}</span>
        <span class="meta">{{ kb.doc_count }} docs</span>
        <Check v-if="settings.currentKbId === kb.id" :size="14" />
      </div>
      <div v-if="store.error" class="error">{{ store.error }}</div>
      <div v-else-if="store.items.length === 0" class="empty">暂无知识库</div>
      <div class="menu-footer">
        <button class="link" @click="goManage">
          <Plus :size="12" /> 管理知识库
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.kb-picker {
  position: relative;
}
.trigger {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg);
  color: var(--text);
  font-size: 13px;
  max-width: 260px;
}
.trigger:hover {
  border-color: var(--primary);
}
.label {
  font-weight: 500;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 160px;
}
.count {
  font-size: 11px;
  color: var(--text-soft);
  background: var(--bg-soft);
  padding: 0 6px;
  border-radius: 4px;
}
.open {
  transform: rotate(180deg);
  transition: transform 0.15s;
}
.menu {
  position: absolute;
  top: calc(100% + 6px);
  left: 0;
  z-index: 50;
  min-width: 260px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 10px;
  box-shadow: var(--shadow);
  padding: 6px;
}
.menu-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 8px;
  font-size: 11px;
  color: var(--text-soft);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
}
.item:hover {
  background: var(--bg-soft);
}
.item.active {
  background: var(--primary-soft);
  color: var(--primary);
}
.item .name {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.meta {
  font-size: 11px;
  color: var(--text-soft);
}
.empty,
.error {
  padding: 14px;
  text-align: center;
  font-size: 12px;
  color: var(--text-soft);
}
.error {
  color: var(--danger);
}
.menu-footer {
  border-top: 1px solid var(--border);
  padding: 4px;
  margin-top: 4px;
}
.link {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: transparent;
  border: none;
  color: var(--primary);
  font-size: 12px;
  padding: 4px 6px;
  border-radius: 4px;
  cursor: pointer;
}
.link:hover {
  background: var(--primary-soft);
}
</style>
