<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { useRoute } from 'vue-router'
import { invoke } from '@tauri-apps/api/core'
import FloatingBall from '@/components/FloatingBall.vue'
import ChatPanel from '@/components/ChatPanel.vue'
import KBView from '@/views/KBView.vue'
import DocsView from '@/views/DocsView.vue'
import SettingsView from '@/views/SettingsView.vue'
import SearchView from '@/views/SearchView.vue'
import { useSettingsStore } from '@/stores/settings'
import { isTauri } from '@/utils'

const settings = useSettingsStore()
const route = useRoute()
const expanded = ref(false)

// 全局拖拽状态：拖文件经过窗口（甚至越过窗口边界）时，置 true。
// 用计数器处理 dragenter/dragleave 嵌套触发；drop 后归零。
const dragDepth = ref(0)
const isDraggingFile = computed(() => dragDepth.value > 0)

function resetDragDepth() {
  dragDepth.value = 0
}

function onWindowDragEnter(e: DragEvent) {
  // 只有带 dataTransfer.files 的才算“拖文件”；普通拖文本/选择文本不阻塞。
  if (!e.dataTransfer || Array.from(e.dataTransfer.types || []).includes('Files')) {
    dragDepth.value += 1
  }
}

function onWindowDragLeave(e: DragEvent) {
  if (!e.dataTransfer || Array.from(e.dataTransfer.types || []).includes('Files')) {
    dragDepth.value = Math.max(0, dragDepth.value - 1)
  }
}

function onWindowDragOver(e: DragEvent) {
  // 需要 preventDefault 才能在大多数浏览器触发 drop；同时刷新深度防止某些场景下 dragleave 漏触发。
  if (e.dataTransfer && Array.from(e.dataTransfer.types || []).includes('Files')) {
    e.preventDefault()
  }
}

function onWindowDrop(e: DragEvent) {
  if (e.dataTransfer && Array.from(e.dataTransfer.types || []).includes('Files')) {
    e.preventDefault()
    resetDragDepth()
  }
}

let unlistenShow: (() => void) | null = null
let unlistenBlur: (() => void) | null = null

async function setMode(mode: 'ball' | 'panel') {
  if (!isTauri()) return
  try {
    await invoke('set_window_mode', { mode })
  } catch (e) {
    console.warn('set_window_mode failed:', e)
  }
}

async function showPanel() {
  expanded.value = true
  await setMode('panel')
  if (isTauri()) {
    const { getCurrentWindow } = await import('@tauri-apps/api/window')
    try {
      await getCurrentWindow().show()
      await getCurrentWindow().setFocus()
    } catch (e) {
      console.warn('show failed:', e)
    }
  }
}

async function closePanel() {
  // 正在拖文件时不要折叠：用户往往把文件从桌面拖向文件选择 dropzone，
  // 期间鼠标会短暂越过窗口边界触发失焦，导致浮窗塌掉，破坏交互。
  if (isDraggingFile.value) return
  await setMode('ball')
  expanded.value = false
}

onMounted(async () => {
  // 监听窗口级拖拽事件（独立于 isTauri，普通浏览器也照样可用，便于纯 web 调试）
  window.addEventListener('dragenter', onWindowDragEnter)
  window.addEventListener('dragleave', onWindowDragLeave)
  window.addEventListener('dragover', onWindowDragOver)
  window.addEventListener('drop', onWindowDrop)
  // 拖拽被中断（Esc、或拖出浏览器到非 drop 目标），收尾 reset
  window.addEventListener('dragend', resetDragDepth)

  if (!isTauri()) return
  // 监听 Rust 发来的事件：⌘⇧Space / 托盘显示时通知前端展开面板
  try {
    const { listen } = await import('@tauri-apps/api/event')
    listen('rag://show', async () => {
      await showPanel()
    }).then((u) => (unlistenShow = u))
    // 窗口失焦 → 折叠成浮球（用户点击窗口外部时）
    listen('rag://blur', async () => {
      await closePanel()
    }).then((u) => (unlistenBlur = u))
  } catch (e) {
    console.warn('event listen failed:', e)
  }
  // 启动时先把窗口切到 ball 模式（如果之前是展开态）
  await setMode('ball')
})

onUnmounted(() => {
  window.removeEventListener('dragenter', onWindowDragEnter)
  window.removeEventListener('dragleave', onWindowDragLeave)
  window.removeEventListener('dragover', onWindowDragOver)
  window.removeEventListener('drop', onWindowDrop)
  window.removeEventListener('dragend', resetDragDepth)
  unlistenShow?.()
  unlistenBlur?.()
})
</script>

<template>
  <div class="app-root" :class="`theme-${settings.theme}`">
    <FloatingBall v-if="!expanded" @click="showPanel" />
    <ChatPanel
      v-else-if="route.name === 'chat'"
      @close="closePanel"
    />
    <!-- view 们自己有返回按钮（router.push('/')），不需要从外层关闭 -->
    <KBView v-else-if="route.name === 'kb'" />
    <DocsView v-else-if="route.name === 'docs'" />
    <SearchView v-else-if="route.name === 'search'" />
    <SettingsView v-else-if="route.name === 'settings'" />
    <!-- 兜底：任何未匹配的路由都回到聊天 -->
    <ChatPanel v-else @close="closePanel" />
  </div>
</template>

<style scoped>
.app-root {
  height: 100vh;
  width: 100vw;
  overflow: hidden;
  background: transparent;
}
</style>
