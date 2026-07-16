<script setup lang="ts">
import { MessageCircle } from 'lucide-vue-next'
import { invoke } from '@tauri-apps/api/core'
import { isTauri } from '@/utils'

const emit = defineEmits<{
  (e: 'click'): void
}>()

// 区分单击与拖拽：mousedown 到 mouseup 之间位移 < 5px 才算 click
let downX = 0
let downY = 0

function onMouseDown(e: MouseEvent) {
  if (e.button !== 0) return
  downX = e.clientX
  downY = e.clientY
  // 在 Tauri 环境立刻调用原生拖拽命令：
  //  - 拖动时由 OS 接管窗口位置，移动 > 5px 时不算 click
  //  - 不动时 mouseup 仍会触发 -> 走 click 分支展开面板
  if (isTauri()) {
    invoke('start_window_drag').catch((err) =>
      console.warn('start_window_drag failed:', err),
    )
  }
}

function onMouseUp(e: MouseEvent) {
  const dx = Math.abs(e.clientX - downX)
  const dy = Math.abs(e.clientY - downY)
  if (dx <= 5 && dy <= 5) emit('click')
}
</script>

<template>
  <div class="ball" @mousedown="onMouseDown" @mouseup="onMouseUp">
    <div class="ring"></div>
    <div class="inner">
      <MessageCircle :size="22" />
    </div>
  </div>
</template>

<style scoped>
.ball {
  position: fixed;
  right: 12px;
  bottom: 12px;
  width: 56px;
  height: 56px;
  border-radius: 50%;
  cursor: pointer;
  z-index: 999;
  transition: transform 0.2s ease;
  -webkit-user-select: none;
  user-select: none;
}
.ball:hover {
  transform: scale(1.08);
}
.ring {
  position: absolute;
  inset: 0;
  border-radius: 50%;
  background: var(--primary);
  opacity: 0.25;
  animation: pulse 2.4s ease-in-out infinite;
}
.inner {
  position: absolute;
  inset: 6px;
  border-radius: 50%;
  background: var(--primary);
  color: var(--primary-fg);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: var(--shadow);
}
@keyframes pulse {
  0%, 100% { transform: scale(1); opacity: 0.25; }
  50% { transform: scale(1.18); opacity: 0.05; }
}
</style>
