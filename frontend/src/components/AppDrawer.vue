<script setup lang="ts">
/**
 * 抽屉薄封装：统一 naive-ui n-drawer 的 props + 主题色，
 * 避免在每个使用点（MessageBubble 引用详情 / DocsView 切片列表）
 * 重复写一遍。带 footer 槽位用于放置「跳转到文档」等动作按钮。
 */
import { NDrawer, NDrawerContent } from 'naive-ui'

defineProps<{
  open: boolean
  title?: string
  width?: number
}>()
const emit = defineEmits<{
  (e: 'update:open', v: boolean): void
}>()
</script>

<template>
  <NDrawer
    :show="open"
    :width="width ?? 480"
    placement="right"
    @update:show="(v: boolean) => emit('update:open', v)"
  >
    <NDrawerContent
      :title="title"
      :native-scrollbar="false"
      closable
    >
      <template #header>
        <slot name="header">
          <span class="drawer-title">{{ title }}</span>
        </slot>
      </template>
      <div class="drawer-body">
        <slot />
      </div>
      <template #footer>
        <slot name="footer" />
      </template>
    </NDrawerContent>
  </NDrawer>
</template>

<style scoped>
.drawer-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--text);
}
.drawer-body {
  font-size: 14px;
  color: var(--text);
  line-height: 1.6;
}
</style>

