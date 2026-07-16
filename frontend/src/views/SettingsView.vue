<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { ArrowLeft, Save, RefreshCw } from 'lucide-vue-next'
import { useSettingsStore } from '@/stores/settings'

const settings = useSettingsStore()
const router = useRouter()

function backToChat() {
  router.push('/')
}

const apiBase = ref(settings.apiBase)
const enableWeb = ref(settings.enableWeb)
const theme = ref(settings.theme)
const health = ref<'idle' | 'checking' | 'ok' | 'fail'>('idle')
const healthMsg = ref('')

async function checkHealth() {
  health.value = 'checking'
  healthMsg.value = ''
  try {
    const r = await fetch(`${apiBase.value.replace(/\/$/, '')}/health`)
    const data = await r.json()
    if (data && data.status === 'ok') {
      health.value = 'ok'
      healthMsg.value = `服务可用 · ${data.version || ''}`
    } else {
      health.value = 'fail'
      healthMsg.value = data?.detail || '服务响应异常'
    }
  } catch (e: any) {
    health.value = 'fail'
    healthMsg.value = e?.message || '无法连接'
  }
}

function save() {
  settings.setApiBase(apiBase.value)
  settings.setEnableWeb(enableWeb.value)
  settings.setTheme(theme.value)
}
</script>

<template>
  <div class="settings-view">
    <header>
      <button class="back" @click="backToChat" title="返回聊天"><ArrowLeft :size="16" /></button>
      <h2>设置</h2>
    </header>

    <div class="content scrollbar">
      <section>
        <h3>后端服务</h3>
        <p class="hint">RAG 后端 FastAPI 的地址，默认为本地 <code>http://127.0.0.1:8765</code></p>
        <div class="row">
          <input v-model="apiBase" placeholder="http://127.0.0.1:8765" />
          <button class="ghost" @click="checkHealth">
            <RefreshCw :size="14" /> 测活
          </button>
        </div>
        <div v-if="health !== 'idle'" class="status" :class="health">
          {{ health === 'checking' ? '检测中…' : healthMsg }}
        </div>
      </section>

      <section>
        <h3>联网搜索</h3>
        <label class="toggle">
          <input type="checkbox" v-model="enableWeb" />
          <span>允许联网搜索（Tavily / DuckDuckGo）</span>
        </label>
        <p class="hint">后端需要在 .env 中配置 TAVILY_API_KEY，未配置时自动降级到 DuckDuckGo</p>
      </section>

      <section>
        <h3>外观</h3>
        <div class="row">
          <label class="radio">
            <input type="radio" value="light" v-model="theme" />
            <span>浅色</span>
          </label>
          <label class="radio">
            <input type="radio" value="dark" v-model="theme" />
            <span>深色</span>
          </label>
        </div>
      </section>

      <section>
        <h3>快捷键</h3>
        <div class="shortcuts">
          <div><kbd>Enter</kbd> 发送 · <kbd>Shift</kbd>+<kbd>Enter</kbd> 换行</div>
          <div><kbd>Cmd</kbd>/<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Space</kbd> 唤起 / 收起（Tauri）</div>
        </div>
      </section>

      <footer>
        <button class="primary" @click="save"><Save :size="14" /> 保存</button>
      </footer>
    </div>
  </div>
</template>

<style scoped>
.settings-view {
  position: fixed;
  inset: 16px;
  background: var(--bg);
  border-radius: 16px;
  box-shadow: var(--shadow);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
}
header h2 { margin: 0; font-size: 16px; }
.back {
  background: transparent;
  border: none;
  cursor: pointer;
  color: var(--text-soft);
  padding: 4px;
  border-radius: 4px;
}
.back:hover { background: var(--bg-soft); color: var(--text); }

.content {
  flex: 1;
  overflow-y: auto;
  padding: 18px 24px;
  max-width: 640px;
}
section {
  margin-bottom: 28px;
}
h3 {
  font-size: 13px;
  font-weight: 600;
  margin: 0 0 6px;
  color: var(--text);
}
.hint {
  margin: 0 0 10px;
  font-size: 12px;
  color: var(--text-soft);
}
.hint code {
  background: var(--code-bg, var(--bg-soft));
  padding: 1px 4px;
  border-radius: 4px;
  font-size: 11.5px;
}
.row {
  display: flex;
  gap: 8px;
  align-items: center;
}
.row input[type='text'],
input {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg);
  color: var(--text);
  font-size: 13px;
  outline: none;
}
input:focus {
  border-color: var(--primary);
  box-shadow: 0 0 0 2px var(--primary-soft);
}
.toggle {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  cursor: pointer;
}
.radio {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 13px;
  margin-right: 16px;
}
.status {
  margin-top: 8px;
  padding: 6px 10px;
  border-radius: 6px;
  font-size: 12px;
}
.status.checking {
  background: var(--bg-soft);
  color: var(--text-soft);
}
.status.ok {
  background: #d1fae5;
  color: #065f46;
}
.status.fail {
  background: #fee2e2;
  color: #991b1b;
}
.shortcuts {
  font-size: 12px;
  color: var(--text-soft);
  display: flex;
  flex-direction: column;
  gap: 4px;
}
kbd {
  background: var(--bg-soft);
  border: 1px solid var(--border);
  padding: 1px 6px;
  border-radius: 4px;
  font-family: monospace;
  font-size: 11.5px;
}
footer {
  border-top: 1px solid var(--border);
  padding-top: 18px;
  display: flex;
  justify-content: flex-end;
}
button {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 14px;
  border-radius: 8px;
  border: 1px solid transparent;
  font-size: 13px;
  cursor: pointer;
}
button.primary {
  background: var(--primary);
  color: var(--primary-fg);
}
button.primary:hover { opacity: 0.9; }
button.ghost {
  background: transparent;
  border-color: var(--border);
  color: var(--text);
}
button.ghost:hover { background: var(--bg-soft); }
</style>
