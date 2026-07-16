<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { ArrowLeft, Search, FileText, Loader2 } from 'lucide-vue-next'
import { searchApi } from '@/api/client'
import { useSettingsStore } from '@/stores/settings'
import type { Source } from '@/types'

const router = useRouter()
const settings = useSettingsStore()

const query = ref('')
const topK = ref(5)
const useRerank = ref(true)
const searching = ref(false)
const results = ref<Source[]>([])
const latencyMs = ref<number | null>(null)
const error = ref<string | null>(null)

function back() {
  router.push('/')
}

async function run() {
  if (!query.value.trim() || !settings.currentKbId) return
  searching.value = true
  error.value = null
  results.value = []
  latencyMs.value = null
  try {
    const res = (await searchApi.search({
      query: query.value.trim(),
      knowledge_base_id: settings.currentKbId,
      top_k: topK.value,
      use_rerank: useRerank.value,
    })) as { sources: Source[]; latency_ms: number }
    results.value = res.sources
    latencyMs.value = res.latency_ms
  } catch (e: any) {
    error.value = e?.message || String(e)
  } finally {
    searching.value = false
  }
}

function openSource(s: Source) {
  if (!s.doc_id || !s.kb_id) return
  router.push(`/docs/${s.kb_id}?doc=${s.doc_id}`)
}

const hasKb = computed(() => !!settings.currentKbId)
</script>

<template>
  <div class="search-view">
    <header>
      <button class="back" @click="back" title="返回聊天"><ArrowLeft :size="16" /></button>
      <h2>独立检索</h2>
    </header>

    <div class="content scrollbar">
      <div v-if="!hasKb" class="hint">
        请先在聊天面板顶部选择知识库，再回来检索。
      </div>

      <div v-else class="form">
        <div class="row">
          <textarea
            v-model="query"
            placeholder="输入查询，不走 LLM，直接看检索结果…"
            rows="2"
            @keydown.enter.exact.prevent="run"
          ></textarea>
        </div>
        <div class="opts">
          <label class="opt">
            <span>top_k</span>
            <input type="number" v-model.number="topK" min="1" max="20" />
          </label>
          <label class="opt">
            <input type="checkbox" v-model="useRerank" />
            <span>使用 Rerank</span>
          </label>
          <button class="primary" :disabled="!query.trim() || searching" @click="run">
            <Loader2 v-if="searching" :size="14" class="spin" />
            <Search v-else :size="14" />
            检索
          </button>
        </div>
      </div>

      <div v-if="error" class="error">{{ error }}</div>

      <div v-if="latencyMs != null" class="meta">
        共 {{ results.length }} 条 · {{ latencyMs }} ms
      </div>

      <div class="results">
        <div v-if="searching" class="empty">
          <Loader2 :size="18" class="spin" />
          检索中…
        </div>
        <div v-else-if="results.length === 0 && !error && latencyMs != null" class="empty">
          没有命中结果
        </div>
        <div
          v-for="(s, i) in results"
          :key="i"
          class="card"
          :class="{ clickable: !!s.doc_id }"
          @click="openSource(s)"
        >
          <div class="card-head">
            <span class="rank">#{{ i + 1 }}</span>
            <FileText :size="13" />
            <span class="doc">{{ s.document }}</span>
            <span v-if="s.page != null" class="page">p.{{ s.page }}</span>
            <span class="src-type">{{ s.source_type }}</span>
            <span v-if="s.score != null" class="score">
              {{ Math.round((s.score ?? 0) * 100) }}%
            </span>
          </div>
          <div class="snippet">{{ s.snippet }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.search-view {
  position: fixed; inset: 16px;
  background: var(--bg);
  border-radius: 16px;
  box-shadow: var(--shadow);
  display: flex; flex-direction: column;
  overflow: hidden;
}
header {
  display: flex; align-items: center; gap: 12px;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
}
header h2 { margin: 0; font-size: 16px; }
.back { background: transparent; border: none; cursor: pointer; color: var(--text-soft); padding: 4px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; }
.back:hover { background: var(--bg-soft); color: var(--text); }
.content {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.hint {
  padding: 24px;
  text-align: center;
  color: var(--text-soft);
  font-size: 13px;
}
.form {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.row textarea {
  width: 100%;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px;
  font-size: 14px;
  font-family: inherit;
  background: var(--bg);
  color: var(--text);
  outline: none;
  resize: vertical;
}
.row textarea:focus {
  border-color: var(--primary);
}
.opts {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.opt {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--text-soft);
}
.opt input[type='number'] {
  width: 60px;
  padding: 4px 6px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--bg);
  color: var(--text);
  font-size: 12px;
}
button.primary {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 6px 12px;
  border-radius: 6px;
  border: none;
  background: var(--primary);
  color: var(--primary-fg);
  cursor: pointer;
  font-size: 13px;
}
button.primary:disabled { opacity: 0.5; cursor: not-allowed; }
.error {
  padding: 8px 10px;
  background: #fee2e2;
  color: #991b1b;
  border-radius: 6px;
  font-size: 12px;
}
.meta {
  font-size: 11px;
  color: var(--text-soft);
  padding: 4px 0;
}
.results {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.empty {
  text-align: center;
  padding: 30px;
  color: var(--text-soft);
  font-size: 13px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
}
.spin { animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.card {
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px 12px;
  background: var(--bg);
}
.card.clickable {
  cursor: pointer;
}
.card.clickable:hover {
  border-color: var(--primary);
  background: var(--primary-soft);
}
.card-head {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  margin-bottom: 6px;
  flex-wrap: wrap;
}
.rank {
  background: var(--primary);
  color: var(--primary-fg);
  padding: 1px 6px;
  border-radius: 4px;
  font-size: 11px;
}
.doc {
  font-weight: 500;
  color: var(--text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 200px;
}
.page {
  color: var(--text-soft);
  font-size: 11px;
}
.src-type {
  background: var(--bg-soft);
  padding: 1px 6px;
  border-radius: 4px;
  font-size: 10px;
  color: var(--text-soft);
  text-transform: uppercase;
}
.score {
  margin-left: auto;
  color: var(--text-soft);
  font-size: 11px;
}
.snippet {
  font-size: 12px;
  line-height: 1.6;
  color: var(--text-soft);
  max-height: 80px;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 4;
  -webkit-box-orient: vertical;
}
</style>