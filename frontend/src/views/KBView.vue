<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import {
  Plus,
  Database,
  Trash2,
  Edit3,
  FileText,
  X,
  Search,
  RefreshCw,
  ArrowLeft,
} from 'lucide-vue-next'
import { useKnowledgeBaseStore } from '@/stores/knowledgeBase'
import { useSettingsStore } from '@/stores/settings'
import { kbApi, docApi } from '@/api/client'
import { formatDate } from '@/utils'
import type { KnowledgeBase } from '@/types'

const store = useKnowledgeBaseStore()
const settings = useSettingsStore()
const router = useRouter()

function backToChat() {
  router.push('/')
}

const showCreate = ref(false)
const showEdit = ref(false)
const showAdvanced = ref(false)
const formName = ref('')
const formDesc = ref('')
const formChunkSize = ref(600)
const formChunkOverlap = ref(120)
const formEmbeddingModel = ref('')
const formEmbeddingDim = ref(1024)
const editingId = ref<number | null>(null)
const search = ref('')
const statsMap = ref<Record<number, any>>({})
const submitting = ref(false)
const deletingId = ref<number | null>(null)

const formValid = computed(
  () =>
    !!formName.value.trim() &&
    formChunkSize.value >= 64 &&
    formChunkOverlap.value >= 0 &&
    formChunkOverlap.value < formChunkSize.value,
)

const filtered = computed(() => {
  const q = search.value.trim().toLowerCase()
  if (!q) return store.items
  return store.items.filter(
    (k) => k.name.toLowerCase().includes(q) || (k.description || '').toLowerCase().includes(q),
  )
})

function openCreate() {
  formName.value = ''
  formDesc.value = ''
  formChunkSize.value = 600
  formChunkOverlap.value = 120
  formEmbeddingModel.value = ''
  formEmbeddingDim.value = 1024
  showAdvanced.value = false
  editingId.value = null
  showCreate.value = true
}

function openEdit(kb: KnowledgeBase) {
  formName.value = kb.name
  formDesc.value = kb.description || ''
  formChunkSize.value = kb.chunk_size
  formChunkOverlap.value = kb.chunk_overlap
  formEmbeddingModel.value = kb.embedding_model
  formEmbeddingDim.value = kb.embedding_dim
  showAdvanced.value = false
  editingId.value = kb.id
  showEdit.value = true
}

async function submitCreate() {
  if (!formValid.value || submitting.value) return
  submitting.value = true
  const body: any = {
    name: formName.value.trim(),
    description: formDesc.value.trim() || undefined,
  }
  if (formChunkSize.value) body.chunk_size = formChunkSize.value
  body.chunk_overlap = formChunkOverlap.value
  if (formEmbeddingModel.value.trim()) body.embedding_model = formEmbeddingModel.value.trim()
  if (formEmbeddingDim.value) body.embedding_dim = formEmbeddingDim.value
  try {
    const kb = await store.create(body)
    showCreate.value = false
    settings.setKb(kb.id)
    await refreshStats(kb.id)
  } catch {
    // store.error is rendered below the toolbar and in the modal
  } finally {
    submitting.value = false
  }
}

async function submitEdit() {
  if (editingId.value == null || !formValid.value || submitting.value) return
  const id = editingId.value
  submitting.value = true
  try {
    await store.update(id, {
      name: formName.value.trim(),
      description: formDesc.value.trim(),
      chunk_size: formChunkSize.value,
      chunk_overlap: formChunkOverlap.value,
      embedding_model: formEmbeddingModel.value.trim() || undefined,
      embedding_dim: formEmbeddingDim.value,
    })
    showEdit.value = false
    await refreshStats(id)
  } catch {
    // store.error is rendered below the toolbar and in the modal
  } finally {
    submitting.value = false
  }
}

async function removeKb(kb: KnowledgeBase) {
  if (deletingId.value !== kb.id) {
    deletingId.value = kb.id
    setTimeout(() => {
      if (deletingId.value === kb.id) deletingId.value = null
    }, 3000)
    return
  }
  deletingId.value = kb.id
  try {
    await store.remove(kb.id)
    delete statsMap.value[kb.id]
    if (settings.currentKbId === kb.id) settings.setKb(null)
  } catch {
    // store.error is rendered below the toolbar
  } finally {
    deletingId.value = null
  }
}

async function refreshStats(id: number) {
  try {
    statsMap.value[id] = await kbApi.stats(id)
  } catch {
    statsMap.value[id] = null
  }
}

async function refreshAll() {
  await store.fetch()
  for (const kb of store.items) {
    refreshStats(kb.id)
  }
}

function gotoDocs(kb: KnowledgeBase) {
  router.push(`/docs/${kb.id}`)
}

function useAndChat(kb: KnowledgeBase) {
  settings.setKb(kb.id)
  router.push('/')
}

onMounted(() => {
  refreshAll()
})
</script>

<template>
  <div class="kb-view">
    <header>
      <button class="back" @click="backToChat" title="返回聊天"><ArrowLeft :size="16" /></button>
      <div class="title">
        <h2><Database :size="18" /> 知识库管理</h2>
        <p>每个知识库对应一个独立的向量与 BM25 索引</p>
      </div>
      <div class="actions">
        <button class="ghost" @click="refreshAll"><RefreshCw :size="14" /> 刷新</button>
        <button class="primary" @click="openCreate"><Plus :size="14" /> 新建</button>
      </div>
    </header>

    <div class="toolbar">
      <div class="search">
        <Search :size="14" />
        <input v-model="search" placeholder="搜索知识库…" />
      </div>
    </div>

    <div v-if="store.error" class="api-error">{{ store.error }}</div>

    <div class="grid scrollbar">
      <div v-if="store.loading" class="empty">加载中…</div>
      <div v-else-if="filtered.length === 0" class="empty">
        <Database :size="32" />
        <p>{{ store.items.length === 0 ? '还没有知识库，新建一个来开始' : '没有匹配的知识库' }}</p>
      </div>
      <div v-for="kb in filtered" :key="kb.id" class="card" :class="{ active: settings.currentKbId === kb.id }">
        <div class="card-header">
          <div class="name">
            <Database :size="16" />
            <span>{{ kb.name }}</span>
            <span v-if="settings.currentKbId === kb.id" class="badge">当前</span>
          </div>
          <div class="card-actions">
            <button class="icon" @click="openEdit(kb)" title="编辑"><Edit3 :size="13" /></button>
            <button
            class="icon danger"
            :class="{ confirming: deletingId === kb.id }"
            :disabled="deletingId !== null && deletingId !== kb.id"
            @click="removeKb(kb)"
            :title="deletingId === kb.id ? '再次点击确认删除' : '删除'"
          >
            <template v-if="deletingId === kb.id">确定？</template>
            <Trash2 v-else :size="13" />
          </button>
          </div>
        </div>
        <div class="desc">{{ kb.description || '—' }}</div>
        <div class="stats">
          <span><FileText :size="12" /> {{ statsMap[kb.id]?.doc_ready ?? '?' }} / {{ kb.doc_count }} docs</span>
          <span>{{ statsMap[kb.id]?.chroma_chunks ?? kb.chunk_count }} chunks</span>
          <span class="muted">{{ formatDate(kb.updated_at) }}</span>
        </div>
        <div class="card-footer">
          <button class="link" @click="useAndChat(kb)">聊天</button>
          <button class="link" @click="gotoDocs(kb)">管理文档</button>
        </div>
      </div>
    </div>

    <!-- Create Modal -->
    <div v-if="showCreate" class="modal-mask" @click.self="showCreate = false">
      <div class="modal">
        <header><h3>新建知识库</h3><button class="icon" @click="showCreate = false"><X :size="14" /></button></header>
        <div class="field">
          <label>名称</label>
          <input v-model="formName" placeholder="例如：产品文档" autofocus />
        </div>
        <div class="field">
          <label>描述（可选）</label>
          <textarea v-model="formDesc" rows="3" placeholder="用途说明"></textarea>
        </div>
        <div class="advanced-toggle">
          <button class="link" @click="showAdvanced = !showAdvanced">
            {{ showAdvanced ? '收起' : '高级选项' }}（分块 / Embedding）
          </button>
        </div>
        <div v-if="showAdvanced" class="advanced">
          <div class="field-row">
            <div class="field">
              <label>chunk_size</label>
              <input type="number" v-model.number="formChunkSize" min="64" max="4000" />
            </div>
            <div class="field">
              <label>chunk_overlap</label>
              <input type="number" v-model.number="formChunkOverlap" min="0" max="2000" />
            </div>
          </div>
          <div class="field-row">
            <div class="field">
              <label>embedding_model</label>
              <input v-model="formEmbeddingModel" placeholder="留空用 .env 默认" />
            </div>
            <div class="field">
              <label>embedding_dim</label>
              <input type="number" v-model.number="formEmbeddingDim" min="64" max="4096" />
            </div>
          </div>
          <p class="hint">这些参数只影响之后新入库的文档；已存在的 chunk 不会重建。</p>
        </div>
        <p v-if="formChunkOverlap >= formChunkSize" class="form-error">chunk_overlap 必须小于 chunk_size</p>
        <p v-if="store.error" class="form-error">{{ store.error }}</p>
        <footer>
          <button class="ghost" :disabled="submitting" @click="showCreate = false">取消</button>
          <button class="primary" :disabled="!formValid || submitting" @click="submitCreate">
            {{ submitting ? '创建中…' : '创建' }}
          </button>
        </footer>
      </div>
    </div>

    <!-- Edit Modal -->
    <div v-if="showEdit" class="modal-mask" @click.self="showEdit = false">
      <div class="modal">
        <header><h3>编辑知识库</h3><button class="icon" @click="showEdit = false"><X :size="14" /></button></header>
        <div class="field">
          <label>名称</label>
          <input v-model="formName" />
        </div>
        <div class="field">
          <label>描述</label>
          <textarea v-model="formDesc" rows="3"></textarea>
        </div>
        <div class="advanced-toggle">
          <button class="link" @click="showAdvanced = !showAdvanced">
            {{ showAdvanced ? '收起' : '高级选项' }}（分块 / Embedding）
          </button>
        </div>
        <div v-if="showAdvanced" class="advanced">
          <div class="field-row">
            <div class="field">
              <label>chunk_size</label>
              <input type="number" v-model.number="formChunkSize" min="64" max="4000" />
            </div>
            <div class="field">
              <label>chunk_overlap</label>
              <input type="number" v-model.number="formChunkOverlap" min="0" max="2000" />
            </div>
          </div>
          <div class="field-row">
            <div class="field">
              <label>embedding_model</label>
              <input v-model="formEmbeddingModel" />
            </div>
            <div class="field">
              <label>embedding_dim</label>
              <input type="number" v-model.number="formEmbeddingDim" min="64" max="4096" />
            </div>
          </div>
          <p class="hint">分块参数只影响之后新入库的文档；知识库已有向量时不能直接修改 Embedding 配置。</p>
        </div>
        <p v-if="formChunkOverlap >= formChunkSize" class="form-error">chunk_overlap 必须小于 chunk_size</p>
        <p v-if="store.error" class="form-error">{{ store.error }}</p>
        <footer>
          <button class="ghost" :disabled="submitting" @click="showEdit = false">取消</button>
          <button class="primary" :disabled="!formValid || submitting" @click="submitEdit">
            {{ submitting ? '保存中…' : '保存' }}
          </button>
        </footer>
      </div>
    </div>
  </div>
</template>

<style scoped>
.kb-view {
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
  justify-content: space-between;
  padding: 14px 18px;
  border-bottom: 1px solid var(--border);
  gap: 8px;
}
.back {
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
.back:hover { background: var(--bg-soft); color: var(--text); }
.title h2 {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 0;
  font-size: 16px;
}
.title p {
  margin: 4px 0 0;
  font-size: 12px;
  color: var(--text-soft);
}
.actions {
  display: flex;
  gap: 8px;
}
button {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border-radius: 8px;
  font-size: 13px;
  cursor: pointer;
  border: 1px solid transparent;
}
button.primary {
  background: var(--primary);
  color: var(--primary-fg);
}
button.primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
button.ghost {
  background: transparent;
  border-color: var(--border);
  color: var(--text);
}
button.ghost:hover {
  background: var(--bg-soft);
}
button.icon {
  background: transparent;
  border: none;
  padding: 4px;
  color: var(--text-soft);
}
button.icon:hover {
  background: var(--bg-soft);
  color: var(--text);
}
button.icon.danger:hover {
  color: var(--danger);
}
button.icon.danger.confirming {
  background: var(--danger);
  color: #fff;
  padding: 2px 6px;
  font-size: 11px;
}
.toolbar {
  padding: 12px 18px;
  border-bottom: 1px solid var(--border);
}
.search {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  background: var(--bg-soft);
  border: 1px solid var(--border);
  border-radius: 8px;
  max-width: 320px;
}
.search input {
  flex: 1;
  border: none;
  outline: none;
  background: transparent;
  color: var(--text);
  font-size: 13px;
}
.api-error {
  margin: 10px 18px 0;
  padding: 8px 10px;
  border-radius: 6px;
  background: #fee2e2;
  color: #991b1b;
  font-size: 12px;
}
.form-error {
  margin: 8px 0 0;
  color: var(--danger);
  font-size: 12px;
}
.grid {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 12px;
  align-content: start;
}
.empty {
  grid-column: 1 / -1;
  text-align: center;
  color: var(--text-soft);
  padding: 60px 20px;
}
.empty p { margin-top: 8px; font-size: 13px; }
.card {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px;
  background: var(--bg);
  transition: all 0.15s;
}
.card:hover {
  border-color: var(--primary);
  box-shadow: var(--shadow-soft);
}
.card.active {
  border-color: var(--primary);
  background: var(--primary-soft);
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}
.name {
  display: flex;
  align-items: center;
  gap: 6px;
  font-weight: 500;
  font-size: 14px;
}
.badge {
  font-size: 10px;
  background: var(--primary);
  color: var(--primary-fg);
  padding: 1px 6px;
  border-radius: 4px;
}
.card-actions {
  display: flex;
  gap: 4px;
}
.desc {
  color: var(--text-soft);
  font-size: 12px;
  margin-bottom: 10px;
  min-height: 18px;
}
.stats {
  display: flex;
  gap: 12px;
  font-size: 11px;
  color: var(--text-soft);
  margin-bottom: 12px;
  flex-wrap: wrap;
}
.stats span {
  display: inline-flex;
  align-items: center;
  gap: 3px;
}
.muted { margin-left: auto; }
.card-footer {
  display: flex;
  gap: 6px;
  border-top: 1px solid var(--border);
  padding-top: 10px;
}
.link {
  padding: 4px 10px;
  font-size: 12px;
  background: var(--bg-soft);
  color: var(--text);
  border-radius: 6px;
}
.link:hover {
  background: var(--primary-soft);
  color: var(--primary);
}

.modal-mask {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.4);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
}
.modal {
  width: 420px;
  background: var(--bg);
  border-radius: 12px;
  box-shadow: var(--shadow);
  overflow: hidden;
}
.modal header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 14px 16px;
  border-bottom: 1px solid var(--border);
}
.modal h3 { margin: 0; font-size: 14px; }
.modal .field {
  padding: 12px 16px;
}
.advanced-toggle {
  padding: 0 16px 8px;
  border-top: 1px dashed var(--border);
  padding-top: 10px;
  margin-top: 4px;
}
.advanced {
  background: var(--bg-soft);
  border-radius: 6px;
  margin: 4px 16px 8px;
  padding: 4px 8px;
}
.advanced .hint {
  font-size: 11px;
  color: var(--text-soft);
  margin: 4px 8px 8px;
  line-height: 1.5;
}
.field-row {
  display: flex;
  gap: 8px;
}
.field-row .field {
  flex: 1;
  padding: 6px 8px;
}
.modal label {
  display: block;
  font-size: 12px;
  color: var(--text-soft);
  margin-bottom: 4px;
}
.modal input,
.modal textarea {
  width: 100%;
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px 10px;
  font-size: 13px;
  background: var(--bg);
  color: var(--text);
  outline: none;
  font-family: inherit;
}
.modal input:focus,
.modal textarea:focus {
  border-color: var(--primary);
}
.modal footer {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  padding: 14px 16px;
  border-top: 1px solid var(--border);
}
</style>
