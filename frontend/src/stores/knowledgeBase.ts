import { defineStore } from 'pinia'
import { ref } from 'vue'
import { kbApi } from '@/api/client'
import type { KnowledgeBase } from '@/types'

function errorMessage(e: unknown): string {
  const error = e as { data?: { detail?: string }; message?: string }
  return error?.data?.detail || error?.message || String(e)
}

export const useKnowledgeBaseStore = defineStore('knowledgeBase', () => {
  const items = ref<KnowledgeBase[]>([])
  const loading = ref(false)
  const loaded = ref(false)
  const error = ref<string | null>(null)

  async function fetch() {
    loading.value = true
    error.value = null
    try {
      items.value = (await kbApi.list()) as KnowledgeBase[]
    } catch (e: unknown) {
      error.value = errorMessage(e)
    } finally {
      loading.value = false
      loaded.value = true
    }
  }

  async function create(body: { name: string; description?: string }) {
    error.value = null
    try {
      const kb = (await kbApi.create(body)) as KnowledgeBase
      items.value = [kb, ...items.value]
      return kb
    } catch (e: unknown) {
      error.value = errorMessage(e)
      throw e
    }
  }

  async function update(
    id: number,
    body: {
      name?: string
      description?: string
      chunk_size?: number
      chunk_overlap?: number
      embedding_model?: string
      embedding_dim?: number
    },
  ) {
    error.value = null
    try {
      const kb = (await kbApi.update(id, body)) as KnowledgeBase
      items.value = items.value.map((x) => (x.id === id ? kb : x))
      return kb
    } catch (e: unknown) {
      error.value = errorMessage(e)
      throw e
    }
  }

  async function remove(id: number) {
    error.value = null
    try {
      await kbApi.remove(id)
      items.value = items.value.filter((x) => x.id !== id)
    } catch (e: unknown) {
      error.value = errorMessage(e)
      throw e
    }
  }

 function getById(id: number | null): KnowledgeBase | undefined {
   if (id == null) return undefined
   return items.value.find((x) => x.id === id)
 }

 return {
   items,
   loading,
   loaded,
   error,
   fetch,
   create,
   update,
   remove,
   getById,
 }
})
