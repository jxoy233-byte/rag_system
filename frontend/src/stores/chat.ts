import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { streamChat, convApi } from '@/api/client'
import { useSettingsStore } from '@/stores/settings'
import type { ChatMessage, ChatMeta, Source } from '@/types'

export interface ConversationItem {
  id: number
  knowledge_base_id: number | null
  title: string
  created_at: string
  updated_at: string
}

export const useChatStore = defineStore('chat', () => {
  const messages = ref<ChatMessage[]>([])
  const streaming = ref(false)
  const currentMeta = ref<ChatMeta | null>(null)
  const conversationId = ref<number | null>(null)
  const abortController = ref<AbortController | null>(null)
  const error = ref<string | null>(null)
  // 会话列表（侧栏用）
  const conversations = ref<ConversationItem[]>([])
  const conversationsLoading = ref(false)
  const settings = useSettingsStore()

  function reset() {
    messages.value = []
    currentMeta.value = null
    conversationId.value = null
    error.value = null
  }

  function hardReset() {
    if (streaming.value) abortController.value?.abort()
    reset()
    streaming.value = false
    abortController.value = null
  }

  function removeMessage(index: number) {
    if (streaming.value) return
    messages.value = messages.value.filter((_, i) => i !== index)
  }

  // 从 userIdx 这条 user 消息开始，重新生成它的回答
  // 把从 userIdx 开始的所有消息截掉（保留 user），重新跑 send
  async function regenerateFrom(userIdx: number) {
    if (streaming.value) return
    const userMsg = messages.value[userIdx]
    if (!userMsg || userMsg.role !== 'user') return
    // 截断到 userIdx（含），清掉之后的
    messages.value = messages.value.slice(0, userIdx)
    const question = userMsg.content
    await send({
      question,
      knowledgeBaseId: settings.currentKbId,
      enableWeb: settings.enableWeb,
    })
  }

  async function loadConversation(id: number) {
    const items = (await convApi.messages(id)) as ChatMessage[]
    messages.value = items
    conversationId.value = id
    currentMeta.value = null
  }

  async function fetchConversations(kbId?: number | null) {
    conversationsLoading.value = true
    try {
      conversations.value = (await convApi.list(kbId)) as ConversationItem[]
    } finally {
      conversationsLoading.value = false
    }
  }

  async function renameConversation(id: number, title: string) {
    const updated = (await convApi.update(id, { title })) as ConversationItem
    conversations.value = conversations.value.map((c) => (c.id === id ? updated : c))
  }

  async function deleteConversation(id: number) {
    await convApi.remove(id)
    conversations.value = conversations.value.filter((c) => c.id !== id)
    if (conversationId.value === id) {
      messages.value = []
      currentMeta.value = null
      conversationId.value = null
      error.value = null
    }
  }

 async function abort() {
   abortController.value?.abort()
   abortController.value = null
   streaming.value = false
 }

 async function send(opts: {
   question: string
   knowledgeBaseId?: number | null
   history?: ChatMessage[]
   enableWeb?: boolean
 }) {
   if (!opts.question.trim()) return
   if (streaming.value) await abort()
   streaming.value = true
   error.value = null

   const userMsg: ChatMessage = { role: 'user', content: opts.question }
   const assistantMsg: ChatMessage = { role: 'assistant', content: '', thinking: true }
   messages.value = [...messages.value, userMsg, assistantMsg]

   const controller = new AbortController()
   abortController.value = controller

   try {
     const gen = streamChat(
       {
         question: opts.question,
         knowledge_base_id: opts.knowledgeBaseId ?? null,
         conversation_id: conversationId.value,
         history: (opts.history ?? []).map((h) => ({ role: h.role, content: h.content })),
         enable_web: opts.enableWeb ?? true,
       },
     )
     let collectedSources: Source[] = []
     let pendingMeta: ChatMeta | null = null
     for await (const ev of gen) {
       if (controller.signal.aborted) break
       if (ev.event === 'token') {
         try {
           const data = JSON.parse(ev.data) as { content: string }
           assistantMsg.content = (assistantMsg.content || '') + data.content
           // reactive update: replace last
           messages.value = [...messages.value.slice(0, -1), { ...assistantMsg }]
         } catch {
           /* skip malformed */
         }
       } else if (ev.event === 'sources') {
         try {
           const data = JSON.parse(ev.data) as { sources: Source[] }
           collectedSources = data.sources || []
           assistantMsg.sources = collectedSources
           assistantMsg.thinking = false
           messages.value = [...messages.value.slice(0, -1), { ...assistantMsg }]
         } catch {
           /* skip */
         }
       } else if (ev.event === 'intent') {
         try {
           const data = JSON.parse(ev.data) as { intent: string }
           assistantMsg.intent = data.intent
           messages.value = [...messages.value.slice(0, -1), { ...assistantMsg }]
         } catch { /* skip */ }
       } else if (ev.event === 'meta') {
         try {
           pendingMeta = JSON.parse(ev.data) as ChatMeta
         } catch { /* skip */ }
       } else if (ev.event === 'final') {
         try {
           const data = JSON.parse(ev.data) as { meta: ChatMeta }
           currentMeta.value = data.meta
           if (data.meta.conversation_id) {
             conversationId.value = data.meta.conversation_id
           }
           assistantMsg.latency_ms = data.meta.latency_ms
           assistantMsg.intent = data.meta.intent
         } catch { /* skip */ }
       } else if (ev.event === 'error') {
         try {
           const data = JSON.parse(ev.data) as { message: string }
           assistantMsg.error = data.message
           assistantMsg.thinking = false
           messages.value = [...messages.value.slice(0, -1), { ...assistantMsg }]
           error.value = data.message
         } catch { /* skip */ }
       } else if (ev.event === 'end') {
         assistantMsg.thinking = false
         messages.value = [...messages.value.slice(0, -1), { ...assistantMsg }]
         currentMeta.value = pendingMeta ?? currentMeta.value
       }
     }
  } catch (e: any) {
    if (e?.name !== 'AbortError') {
      error.value = e?.message || String(e)
      assistantMsg.error = error.value ?? undefined
      assistantMsg.thinking = false
      messages.value = [...messages.value.slice(0, -1), { ...assistantMsg }]
    }
   } finally {
     streaming.value = false
     abortController.value = null
   }
 }

 const hasMessages = computed(() => messages.value.length > 0)

 return {
   messages,
   streaming,
   currentMeta,
   conversationId,
   error,
   hasMessages,
   conversations,
   conversationsLoading,
   send,
   abort,
   reset,
   hardReset,
   loadConversation,
   fetchConversations,
   renameConversation,
   deleteConversation,
   removeMessage,
   regenerateFrom,
 }
})
