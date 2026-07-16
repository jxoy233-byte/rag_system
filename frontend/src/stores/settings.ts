import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export type Theme = 'light' | 'dark'

const KEY = 'rag.settings'

interface PersistedSettings {
  theme: Theme
  currentKbId: number | null
  apiBase: string
  enableWeb: boolean
  panelWidth: number
  panelHeight: number
}

function loadInitial(): PersistedSettings {
  const fallback: PersistedSettings = {
    theme: 'light',
    currentKbId: null,
    apiBase: 'http://127.0.0.1:8765',
    enableWeb: true,
    panelWidth: 820,
    panelHeight: 600,
  }
  if (typeof window === 'undefined') return fallback
  try {
    const raw = localStorage.getItem(KEY)
    if (!raw) return fallback
    return { ...fallback, ...(JSON.parse(raw) as PersistedSettings) }
  } catch {
    return fallback
  }
}

export const useSettingsStore = defineStore('settings', () => {
  const init = loadInitial()
  const theme = ref<Theme>(init.theme)
  const currentKbId = ref<number | null>(init.currentKbId)
  const apiBase = ref<string>(init.apiBase)
  const enableWeb = ref<boolean>(init.enableWeb)
  const panelWidth = ref<number>(init.panelWidth)
  const panelHeight = ref<number>(init.panelHeight)

  function persist() {
    if (typeof window === 'undefined') return
    const snapshot: PersistedSettings = {
      theme: theme.value,
      currentKbId: currentKbId.value,
      apiBase: apiBase.value,
      enableWeb: enableWeb.value,
      panelWidth: panelWidth.value,
      panelHeight: panelHeight.value,
    }
    localStorage.setItem(KEY, JSON.stringify(snapshot))
  }

  function toggleTheme() {
    theme.value = theme.value === 'light' ? 'dark' : 'light'
    persist()
  }

  function setTheme(t: Theme) {
    theme.value = t
    persist()
  }

  function setKb(id: number | null) {
    currentKbId.value = id
    persist()
  }

  function setApiBase(v: string) {
    apiBase.value = v.trim()
    persist()
  }

  function setEnableWeb(v: boolean) {
    enableWeb.value = v
    persist()
  }

  function setPanelSize(w: number, h: number) {
    panelWidth.value = Math.max(480, Math.min(1200, w))
    panelHeight.value = Math.max(420, Math.min(1000, h))
    persist()
  }

 const themeClass = computed(() => (theme.value === 'dark' ? 'theme-dark' : 'theme-light'))

 return {
   theme,
   currentKbId,
   apiBase,
   enableWeb,
   panelWidth,
   panelHeight,
   themeClass,
   toggleTheme,
   setTheme,
   setKb,
   setApiBase,
   setEnableWeb,
   setPanelSize,
 }
})
