/**
 * 菜品索引：all = Set(菜名), images = Map(菜名→URL)。
 * 后端不一定实现了 /api/dish-images，404 时静默。
 */
import { defineStore } from 'pinia'
import { reactive } from 'vue'
import { apiFetch } from '@/api/client'

export const useDishIndexStore = defineStore('dishIndex', () => {
  const state = reactive({
    all: new Set(),
    images: new Map(),
    loaded: false,
    loading: false,
  })

  async function load() {
    if (state.loading || state.loaded) return
    state.loading = true
    try {
      const data = await apiFetch('/dish-images')
      state.all = new Set(data.dishes || [])
      state.images = new Map(Object.entries(data.images || {}))
      state.loaded = true
    } catch (_) {
      // 后端若没实现，端点 404，这里静默忽略
      state.loaded = true
    } finally {
      state.loading = false
    }
  }

  return { state, load }
})
