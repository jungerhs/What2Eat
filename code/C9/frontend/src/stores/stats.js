/**
 * Stats store：知识库与路由计数。每 15s 自动刷新。
 */
import { defineStore } from 'pinia'
import { reactive } from 'vue'
import { apiFetch } from '@/api/client'

export const useStatsStore = defineStore('stats', () => {
  const data = reactive({
    status: 'connecting', // 'ok' | 'initializing' | 'error' | 'connecting'
    totalQueries: 0,
    recipes: 0,
    ingredients: 0,
    steps: 0,
    chunks: 0,
    vectors: 0,
  })
  let timer = null

  async function refreshHealth() {
    try {
      const res = await apiFetch('/health')
      data.status = res.status || 'ok'
    } catch (_) {
      data.status = 'error'
    }
  }

  async function refresh() {
    try {
      const res = await apiFetch('/stats')
      data.totalQueries = res.routing?.total_queries || 0
      const kb = res.knowledge_base || {}
      data.recipes = kb.total_recipes || 0
      data.ingredients = kb.total_ingredients || 0
      data.steps = kb.total_cooking_steps || 0
      data.chunks = kb.total_chunks || 0
      data.vectors = kb.vector_count || 0
    } catch (_) { /* silent */ }
  }

  function start() {
    if (timer) return
    refreshHealth()
    refresh()
    timer = setInterval(() => {
      refreshHealth()
      refresh()
    }, 30000)
  }
  function stop() {
    if (timer) { clearInterval(timer); timer = null }
  }

  return { data, refresh, refreshHealth, start, stop }
})
