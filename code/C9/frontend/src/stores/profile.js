/**
 * 用户偏好 store：菜系/口味/忌口等。404 时按空对象处理。
 */
import { defineStore } from 'pinia'
import { reactive } from 'vue'
import { apiFetch } from '@/api/client'

export const useProfileStore = defineStore('profile', () => {
  const state = reactive({
    prefs: null,        // 后端 profile payload
    message: '',
    loading: false,
  })

  async function refresh() {
    state.loading = true
    try {
      const data = await apiFetch('/user/profile')
      state.prefs = data.profile || null
      state.message = data.message || ''
    } catch (_) {
      state.prefs = null
    } finally {
      state.loading = false
    }
  }

  function clear() {
    state.prefs = null
    state.message = ''
  }

  return { state, refresh, clear }
})
