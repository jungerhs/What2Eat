/**
 * Auth store：管理 token/user/role。localStorage 持久化。
 */
import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { apiFetch, configureApi } from '@/api/client'

const STORAGE_KEY = 'c9_auth'

function loadFromStorage() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    return JSON.parse(raw)
  } catch (_) {
    return null
  }
}

function persist(state) {
  try {
    if (state.token && state.userId) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
    } else {
      localStorage.removeItem(STORAGE_KEY)
    }
  } catch (_) { /* ignore */ }
}

export const useAuthStore = defineStore('auth', () => {
  const initial = loadFromStorage() || {}
  const token = ref(initial.token || null)
  const userId = ref(initial.userId || null)
  const username = ref(initial.username || '')
  const role = ref(initial.role || 'user')

  const isLoggedIn = computed(() => Boolean(token.value && userId.value))
  const isAdmin = computed(() => role.value === 'admin')

  function setSession({ token: t, user }) {
    token.value = t
    userId.value = user.id
    username.value = user.username
    role.value = user.role || 'user'
    persist({ token: token.value, userId: userId.value, username: username.value, role: role.value })
  }

  function clearSession() {
    token.value = null
    userId.value = null
    username.value = ''
    role.value = 'user'
    persist({})
  }

  async function register(username_, password) {
    const data = await apiFetch('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ username: username_, password }),
    })
    setSession(data)
    return data
  }

  async function login(username_, password) {
    const data = await apiFetch('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username: username_, password }),
    })
    setSession(data)
    return data
  }

  async function fetchMe() {
    if (!token.value) return null
    try {
      const me = await apiFetch('/auth/me')
      role.value = me.role || 'user'
      username.value = me.username || username.value
      userId.value = me.id || userId.value
      persist({
        token: token.value, userId: userId.value,
        username: username.value, role: role.value,
      })
      return me
    } catch (_) {
      clearSession()
      return null
    }
  }

  function logout() {
    clearSession()
  }

  function getToken() { return token.value }

  function bindApi() {
    configureApi({
      getToken,
      onUnauthorized: () => clearSession(),
    })
  }
  bindApi()

  return {
    token, userId, username, role,
    isLoggedIn, isAdmin,
    register, login, fetchMe, logout, getToken,
    setSession, clearSession,
  }
})
