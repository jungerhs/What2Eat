/**
 * 全局 API 配置：dev 走 vite proxy，prod 走同源 '/api'。
 */
export const API_BASE = (() => {
  if (import.meta.env?.VITE_API_BASE) return import.meta.env.VITE_API_BASE
  if (typeof window === 'undefined') return ''
  return window.location.origin // 生产：FastAPI 同源暴露 SPA + /api/*
})()

export const API_PREFIX = '/api'

/**
 * 统一 fetch 封装：
 *  - 自动注入 Bearer
 *  - 401 时清掉本地 token 派发全局事件（router 守卫会重定向到 /auth）
 */
let _tokenGetter = () => null
let _unauthorizedHandler = () => {}

export function configureApi({ getToken, onUnauthorized }) {
  if (typeof getToken === 'function') _tokenGetter = getToken
  if (typeof onUnauthorized === 'function') _unauthorizedHandler = onUnauthorized
}

export async function apiFetch(path, options = {}) {
  const url = path.startsWith('http') ? path : `${API_BASE}${API_PREFIX}${path.startsWith('/') ? path : `/${path}`}`
  const headers = { ...(options.headers || {}) }
  const token = _tokenGetter()
  if (token) headers['Authorization'] = `Bearer ${token}`
  if (options.body && !(options.body instanceof FormData) && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json'
  }
  const res = await fetch(url, { ...options, headers })
  if (res.status === 401) {
    _unauthorizedHandler({ url })
    throw new Error('unauthorized')
  }
  if (!res.ok) {
    let detail
    try {
      detail = await res.json()
    } catch (_) {
      detail = { message: res.statusText }
    }
    const msg = (detail && detail.detail && detail.detail.message) || detail?.message || `HTTP ${res.status}`
    const err = new Error(msg)
    err.detail = detail
    err.status = res.status
    throw err
  }
  // 处理 204
  if (res.status === 204) return null
  return res.json()
}
