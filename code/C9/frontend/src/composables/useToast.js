/**
 * 全局 Toast：极简实现，把容器挂到 body 上，组件是空模板（不使用 SFC 便于共享）。
 * 调用方式：toast('登录过期', 'error')
 */
import { reactive } from 'vue'

const state = reactive({ items: [] })
let container = null
let seq = 0

const COLORS = {
  info:    'bg-warm-800/85 text-white',
  error:   'bg-red-500 text-white',
  success: 'bg-emerald-500 text-white',
}

function ensureContainer() {
  if (container) return container
  container = document.createElement('div')
  container.id = 'toast-container'
  container.className = 'fixed top-6 left-1/2 -translate-x-1/2 z-[100] flex flex-col items-center gap-2 pointer-events-none'
  document.body.appendChild(container)
  return container
}

function render() {
  const root = ensureContainer()
  root.innerHTML = ''
  for (const t of state.items) {
    const el = document.createElement('div')
    el.className = `toast-enter pointer-events-auto px-4 py-2 rounded-xl text-sm font-medium shadow-lg ${COLORS[t.type] || COLORS.info}`
    el.textContent = t.message
    root.appendChild(el)
    setTimeout(() => { el.style.opacity = '0'; el.style.transition = 'opacity 0.3s' }, 3500)
    setTimeout(() => { t._removed = true; render() }, 3900)
  }
  state.items = state.items.filter(t => !t._removed)
}

export function toast(message, type = 'info') {
  state.items.push({ id: ++seq, message, type })
  render()
}

export function useToast() {
  return { toast }
}
