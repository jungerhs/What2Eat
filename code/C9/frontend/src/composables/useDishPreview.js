/**
 * 菜名图片悬停预览：单例浮层，document 事件委托。
 * 安装一次 (installDishPreview) 后所有页面均可生效。
 */
let installed = false
let currentDish = null
let currentImg = null
let previewEl = null

const PAD = 12

function positionPreview(x, y) {
  if (!previewEl) return
  const W = previewEl.offsetWidth || 240
  const H = previewEl.offsetHeight || 200
  let nx = x + 16
  let ny = y + 16
  if (nx + W + PAD > window.innerWidth) nx = x - W - 16
  if (ny + H + PAD > window.innerHeight) ny = y - H - 16
  if (nx < PAD) nx = PAD
  if (ny < PAD) ny = PAD
  previewEl.style.left = nx + 'px'
  previewEl.style.top = ny + 'px'
}

function showPreview(src, dish, x, y) {
  if (!previewEl) return
  if (currentDish !== dish) {
    currentDish = dish
    previewEl.innerHTML = '<div class="pv-loading">加载中…</div>'
  }
  previewEl.classList.add('visible')
  positionPreview(x, y)
  const existing = previewEl.querySelector('img')
  if (existing && existing.dataset.src === src && existing.complete && existing.naturalWidth > 0) {
    return
  }
  const img = new Image()
  img.dataset.src = src
  img.alt = dish
  currentImg = img
  img.onload = () => {
    if (currentDish === dish) {
      previewEl.innerHTML = ''
      previewEl.appendChild(img)
      const cap = document.createElement('div')
      cap.className = 'pv-caption'
      cap.textContent = dish
      previewEl.appendChild(cap)
    }
  }
  img.onerror = () => {
    if (currentDish === dish) {
      previewEl.innerHTML = `<div class="pv-error">图片加载失败<br>${escapeText(dish)}</div>`
    }
  }
  img.src = src
}

function escapeText(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[c])
}

function hidePreview() {
  if (!previewEl) return
  previewEl.classList.remove('visible')
  currentDish = null
  document.querySelectorAll('.ai-message.dim-bg').forEach(el => el.classList.remove('dim-bg'))
  document.querySelectorAll('a.dish-link.is-previewing').forEach(el => el.classList.remove('is-previewing'))
}

export function installDishPreview() {
  if (installed || typeof window === 'undefined') return
  installed = true
  previewEl = document.createElement('div')
  previewEl.id = 'dish-preview'
  previewEl.innerHTML = '<div class="pv-loading">加载中…</div>'
  document.body.appendChild(previewEl)

  document.addEventListener('mouseover', (e) => {
    const a = e.target.closest('a.dish-link[data-src]')
    if (!a) return
    const msg = a.closest('.ai-message')
    if (msg) {
      document.querySelectorAll('.ai-message.dim-bg').forEach(el => {
        if (el !== msg) el.classList.remove('dim-bg')
      })
      msg.classList.add('dim-bg')
    }
    document.querySelectorAll('a.dish-link.is-previewing').forEach(el => {
      if (el !== a) el.classList.remove('is-previewing')
    })
    a.classList.add('is-previewing')
    showPreview(a.dataset.src, a.dataset.dish, e.clientX, e.clientY)
  })
  document.addEventListener('mousemove', (e) => {
    const a = e.target.closest('a.dish-link[data-src]')
    if (!a) return
    positionPreview(e.clientX, e.clientY)
  })
  document.addEventListener('mouseout', (e) => {
    const a = e.target.closest('a.dish-link[data-src]')
    if (!a) return
    const next = e.relatedTarget
    if (next && a.contains(next)) return
    hidePreview()
  })
  document.addEventListener('click', (e) => {
    const a = e.target.closest('a.dish-link[data-src]')
    if (!a) return
    if (e.ctrlKey || e.metaKey || e.shiftKey) return
    if (!('ontouchstart' in window)) return
    e.preventDefault()
    showPreview(a.dataset.src, a.dataset.dish, e.clientX, e.clientY)
  })
}
