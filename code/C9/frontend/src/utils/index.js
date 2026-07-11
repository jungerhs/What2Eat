/**
 * 通用工具：escapeHtml / formatTime / 现在时间等。
 */
export function escapeHtml(s) {
  if (s == null) return ''
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[c])
}

export function formatTime(iso) {
  if (!iso) return '--'
  try {
    const d = new Date(iso)
    return d.toLocaleString('zh-CN', {
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', hour12: false,
    }).replace(/\//g, '-')
  } catch (_) {
    return iso
  }
}

export function now() {
  return new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
}

export function formatVal(v) {
  if (v == null) return ''
  if (typeof v === 'number') return String(v)
  if (typeof v === 'boolean') return String(v)
  if (Array.isArray(v)) return v.map(formatVal).join(', ')
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}
