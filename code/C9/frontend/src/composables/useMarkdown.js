/**
 * Markdown 渲染与菜名链接：完整保留原 app.html 中的 prelinkDishNames + linkifyTokens 逻辑。
 * 注意：marked/dompurify 通过 npm 安装，构建时进入 bundle，避免外部 CDN 抖动。
 */
import { marked } from 'marked'
import DOMPurify from 'dompurify'

marked.setOptions({
  gfm: true,
  breaks: true,
  smartypants: false,
  pedantic: false,
})

const isWordChar = (c) => /[一-鿿㐀-䶿a-zA-Z0-9]/.test(c || '')

/**
 * 把 [[菜名]] 或裸菜名替换为占位 token。
 */
function prelinkDishNames(text, allDishNames) {
  if (!text || !allDishNames || allDishNames.size === 0) return { text, tokens: [] }
  const names = [...allDishNames].sort((a, b) => b.length - a.length)
  const tokens = []
  let result = ''
  let i = 0
  let tokenIdx = 0
  const N = text.length

  const isInsideBrackets = (pos) => {
    const window = text.slice(Math.max(0, pos - 30), pos)
    const opens = (window.match(/\[\[/g) || []).length
    const closes = (window.match(/\]\]/g) || []).length
    return opens > closes
  }

  while (i < N) {
    if (text[i] === '[' && text[i + 1] === '[') {
      const end = text.indexOf(']]', i + 2)
      if (end !== -1) {
        const inner = text.slice(i + 2, end)
        tokens.push({ idx: tokenIdx++, name: inner, hasBrackets: true })
        result += `\u0001DISH${tokenIdx - 1}\u0002`
        i = end + 2
        continue
      }
    }
    let matched = null
    for (const name of names) {
      if (i + name.length > N) continue
      if (text.slice(i, i + name.length) !== name) continue
      const prev = i > 0 ? text[i - 1] : ''
      const next = i + name.length < N ? text[i + name.length] : ''
      if (isWordChar(prev) || isWordChar(next)) continue
      matched = name
      break
    }
    if (matched) {
      tokens.push({ idx: tokenIdx++, name: matched, hasBrackets: false })
      result += `\u0001DISH${tokenIdx - 1}\u0002`
      i += matched.length
    } else {
      result += text[i]
      i++
    }
  }
  return { text: result, tokens }
}

function linkifyTokens(html, tokens, dishImages) {
  if (!tokens || tokens.length === 0) return html
  return html.replace(/\u0001DISH(\d+)\u0002/g, (m, n) => {
    const idx = parseInt(n, 10)
    const t = tokens[idx]
    if (!t) return m
    const url = dishImages.get(t.name)
    if (url) {
      return `<a class="dish-link" href="${url}" target="_blank" rel="noopener noreferrer" data-dish="${t.name}" data-src="${url}">${t.name}</a>`
    }
    return `<span class="dish-link dish-link--noimg">${t.name}</span>`
  })
}

/**
 * 主入口：传入原文 + dish 索引，返回净化后的 HTML。
 */
export function renderMarkdown(text, dishIndex) {
  if (!text) return ''
  const allNames = dishIndex?.all || new Set()
  const images = dishIndex?.images || new Map()
  try {
    const { text: preText, tokens } = prelinkDishNames(text, allNames)
    const raw = marked.parse(preText)
    const linked = linkifyTokens(raw, tokens, images)
    return DOMPurify.sanitize(linked, {
      ADD_ATTR: ['target', 'rel', 'title', 'data-dish', 'data-src'],
      ADD_TAGS: ['span'],
      FORBID_TAGS: ['style', 'iframe', 'form', 'input'],
      FORBID_ATTR: ['style', 'onerror', 'onload', 'onclick'],
    })
  } catch (_) {
    return `<p>${(text || '').replace(/\n/g, '<br>')}</p>`
  }
}
