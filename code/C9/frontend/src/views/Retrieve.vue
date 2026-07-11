<template>
  <div class="bg-admin bg-grid text-warm-900 relative min-h-screen">
    <div class="relative z-10 max-w-5xl mx-auto px-4 sm:px-6 py-6">
      <!-- NAV -->
      <nav class="glass-strong rounded-2xl px-5 py-3 flex items-center justify-between mb-6">
        <div class="flex items-center gap-3">
          <router-link to="/" class="flex items-center gap-2.5 no-underline shrink-0">
            <span class="text-xl">🥢</span>
            <div class="leading-tight">
              <div class="font-semibold text-base text-warm-900">What2Eat <span class="text-brand-500 font-light">检索测试</span></div>
              <div class="text-[0.68rem] text-warm-800/45 font-mono tracking-wider uppercase">Retrieval Playground</div>
            </div>
          </router-link>
        </div>
        <div class="flex items-center gap-2 sm:gap-3">
          <router-link to="/chat"
            class="hidden sm:flex items-center gap-1.5 text-sm text-warm-800/65 hover:text-brand-500 transition-colors px-3 py-1.5 rounded-lg hover:bg-white/50">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 3v-3z" />
            </svg> 返回聊天
          </router-link>
          <router-link to="/admin"
            class="hidden sm:flex items-center gap-1.5 text-sm text-warm-800/65 hover:text-gold-600 transition-colors px-3 py-1.5 rounded-lg hover:bg-white/50">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
            </svg> 用户管理
          </router-link>
          <span class="hidden sm:inline text-sm text-warm-800/70 font-medium">
            👤 {{ auth.username }}
            <span class="chip chip-category ml-1" style="font-size:0.65rem; padding:1px 7px;">{{ auth.role }}</span>
          </span>
          <button @click="onLogout"
            class="text-xs px-3 py-1.5 rounded-lg bg-white/50 hover:bg-white/80 text-warm-800/70 hover:text-brand-600 transition-colors border border-warm-800/10">
            登出
          </button>
        </div>
      </nav>

      <header class="mb-5">
        <h1 class="text-2xl sm:text-3xl font-bold text-warm-900 tracking-tight">
          检索测试
          <span class="text-brand-500 font-light text-base sm:text-lg ml-2 font-mono">/retrieve</span>
        </h1>
        <p class="text-sm text-warm-800/55 mt-1">输入查询，直接查看路由分析与检索结果（跳过 LLM 生成）。用于调试检索质量与路由策略。</p>
      </header>

      <!-- INPUT -->
      <section class="glass-strong rounded-2xl p-5 mb-5">
        <form @submit.prevent="onSubmit" class="space-y-3">
          <div class="flex flex-col sm:flex-row gap-3">
            <div class="flex-1">
              <input ref="questionInput" v-model="question"
                     type="text" placeholder="输入要测试检索的查询，例如：酸辣土豆丝怎么做？"
                     class="input" autocomplete="off" @keydown="onEnter" />
            </div>
            <div class="flex items-center gap-2">
              <label class="text-xs text-warm-800/55 font-mono whitespace-nowrap">top_k</label>
              <input v-model.number="topK" type="number" min="1" max="50"
                     class="input" style="width: 70px; text-align: center; padding: 0.6rem 0.5rem; font-family: 'JetBrains Mono', monospace;" />
            </div>
            <button type="submit" :disabled="loading"
              class="btn btn-primary shrink-0">
              <svg v-if="loading" class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
              </svg>
              <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-4.35-4.35M11 18a7 7 0 100-14 7 7 0 000 14z" />
              </svg>
              <span>{{ loading ? '检索中…' : '检索' }}</span>
            </button>
          </div>
          <div class="flex flex-wrap items-center gap-2 pt-1">
            <span class="text-[0.7rem] text-warm-800/45 font-mono uppercase tracking-wider mr-1">示例</span>
            <span v-for="q in examples" :key="q"
                  class="example-chip" @click="fillExample(q)">{{ q }}</span>
          </div>
        </form>
      </section>

      <!-- RESULT -->
      <section v-if="resultShown" class="space-y-4">
        <div class="glass-strong rounded-2xl p-5 fade-anim" v-if="result?.analysis">
          <div class="flex items-center justify-between flex-wrap gap-3 mb-4">
            <div class="flex items-center gap-3">
              <h2 class="text-base font-bold text-warm-900">路由分析</h2>
              <span :class="['strategy-badge', strategyBadge.class]" v-html="strategyBadge.html"></span>
            </div>
            <div class="flex items-center gap-4 text-xs font-mono text-warm-800/55">
              <span>耗时 <span class="text-warm-900 font-semibold">{{ formatNum(result.elapsed_ms, 1) }}</span> ms</span>
              <span>文档 <span class="text-warm-900 font-semibold">{{ result.total }}</span> 条</span>
              <span>平均分 <span class="text-warm-900 font-semibold">{{ avgScore }}</span></span>
            </div>
          </div>

          <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
            <div class="bg-white/50 rounded-xl p-3 border border-warm-800/8">
              <div class="text-[0.65rem] font-semibold tracking-wider uppercase text-warm-800/45">查询复杂度</div>
              <div class="flex items-baseline gap-1 mt-1"><span class="text-xl font-bold font-mono text-warm-900">{{ clamp(result.analysis.query_complexity).toFixed(2) }}</span></div>
              <div class="meter mt-2"><div class="meter-fill" :style="{ width: meterPct(result.analysis.query_complexity) + '%' }"></div></div>
            </div>
            <div class="bg-white/50 rounded-xl p-3 border border-warm-800/8">
              <div class="text-[0.65rem] font-semibold tracking-wider uppercase text-warm-800/45">关系密集度</div>
              <div class="flex items-baseline gap-1 mt-1"><span class="text-xl font-bold font-mono text-warm-900">{{ clamp(result.analysis.relationship_intensity).toFixed(2) }}</span></div>
              <div class="meter mt-2"><div class="meter-fill meter-fill-purple" :style="{ width: meterPct(result.analysis.relationship_intensity) + '%' }"></div></div>
            </div>
            <div class="bg-white/50 rounded-xl p-3 border border-warm-800/8">
              <div class="text-[0.65rem] font-semibold tracking-wider uppercase text-warm-800/45">置信度</div>
              <div class="flex items-baseline gap-1 mt-1"><span class="text-xl font-bold font-mono text-warm-900">{{ clamp(result.analysis.confidence).toFixed(2) }}</span></div>
              <div class="meter mt-2"><div class="meter-fill meter-fill-blue" :style="{ width: meterPct(result.analysis.confidence) + '%' }"></div></div>
            </div>
            <div class="bg-white/50 rounded-xl p-3 border border-warm-800/8">
              <div class="text-[0.65rem] font-semibold tracking-wider uppercase text-warm-800/45">实体数 / 推理</div>
              <div class="flex items-baseline gap-2 mt-1">
                <span class="text-xl font-bold font-mono text-warm-900">{{ result.analysis.entity_count }}</span>
                <span :class="['text-xs', result.analysis.reasoning_required ? 'text-brand-500 font-semibold' : 'text-warm-800/45']">
                  {{ result.analysis.reasoning_required ? '· 需推理' : '· 无需推理' }}
                </span>
              </div>
            </div>
          </div>

          <div v-if="result.analysis.reasoning && result.analysis.reasoning.trim()"
               class="bg-white/40 rounded-xl p-3 border border-warm-800/8">
            <div class="text-[0.65rem] font-semibold tracking-wider uppercase text-warm-800/45 mb-1">推荐理由</div>
            <div class="text-sm text-warm-800/75 leading-relaxed">{{ result.analysis.reasoning }}</div>
          </div>
        </div>

        <div v-if="errorMsg" class="glass rounded-2xl p-4 border-red-300/40">
          <div class="flex items-start gap-3">
            <svg class="w-5 h-5 text-red-500 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <div>
              <div class="text-sm font-semibold text-red-700">检索失败</div>
              <div class="text-xs text-red-600/80 mt-1 font-mono break-all">{{ errorMsg }}</div>
            </div>
          </div>
        </div>

        <div>
          <div class="flex items-center justify-between mb-3">
            <h2 class="text-base font-bold text-warm-900">检索结果 <span class="text-warm-800/45 font-normal text-sm ml-1">({{ (result?.docs || []).length }})</span></h2>
            <div class="flex items-center gap-2">
              <button @click="expandAll(true)" class="text-xs text-warm-800/55 hover:text-brand-500 transition-colors px-2 py-1">全部展开</button>
              <span class="text-warm-800/20">·</span>
              <button @click="expandAll(false)" class="text-xs text-warm-800/55 hover:text-brand-500 transition-colors px-2 py-1">全部收起</button>
            </div>
          </div>
          <div v-if="loading" class="space-y-2">
            <div v-for="i in 4" :key="i" class="doc-card"><div class="px-4 py-3"><div class="skeleton h-5 w-1/3 mb-2"></div><div class="skeleton h-3 w-2/3"></div></div></div>
          </div>
          <div v-else-if="docs.length === 0" class="empty-state">
            <div class="text-4xl mb-2 opacity-40">🔍</div>
            <div class="text-sm">没有检索到任何文档</div>
          </div>
          <div v-else class="space-y-2">
            <div v-for="(d, i) in docs" :key="i"
                 :class="['doc-card row-anim', expanded[i] ? 'expanded' : '']"
                 :style="{ animationDelay: `${i * 0.04}s` }">
              <div class="doc-head px-4 py-3 flex items-center gap-3" @click="toggleDoc(i)">
                <span :class="['shrink-0 w-7 h-7 rounded-lg flex items-center justify-center text-xs font-bold font-mono', rankClass(i + 1)]">{{ i + 1 }}</span>
                <div class="flex-1 min-w-0">
                  <div class="flex items-center gap-2 flex-wrap">
                    <span class="font-semibold text-warm-900 text-sm truncate">{{ d.recipe_name || '(未命名)' }}</span>
                    <span v-if="nodeTypeChip(d.node_type)" :class="['chip', nodeTypeChip(d.node_type).cls]">{{ nodeTypeChip(d.node_type).label }}</span>
                    <span v-if="d.category" class="chip chip-category">{{ d.category }}</span>
                    <span v-if="d.cuisine_type" class="chip chip-other">{{ d.cuisine_type }}</span>
                  </div>
                  <div class="text-xs text-warm-800/45 mt-0.5 truncate font-mono">{{ truncate(d.content) }}</div>
                </div>
                <div class="flex items-center gap-2 shrink-0">
                  <div class="score-bar"><div class="score-bar-fill" :style="{ width: scorePct(d.score) + '%' }"></div></div>
                  <span class="text-xs font-mono font-semibold text-warm-900 w-12 text-right">{{ formatNum(d.score, 3) }}</span>
                  <svg :class="['w-4 h-4 text-warm-800/40 transition-transform', expanded[i] ? 'rotate-180' : '']" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
              </div>
              <div class="doc-body">
                <div class="px-4 pb-4 pt-1 space-y-3">
                  <div class="flex flex-wrap items-center gap-1.5">
                    <span v-if="d.search_method" class="chip chip-other">method: {{ d.search_method }}</span>
                    <span v-if="d.source" class="chip chip-other">source: {{ d.source }}</span>
                    <span v-if="d.search_type" class="chip chip-other">search_type: {{ d.search_type }}</span>
                    <span v-if="d.node_id" class="chip chip-other font-mono">id: {{ String(d.node_id).slice(0, 16) }}</span>
                  </div>
                  <div>
                    <div class="text-[0.65rem] font-semibold tracking-wider uppercase text-warm-800/45 mb-1.5">内容 (page_content)</div>
                    <div class="doc-content">{{ d.content }}</div>
                  </div>
                  <div v-if="extraMeta(d).length">
                    <div class="text-[0.65rem] font-semibold tracking-wider uppercase text-warm-800/45 mb-1.5">其他元数据</div>
                    <table class="kv-table">
                      <tr v-for="row in extraMeta(d)" :key="row.k"><td>{{ row.k }}</td><td>{{ row.v }}</td></tr>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section v-else class="glass rounded-2xl p-10 text-center">
        <div class="text-5xl mb-3 opacity-30">🔬</div>
        <div class="text-base font-semibold text-warm-800/60">输入查询开始检索测试</div>
        <div class="text-xs text-warm-800/40 mt-2">结果会显示路由策略、检索耗时、每条文档的分数与完整内容</div>
      </section>

      <footer class="text-center text-xs text-warm-800/35 mt-6 font-mono">
        What2Eat Retrieve Playground · v1.0 · 仅 admin 可访问 · 不经过 LLM 生成
      </footer>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { apiFetch } from '@/api/client'
import { toast } from '@/composables/useToast'
import { formatVal, escapeHtml } from '@/utils'

const router = useRouter()
const auth = useAuthStore()

const question = ref('')
const topK = ref(10)
const questionInput = ref(null)
const loading = ref(false)
const result = ref(null)
const errorMsg = ref('')
const expanded = ref({})

const resultShown = computed(() => result.value !== null || errorMsg.value)

const examples = [
  '酸辣土豆丝怎么做？',
  '哪些菜适合给小孩吃？',
  '鸡蛋和番茄能做什么菜？',
  '红烧肉和糖醋排骨的做法有什么区别？',
  '推荐一道简单的早餐',
]

const docs = computed(() => result.value?.docs || [])

const STRATEGY_META = {
  hybrid_traditional: { cls: 'str-hybrid',   label: '混合检索', icon: '🔍' },
  graph_rag:          { cls: 'str-graph',    label: '图RAG',    icon: '🕸️' },
  combined:           { cls: 'str-combined', label: '组合',     icon: '🔄' },
}

const strategyBadge = computed(() => {
  const a = result.value?.analysis
  const key = a?.recommended_strategy || 'unknown'
  const meta = STRATEGY_META[key] || { cls: 'str-unknown', label: key || '未知', icon: '❓' }
  return {
    class: meta.cls,
    html: `<span>${meta.icon}</span><span>${meta.label}</span><span class="font-mono text-[0.7rem] opacity-70">${escapeHtml(key)}</span>`,
  }
})

const avgScore = computed(() => {
  const list = docs.value
  if (!list.length) return '--'
  const s = list.map(d => d.score).filter(x => typeof x === 'number')
  if (!s.length) return '--'
  return (s.reduce((a, b) => a + b, 0) / s.length).toFixed(3)
})

function clamp(v) { return Math.max(0, Math.min(1, v || 0)) }
function meterPct(v) { return (clamp(v) * 100).toFixed(1) }
function scorePct(v) { return meterPct(v) }
function formatNum(v, digits = 1) { return typeof v === 'number' ? v.toFixed(digits) : '--' }
function truncate(s) {
  const t = (s || '').slice(0, 140).replace(/\s+/g, ' ')
  return t + ((s || '').length > 140 ? '…' : '')
}
function rankClass(rank) {
  if (rank === 1) return 'bg-brand-500 text-white'
  if (rank <= 3) return 'bg-brand-100 text-brand-600'
  return 'bg-warm-100 text-warm-800/60'
}
function nodeTypeChip(t) {
  if (!t) return null
  const map = { recipe: { cls: 'chip-recipe', label: t }, ingredient: { cls: 'chip-ingredient', label: t }, step: { cls: 'chip-step', label: t } }
  return map[t] || { cls: 'chip-other', label: t }
}
function extraMeta(d) {
  const hidden = new Set(['recipe_name','node_type','node_id','category','cuisine_type','search_method','search_type','source','final_score','relevance_score','score'])
  return Object.entries(d.metadata || {}).filter(([k]) => !hidden.has(k)).map(([k, v]) => ({ k, v: formatVal(v) }))
}

function toggleDoc(i) {
  expanded.value = { ...expanded.value, [i]: !expanded.value[i] }
}
function expandAll(open) {
  const next = {}
  docs.value.forEach((_, i) => { next[i] = open })
  expanded.value = next
}
function fillExample(q) {
  question.value = q
  questionInput.value?.focus()
}
function onEnter(e) {
  if (e.key === 'Enter') {
    e.preventDefault()
    onSubmit()
  }
}

async function onSubmit() {
  const q = question.value.trim()
  if (!q) return
  loading.value = true
  result.value = null
  errorMsg.value = ''
  expanded.value = {}
  try {
    const data = await apiFetch('/admin/retrieve', {
      method: 'POST', body: JSON.stringify({ question: q, top_k: topK.value }),
    })
    result.value = data
    errorMsg.value = data.error || ''
    if (data.docs && data.docs.length) {
      // 默认展开第一条
      const next = {}; next[0] = true
      expanded.value = next
    }
  } catch (err) {
    errorMsg.value = err.message || '请求失败'
    toast(err.message, 'error')
  } finally {
    loading.value = false
  }
}

function onLogout() { auth.logout(); router.replace('/auth') }

onMounted(() => {
  questionInput.value?.focus()
})
</script>
