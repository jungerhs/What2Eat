<template>
  <div class="min-h-screen text-warm-900 bg-mesh relative">
    <!-- ====== NAV ====== -->
    <nav class="fixed top-3 left-3 right-3 z-50" aria-label="主导航">
      <div class="max-w-7xl mx-auto glass-strong rounded-2xl px-4 sm:px-6 py-3 flex items-center justify-between">
        <router-link to="/" class="flex items-center gap-2.5 no-underline shrink-0" aria-label="What2Eat 首页">
          <span class="text-xl sm:text-2xl">🥢</span>
          <span class="font-semibold text-base sm:text-lg text-warm-900">
            What2Eat <span class="text-brand-500 font-light">尝尝咸淡</span>
          </span>
        </router-link>
        <div class="flex items-center gap-2 sm:gap-3">
          <button @click="sidebarOpen = !sidebarOpen"
            class="lg:hidden p-2 rounded-lg hover:bg-white/50 transition-colors duration-200 cursor-pointer text-warm-800/60"
            aria-label="统计面板" title="统计面板" touch-action="manipulation">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </button>
          <span class="w-2 h-2 rounded-full" :class="dotColor" title="连接状态"></span>
          <span class="hidden sm:inline text-sm text-warm-800/70 font-medium" title="当前登录用户">
            👤 {{ auth.username }}
          </span>
          <router-link v-if="auth.isAdmin" to="/admin"
            class="hidden sm:flex items-center gap-1 text-xs px-2.5 py-1.5 rounded-lg bg-gold-400/15 hover:bg-gold-400/25 text-gold-600 transition-colors duration-200 cursor-pointer border border-gold-400/30"
            title="用户管理后台">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
            </svg> 管理
          </router-link>
          <router-link v-if="auth.isAdmin" to="/retrieve"
            class="hidden sm:flex items-center gap-1 text-xs px-2.5 py-1.5 rounded-lg bg-indigo-50 hover:bg-indigo-100 text-indigo-600 transition-colors duration-200 cursor-pointer border border-indigo-200"
            title="检索测试 playground">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-4.35-4.35M11 18a7 7 0 100-14 7 7 0 000 14z" />
            </svg> 检索测试
          </router-link>
          <button @click="onLogout"
            class="text-xs px-2.5 py-1.5 rounded-lg bg-white/50 hover:bg-white/80 text-warm-800/70 hover:text-brand-600 transition-colors duration-200 cursor-pointer border border-warm-800/10">
            登出
          </button>
          <router-link to="/"
            class="hidden sm:flex items-center gap-1 text-sm text-warm-800/60 hover:text-brand-500 transition-colors duration-200 cursor-pointer">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg> 返回首页
          </router-link>
        </div>
      </div>
    </nav>

    <!-- ====== MAIN LAYOUT ====== -->
    <main class="flex flex-col lg:flex-row gap-4 lg:gap-6 pt-24 pb-4 px-4 sm:px-6 max-w-7xl mx-auto"
          style="height: calc(100vh - 16px);">
      <div class="flex flex-col lg:flex-row gap-4 lg:gap-6 w-full">
        <!-- ====== CHAT ====== -->
        <section class="flex-1 flex flex-col min-w-0">
          <div class="rounded-2xl flex flex-col overflow-hidden bg-white/40"
               style="height: calc(100vh - 112px); min-height: 500px;">
            <!-- Chat header -->
            <div class="px-4 sm:px-6 py-3 border-b border-warm-800/8 flex items-center gap-3 shrink-0">
              <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-400/20 to-gold-400/20 flex items-center justify-center text-lg">🥢</div>
              <div>
                <div class="text-sm font-semibold text-warm-900">What2Eat 尝咸淡</div>
                <div class="text-xs text-warm-800/50 flex items-center gap-1">
                  <span class="w-1.5 h-1.5 rounded-full" :class="dotColor"></span>
                  <span>{{ statusLabel }}</span>
                </div>
              </div>
            </div>

            <!-- Message list -->
            <div id="message-list"
                 class="flex-1 overflow-y-auto overflow-x-hidden px-4 sm:px-6 py-4 space-y-4 overscroll-contain"
                 style="min-height: 0;" ref="listRef">
              <div v-if="messages.length === 0"
                   class="flex flex-col items-center justify-center h-full text-center py-8">
                <div class="text-5xl sm:text-6xl mb-4">🥢</div>
                <h2 class="text-lg sm:text-xl font-semibold mb-2 text-warm-900">欢迎使用 What2Eat 尝咸淡</h2>
                <p class="text-sm sm:text-base text-warm-800/60 mb-6 max-w-md">我是你的 AI 烹饪助手，基于知识图谱为你提供专业的中餐指导</p>
                <div class="flex flex-wrap justify-center gap-2">
                  <button v-for="q in presets" :key="q"
                          class="glass rounded-full px-4 py-2 cursor-pointer hover:bg-white/80 transition-all duration-200 text-xs sm:text-sm text-warm-800/70 font-medium hover:text-brand-600 hover:shadow-md"
                          @click="send(q)">{{ q }}</button>
                </div>
              </div>
              <MessageBubble v-for="m in messages" :key="m.id" :message="m" />
            </div>

            <!-- Input -->
            <div class="px-4 sm:px-6 py-3 border-t border-warm-800/8 bg-white/30 shrink-0">
              <div class="flex items-end gap-2">
                <textarea v-model="input" ref="inputRef"
                          rows="1"
                          placeholder="输入你的烹饪问题…"
                          aria-label="烹饪问题输入"
                          class="flex-1 resize-none bg-white/60 rounded-xl border border-warm-800/10 outline-none text-sm sm:text-base text-warm-900 placeholder-warm-800/40 py-2.5 px-3 max-h-32 focus:border-brand-400/50 focus:bg-white/80 transition-colors"
                          @input="autoResize"
                          @keydown="handleKeyDown"></textarea>
                <div class="flex items-center gap-1.5 sm:gap-2 shrink-0">
                  <button type="button" :data-stream="streamOn ? 'on' : 'off'" @click="streamOn = !streamOn"
                    class="stream-toggle-btn px-3 py-2.5 rounded-xl text-xs sm:text-sm font-medium flex items-center gap-1.5 transition-all duration-200 cursor-pointer shrink-0"
                    :aria-pressed="streamOn"
                    :title="streamOn ? '流式响应已开启，点击关闭' : '流式响应已关闭，点击开启'">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    <span>流式</span>
                  </button>
                  <button @click="send()" :disabled="isStreaming"
                    class="px-4 sm:px-5 py-2.5 bg-brand-500 text-white rounded-xl hover:bg-brand-600 active:scale-95 transition-all duration-200 cursor-pointer shadow-md shadow-brand-500/25 hover:shadow-lg hover:shadow-brand-500/30 flex items-center gap-1.5 disabled:opacity-40 disabled:cursor-not-allowed"
                    aria-label="发送" touch-action="manipulation">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                    </svg>
                    <span class="hidden sm:inline text-sm font-medium">发送</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </section>

        <!-- ====== SIDEBAR ====== -->
        <aside id="stats-sidebar"
               :class="['w-full lg:w-80 shrink-0 flex flex-col gap-3 lg:gap-4', sidebarOpen ? 'open' : '']">
          <div class="rounded-2xl p-4 bg-white/40">
            <h3 class="text-sm font-semibold text-warm-800/80 mb-3 flex items-center gap-2">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5.636 18.364a9 9 0 010-12.728m12.728 0a9 9 0 010 12.728m-9.9-2.829a5 5 0 010-7.07m7.072 0a5 5 0 010 7.07M13 12a1 1 0 11-2 0 1 1 0 012 0z" />
              </svg> 系统状态
            </h3>
            <div class="space-y-2 text-sm">
              <div class="flex justify-between">
                <span class="text-warm-800/60">状态</span>
                <span class="font-medium" :class="dotTextColor">{{ statusLabel }}</span>
              </div>
              <div class="flex justify-between"><span class="text-warm-800/60">版本</span><span class="text-warm-800/80">1.0.0</span></div>
              <div class="flex justify-between"><span class="text-warm-800/60">总查询</span><span class="text-warm-800/80">{{ stats.data.totalQueries }}</span></div>
            </div>
          </div>

          <div class="rounded-2xl p-4 bg-white/40">
            <h3 class="text-sm font-semibold text-warm-800/80 mb-3 flex items-center gap-2">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7v10c0 2 1 3 3 3h10c2 0 3-1 3-3V7M4 7h16M4 7l2-3h12l2 3" />
              </svg> 知识库
            </h3>
            <div class="space-y-2 text-sm">
              <div class="flex justify-between"><span class="text-warm-800/60">菜谱</span><span class="text-warm-800/80">{{ stats.data.recipes }}</span></div>
              <div class="flex justify-between"><span class="text-warm-800/60">食材</span><span class="text-warm-800/80">{{ stats.data.ingredients }}</span></div>
              <div class="flex justify-between"><span class="text-warm-800/60">烹饪步骤</span><span class="text-warm-800/80">{{ stats.data.steps }}</span></div>
              <div class="flex justify-between"><span class="text-warm-800/60">文档块</span><span class="text-warm-800/80">{{ stats.data.chunks }}</span></div>
              <div class="flex justify-between"><span class="text-warm-800/60">向量</span><span class="text-warm-800/80">{{ stats.data.vectors }}</span></div>
            </div>
          </div>

          <!-- 用户记忆 -->
          <div class="rounded-2xl p-4 bg-white/40">
            <h3 class="text-sm font-semibold text-warm-800/80 mb-3 flex items-center justify-between gap-2">
              <span class="flex items-center gap-2">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg> 我的记忆
              </span>
              <button @click="profile.refresh()" title="刷新记忆"
                class="p-1 rounded hover:bg-warm-800/10 transition-colors cursor-pointer">
                <svg class="w-3.5 h-3.5 text-warm-800/40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
              </button>
            </h3>
            <ProfilePanel :state="profile.state" />
          </div>

          <!-- 上传 -->
          <div class="space-y-2">
            <div class="glass rounded-2xl p-4">
              <h3 class="text-sm font-semibold text-warm-800/80 mb-3 flex items-center gap-2">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg> 上传食谱
              </h3>
              <label for="file-upload"
                class="flex flex-col items-center gap-1.5 p-3 border-2 border-dashed border-warm-800/15 rounded-xl cursor-pointer hover:border-brand-400/50 hover:bg-brand-50/30 transition-colors duration-200 text-center">
                <svg class="w-5 h-5 text-warm-800/30" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                </svg>
                <span class="text-xs text-warm-800/50">选择 .md / .pdf / .txt</span>
                <input id="file-upload" type="file" accept=".md,.markdown,.pdf,.txt" class="hidden" @change="onFileSelect" />
              </label>
              <div v-if="uploadStatus" class="mt-2 text-xs text-center" :class="uploadStatusClass">{{ uploadStatus }}</div>
            </div>
            <div v-if="uploadedFiles.length" class="glass rounded-2xl p-3">
              <div class="space-y-1.5 max-h-32 overflow-y-auto text-xs">
                <div v-for="f in uploadedFiles" :key="f" class="flex items-center justify-between py-1 px-2 rounded-lg hover:bg-white/30">
                  <span class="text-warm-800/70 truncate flex-1">{{ f }}</span>
                  <button class="text-warm-800/30 hover:text-red-400 transition-colors cursor-pointer ml-1 shrink-0"
                          @click="removeFile(f)" aria-label="删除文件">
                    <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              </div>
            </div>
            <button @click="processUploadedFiles" :disabled="isProcessing || uploadedFiles.length === 0"
              class="w-full py-2 glass rounded-xl text-xs font-medium text-brand-500 hover:text-brand-600 hover:bg-white/60 transition-colors duration-200 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed">
              {{ isProcessing ? '处理中…' : `处理全部食谱 (${uploadedFiles.length})` }}
            </button>
            <div v-if="isProcessing" class="space-y-1">
              <div class="h-1.5 rounded-full bg-warm-800/10 overflow-hidden">
                <div class="h-full rounded-full bg-brand-400 transition-all duration-500" :style="{ width: progress + '%' }"></div>
              </div>
              <div class="text-xs text-warm-800/50 mt-1 text-center">{{ progressText }}</div>
            </div>
          </div>

          <button @click="onRebuild" :disabled="rebuildText === '重建中…'"
            class="w-full py-2.5 glass-strong rounded-xl text-sm font-medium text-warm-800/60 hover:text-brand-600 hover:bg-white/80 transition-colors duration-200 cursor-pointer shrink-0 disabled:opacity-50">
            {{ rebuildText }}
          </button>
        </aside>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, onBeforeUnmount, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { useStatsStore } from '@/stores/stats'
import { useProfileStore } from '@/stores/profile'
import { useDishIndexStore } from '@/stores/dishIndex'
import { toast } from '@/composables/useToast'
import { apiFetch, API_BASE, API_PREFIX } from '@/api/client'
import { now, escapeHtml } from '@/utils'
import MessageBubble from '@/components/MessageBubble.vue'
import ProfilePanel from '@/components/ProfilePanel.vue'

const router = useRouter()
const auth = useAuthStore()
const stats = useStatsStore()
const profile = useProfileStore()
const dishIndex = useDishIndexStore()

const messages = ref([]) // {id, role, content, time, intent?, stages?, error?}
const input = ref('')
const isStreaming = ref(false)
const streamOn = ref(true)
const sidebarOpen = ref(false)

const inputRef = ref(null)
const listRef = ref(null)

const presets = ['鱼香肉丝怎么做？', '家里有土豆和牛肉能做什么菜？', '推荐几道简单的川菜']

const uploadedFiles = ref([])
const isProcessing = ref(false)
const progress = ref(0)
const progressText = ref('')
const uploadStatus = ref('')
const uploadStatusClass = ref('')

const rebuildConfirm = ref(false)
const rebuildText = ref('重建知识库')

const STATUS_COLORS = {
  ok: { dot: 'bg-emerald-400', text: 'text-emerald-500', label: '运行中' },
  initializing: { dot: 'bg-yellow-400', text: 'text-yellow-500', label: '初始化中' },
  error: { dot: 'bg-red-400', text: 'text-red-500', label: '异常' },
  connecting: { dot: 'bg-yellow-400', text: 'text-yellow-500', label: '检测中' },
}
const dotColor = computed(() => STATUS_COLORS[stats.data.status]?.dot || 'bg-gray-400')
const dotTextColor = computed(() => STATUS_COLORS[stats.data.status]?.text || 'text-gray-500')
const statusLabel = computed(() => STATUS_COLORS[stats.data.status]?.label || '未知')

function genId() { return 'msg-' + Date.now() + '-' + Math.random().toString(36).slice(2, 6) }

function pushUserMessage(text) {
  const id = genId()
  messages.value.push({ id, role: 'user', content: text, time: now() })
  return id
}

function pushAssistantSkeleton() {
  const id = genId()
  messages.value.push({
    id, role: 'assistant', content: '',
    intent: null, statusPill: null, stages: null, error: null, time: now(),
  })
  return id
}

function updateMessage(id, patch) {
  const idx = messages.value.findIndex(m => m.id === id)
  if (idx >= 0) messages.value[idx] = { ...messages.value[idx], ...patch }
}

function scrollToBottom() {
  nextTick(() => {
    const el = listRef.value
    if (el) el.scrollTop = el.scrollHeight
  })
}

function autoResize(e) {
  const el = e.target
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 128) + 'px'
}

function handleKeyDown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    send()
  }
}

async function send(presetText) {
  if (isStreaming.value) return
  const q = (presetText ?? input.value).trim()
  if (!q) return
  if (!presetText) {
    input.value = ''
    if (inputRef.value) inputRef.value.style.height = 'auto'
  }
  pushUserMessage(q)
  isStreaming.value = true
  try {
    if (streamOn.value) {
      await sendStreaming(q)
    } else {
      await sendNonStreaming(q)
    }
  } finally {
    isStreaming.value = false
    stats.refresh()
    inputRef.value?.focus()
  }
}

const INTENT_LABELS = {
  general:    { label: '常规问答', color: 'bg-emerald-50 text-emerald-600' },
  detail:     { label: '细节查询', color: 'bg-sky-50 text-sky-600' },
  recommend:  { label: '菜品推荐', color: 'bg-amber-50 text-amber-600' },
  'multi-hop':{ label: '图RAG',    color: 'bg-purple-50 text-purple-600' },
  unknown:    { label: '未识别',   color: 'bg-warm-100 text-warm-600' },
}
const STRATEGY_LABELS = {
  hybrid_traditional: { label: '混合检索', color: 'bg-brand-50 text-brand-600' },
  graph_rag: { label: '图RAG', color: 'bg-purple-50 text-purple-600' },
  combined: { label: '组合策略', color: 'bg-teal-50 text-teal-600' },
}

function renderIntentLabel(label) {
  const info = INTENT_LABELS[label]
  if (info) return `<span class="text-xs px-2 py-0.5 rounded-full font-medium ${info.color}">${info.label}</span>`
  return ''
}

function renderStrategyLabel(strategy) {
  const s = STRATEGY_LABELS[strategy] || STRATEGY_LABELS.hybrid_traditional
  return `<span class="text-xs px-2 py-0.5 rounded-full font-medium ${s.color}">${s.label}</span>`
}

async function sendNonStreaming(question) {
  const msgId = pushAssistantSkeleton()
  scrollToBottom()
  try {
    const data = await apiFetch('/chat', {
      method: 'POST',
      body: JSON.stringify({ question, stream: false, session_id: sessionId.value, user_id: auth.userId }),
    })
    const strategy = data.analysis?.recommended_strategy || 'hybrid_traditional'
    updateMessage(msgId, {
      content: data.answer || '抱歉，没有找到与您问题相关的烹饪信息。',
      headerHtml: renderStrategyLabel(strategy),
    })
  } catch (err) {
    updateMessage(msgId, {
      content: err.message || '请求失败，请稍后重试',
      error: true,
    })
  }
  scrollToBottom()
}

async function sendStreaming(question) {
  const msgId = pushAssistantSkeleton()
  updateMessage(msgId, {
    intentHtml: '<span class="text-xs px-2 py-0.5 rounded-full font-medium bg-warm-100 text-warm-600">检索中…</span>',
    streaming: true,
  })
  scrollToBottom()

  const url = `${API_BASE}${API_PREFIX}/chat`
  let answerText = ''
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${auth.token}`,
      },
      body: JSON.stringify({ question, stream: true, session_id: sessionId.value, user_id: auth.userId }),
    })
    if (!res.ok) {
      const data = await res.json().catch(() => ({}))
      const msg = data?.detail?.message || '请求失败，请稍后重试'
      toast(msg, 'error')
      updateMessage(msgId, { content: msg, error: true, streaming: false })
      return
    }
    const reader = res.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop()
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        try {
          const event = JSON.parse(line.slice(6))
          if (event.type === 'intent') {
            const label = event.data?.label
            updateMessage(msgId, {
              intentHtml: renderIntentLabel(label),
              statusPill: '正在检索…',
            })
          } else if (event.type === 'analysis') {
            // 没拿到 intent 时兜底
            const cur = messages.value.find(m => m.id === msgId)
            if (cur && !cur.intentHtml) {
              const strategy = event.data.recommended_strategy
              updateMessage(msgId, {
                intentHtml: renderStrategyLabel(strategy),
                statusPill: '正在检索…',
              })
            }
          } else if (event.type === 'status') {
            const c = String(event.content || '')
            if (c === '__thinking__' || c.startsWith('__tool__:') || c.startsWith('__cache_hit__')) {
              updateMessage(msgId, { statusPill: '正在检索…' })
            }
          } else if (event.type === 'chunk') {
            const cur = messages.value.find(m => m.id === msgId)
            if (cur && cur.statusPill === '正在检索…') {
              updateMessage(msgId, { statusPill: '正在生成…' })
            }
            answerText += event.content
            updateMessage(msgId, { content: answerText })
            scrollToBottom()
          } else if (event.type === 'done') {
            updateMessage(msgId, { streaming: false, statusPill: null })
            if (event.stages && Object.keys(event.stages).length) {
              const text = Object.entries(event.stages).map(([k, v]) => `${k}=${(+v).toFixed(0)}ms`).join('  ')
              updateMessage(msgId, { stages: text })
            }
          } else if (event.type === 'error') {
            updateMessage(msgId, {
              content: (answerText || '') + `\n\n⚠ ${event.message}`,
              error: true,
              streaming: false,
            })
          }
        } catch (_) { /* skip */ }
      }
    }
  } catch (e) {
    updateMessage(msgId, {
      content: (answerText ? answerText + '\n\n[回答中断]' : '网络连接中断，请重试'),
      error: true,
      streaming: false,
    })
  }
  scrollToBottom()
  // 拉取偏好
  profile.refresh()
}

// session id
const sessionId = ref(sessionStorage.getItem('c9_session_id') || (() => {
  const id = 's_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8)
  sessionStorage.setItem('c9_session_id', id)
  return id
})())

function onLogout() {
  auth.logout()
  sessionStorage.removeItem('c9_session_id')
  router.replace('/auth')
}

async function onFileSelect(e) {
  const file = e.target.files?.[0]
  if (!file) return
  uploadStatus.value = '上传中…'
  uploadStatusClass.value = 'text-warm-800/60'
  try {
    const fd = new FormData()
    fd.append('file', file)
    const data = await apiFetch('/upload', { method: 'POST', body: fd })
    if (data.status === 'ok') {
      uploadStatus.value = `"${file.name}" 上传成功`
      uploadStatusClass.value = 'text-emerald-500 font-medium'
      if (!uploadedFiles.value.includes(file.name)) {
        uploadedFiles.value.push(file.name)
      }
      toast('食谱上传成功', 'success')
    } else {
      uploadStatus.value = data.message || '上传失败'
      uploadStatusClass.value = 'text-red-500'
      toast(uploadStatus.value, 'error')
    }
  } catch (err) {
    uploadStatus.value = '网络错误，上传失败'
    uploadStatusClass.value = 'text-red-500'
  }
  e.target.value = ''
}

function removeFile(name) {
  uploadedFiles.value = uploadedFiles.value.filter(f => f !== name)
}

let pollTimer = null
async function processUploadedFiles() {
  if (isProcessing.value || uploadedFiles.value.length === 0) return
  isProcessing.value = true
  progress.value = 0
  progressText.value = '准备中…'
  try {
    await apiFetch('/recipes/process', {
      method: 'POST',
      body: JSON.stringify({ filenames: uploadedFiles.value, skip_existing: false }),
    })
    progress.value = 10
    progressText.value = '解析菜谱中…'
    pollTimer = setInterval(async () => {
      try {
        const sd = await apiFetch('/recipes/process/status')
        progress.value = sd.progress_pct || 0
        if (sd.stage === 'processing') {
          progressText.value = sd.current_file ? `处理: ${sd.current_file}` : `处理中 ${sd.processed}/${sd.total_files}`
        } else if (sd.stage === 'indexing') {
          progressText.value = '更新 RAG 索引中…'
          progress.value = 95
        } else if (sd.stage === 'done') {
          progress.value = 100
          progressText.value = '处理完成'
          clearInterval(pollTimer)
          pollTimer = null
          uploadedFiles.value = []
          stats.refresh()
          toast('食谱处理完成', 'success')
          setTimeout(() => { progressText.value = '' }, 3000)
        } else if (sd.stage === 'error') {
          progressText.value = '处理出错'
          clearInterval(pollTimer)
          pollTimer = null
          toast('食谱处理出错', 'error')
        }
      } catch (_) {}
    }, 2000)
  } catch (err) {
    toast(err.message || '处理请求失败', 'error')
  } finally {
    isProcessing.value = false
  }
}

async function onRebuild() {
  if (rebuildText.value === '重建中…') return
  if (!rebuildConfirm.value) {
    rebuildConfirm.value = true
    rebuildText.value = '确认重建？再次点击执行'
    setTimeout(() => {
      if (rebuildConfirm.value) {
        rebuildConfirm.value = false
        rebuildText.value = '重建知识库'
      }
    }, 5000)
    return
  }
  rebuildConfirm.value = false
  rebuildText.value = '重建中…'
  try {
    const res = await apiFetch('/rebuild', { method: 'POST' })
    if (res.status === 'ok') toast('知识库重建成功', 'success')
    else toast(res.message || '重建失败', 'error')
  } catch (err) {
    toast(err.message || '网络错误', 'error')
  }
  rebuildText.value = '重建知识库'
  stats.refresh()
}

// 启动
let stats2Timer = null
let pagehideHandler = null
onMounted(async () => {
  stats.start()
  // 15s 单独刷新统计数据
  stats2Timer = setInterval(() => stats.refresh(), 15000)
  // 预加载菜品索引 & 偏好
  dishIndex.load()
  profile.refresh()
  // 关闭页面即销毁当前会话
  pagehideHandler = () => sessionStorage.removeItem('c9_session_id')
  window.addEventListener('pagehide', pagehideHandler)
})

onBeforeUnmount(() => {
  stats.stop()
  if (stats2Timer) clearInterval(stats2Timer)
  if (pollTimer) clearInterval(pollTimer)
  if (pagehideHandler) window.removeEventListener('pagehide', pagehideHandler)
})
</script>
