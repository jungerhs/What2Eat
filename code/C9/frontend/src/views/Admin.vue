<template>
  <div class="bg-admin bg-grid text-warm-900 relative min-h-screen">
    <div class="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 py-6">
      <!-- NAV -->
      <nav class="glass-strong rounded-2xl px-5 py-3 flex items-center justify-between mb-6">
        <div class="flex items-center gap-3">
          <router-link to="/" class="flex items-center gap-2.5 no-underline shrink-0">
            <span class="text-xl">🥢</span>
            <div class="leading-tight">
              <div class="font-semibold text-base text-warm-900">What2Eat <span class="text-brand-500 font-light">管理后台</span></div>
              <div class="text-[0.68rem] text-warm-800/45 font-mono tracking-wider uppercase">User Administration</div>
            </div>
          </router-link>
        </div>
        <div class="flex items-center gap-3">
          <router-link to="/chat"
            class="hidden sm:flex items-center gap-1.5 text-sm text-warm-800/65 hover:text-brand-500 transition-colors px-3 py-1.5 rounded-lg hover:bg-white/50">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 3v-3z" />
            </svg> 返回聊天
          </router-link>
          <span class="hidden sm:inline text-sm text-warm-800/70 font-medium">
            👤 {{ auth.username }}
            <span class="chip chip-admin ml-1" style="font-size:0.65rem; padding:1px 7px;">{{ auth.role }}</span>
          </span>
          <button @click="onLogout"
            class="text-xs px-3 py-1.5 rounded-lg bg-white/50 hover:bg-white/80 text-warm-800/70 hover:text-brand-600 transition-colors border border-warm-800/10">
            登出
          </button>
        </div>
      </nav>

      <header class="mb-6">
        <div class="flex items-end justify-between flex-wrap gap-3">
          <div>
            <h1 class="text-2xl sm:text-3xl font-bold text-warm-900 tracking-tight">
              用户管理
              <span class="text-brand-500 font-light text-base sm:text-lg ml-2 font-mono">/users</span>
            </h1>
            <p class="text-sm text-warm-800/55 mt-1">管理 What2Eat 平台所有用户账号、角色与登录状态。</p>
          </div>
          <div class="text-xs text-warm-800/45 font-mono">
            <span>{{ lastSyncTime }}</span> · <span>{{ syncStatus }}</span>
          </div>
        </div>
      </header>

      <!-- STATS -->
      <section class="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 mb-6">
        <div class="glass rounded-2xl p-4 sm:p-5 relative overflow-hidden">
          <div class="absolute -top-2 -right-2 w-16 h-16 bg-brand-500/8 rounded-full blur-xl"></div>
          <div class="relative">
            <div class="text-[0.7rem] font-semibold tracking-wider uppercase text-warm-800/45">总用户数</div>
            <div class="mt-1 flex items-baseline gap-1.5">
              <span ref="statTotal" class="text-3xl sm:text-4xl font-bold text-warm-900 font-mono num-anim">{{ stats.total }}</span>
              <span class="text-xs text-warm-800/40">人</span>
            </div>
          </div>
        </div>
        <div class="glass rounded-2xl p-4 sm:p-5 relative overflow-hidden">
          <div class="absolute -top-2 -right-2 w-16 h-16 bg-gold-400/15 rounded-full blur-xl"></div>
          <div class="relative">
            <div class="text-[0.7rem] font-semibold tracking-wider uppercase text-warm-800/45">管理员</div>
            <div class="mt-1 flex items-baseline gap-1.5">
              <span ref="statAdmins" class="text-3xl sm:text-4xl font-bold font-mono num-anim" style="color:#B8741E;">{{ stats.admins }}</span>
              <span class="text-xs text-warm-800/40">人</span>
            </div>
          </div>
        </div>
        <div class="glass rounded-2xl p-4 sm:p-5 relative overflow-hidden">
          <div class="absolute -top-2 -right-2 w-16 h-16 bg-emerald-400/12 rounded-full blur-xl"></div>
          <div class="relative">
            <div class="text-[0.7rem] font-semibold tracking-wider uppercase text-warm-800/45">活跃账号</div>
            <div class="mt-1 flex items-baseline gap-1.5">
              <span ref="statActive" class="text-3xl sm:text-4xl font-bold font-mono num-anim" style="color:#047857;">{{ stats.active }}</span>
              <span class="text-xs text-warm-800/40">人</span>
            </div>
          </div>
        </div>
        <div class="glass rounded-2xl p-4 sm:p-5 relative overflow-hidden">
          <div class="absolute -top-2 -right-2 w-16 h-16 bg-red-400/10 rounded-full blur-xl"></div>
          <div class="relative">
            <div class="text-[0.7rem] font-semibold tracking-wider uppercase text-warm-800/45">已禁用</div>
            <div class="mt-1 flex items-baseline gap-1.5">
              <span ref="statDisabled" class="text-3xl sm:text-4xl font-bold font-mono num-anim" style="color:#B91C1C;">{{ stats.disabled }}</span>
              <span class="text-xs text-warm-800/40">人</span>
            </div>
          </div>
        </div>
      </section>

      <!-- TOOLBAR -->
      <section class="glass rounded-2xl p-3 mb-4 flex flex-wrap items-center gap-3">
        <div class="relative flex-1 min-w-[200px]">
          <svg class="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-warm-800/40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-4.35-4.35M11 18a7 7 0 100-14 7 7 0 000 14z" />
          </svg>
          <input v-model="keyword" type="text" placeholder="搜索用户名…"
                 class="input pl-9" @input="renderTable" />
        </div>
        <div class="flex items-center gap-1 p-1 rounded-xl bg-warm-100/50 border border-warm-800/8">
          <button v-for="f in filters" :key="f.key" :data-filter="f.key"
            @click="setFilter(f.key)"
            :class="['px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer',
                     currentFilter === f.key ? 'bg-white text-warm-900 shadow-sm' : 'text-warm-800/55 hover:bg-white/50']">
            {{ f.label }}
          </button>
        </div>
        <button @click="openCreateModal"
          class="btn btn-primary shrink-0">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M12 4v16m8-8H4" />
          </svg> 新增用户
        </button>
      </section>

      <!-- TABLE -->
      <section class="glass-strong rounded-2xl overflow-hidden">
        <div class="overflow-x-auto">
          <table class="user-table w-full">
            <thead>
              <tr>
                <th style="width: 26%;">用户名</th>
                <th style="width: 14%;">角色</th>
                <th style="width: 14%;">状态</th>
                <th style="width: 18%;">创建时间</th>
                <th style="width: 18%;">最近登录</th>
                <th style="width: 10%; text-align: right;">操作</th>
              </tr>
            </thead>
            <tbody>
              <tr v-if="initialLoading" v-for="i in 3" :key="`s${i}`"><td colspan="6"><div class="skeleton h-8 w-full"></div></td></tr>
              <tr v-for="(u, i) in filtered" :key="u.id" class="row-anim" :style="{ animationDelay: `${i * 0.035}s` }">
                <td>
                  <div class="flex items-center gap-2.5">
                    <div :class="['w-8 h-8 rounded-lg flex items-center justify-center text-sm font-semibold shrink-0', avatarClass(u)]">{{ avatarChar(u.username) }}</div>
                    <div>
                      <div class="font-semibold text-warm-900">
                        {{ u.username }}<span v-if="u.id === auth.userId" class="text-[0.62rem] text-brand-500 font-semibold ml-1.5 align-middle">YOU</span>
                      </div>
                      <div class="text-[0.68rem] text-warm-800/40 font-mono">{{ u.id.slice(0, 8) }}</div>
                    </div>
                  </div>
                </td>
                <td>
                  <span v-if="u.role === 'admin'" class="chip chip-admin"><span class="chip-dot"></span>管理员</span>
                  <span v-else class="chip chip-user"><span class="chip-dot"></span>普通用户</span>
                </td>
                <td>
                  <span v-if="u.is_active" class="chip chip-active"><span class="chip-dot"></span>启用</span>
                  <span v-else class="chip chip-disabled"><span class="chip-dot"></span>禁用</span>
                </td>
                <td><span class="font-mono text-xs text-warm-800/70">{{ formatTime(u.created_at) }}</span></td>
                <td>
                  <span v-if="u.last_login_at" class="font-mono text-xs text-warm-800/70">{{ formatTime(u.last_login_at) }}</span>
                  <span v-else class="text-warm-800/35">从未登录</span>
                </td>
                <td>
                  <div class="flex items-center justify-end gap-1">
                    <button @click="openEditModal(u.id)" class="icon-btn icon-btn-edit" title="编辑">
                      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                      </svg>
                    </button>
                    <button @click="toggleRole(u.id)" :disabled="u.id === auth.userId" class="icon-btn icon-btn-role" title="切换角色">
                      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                      </svg>
                    </button>
                    <button @click="toggleStatus(u.id)" :disabled="u.id === auth.userId" class="icon-btn icon-btn-toggle" :title="u.is_active ? '禁用' : '启用'">
                      <svg v-if="u.is_active" class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728L5.636 5.636m12.728 12.728L18.364 5.636M5.636 18.364l12.728-12.728" />
                      </svg>
                      <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </button>
                    <button @click="openDeleteModal(u.id)" :disabled="u.id === auth.userId" class="icon-btn icon-btn-del" title="删除">
                      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <div v-if="!initialLoading && filtered.length === 0" class="empty-state">
          <div class="text-4xl mb-2 opacity-40">📭</div>
          <div class="text-sm">没有匹配的用户</div>
        </div>
      </section>

      <footer class="text-center text-xs text-warm-800/35 mt-6 font-mono">
        What2Eat Admin · v1.0 · 默认管理员 <span class="text-brand-500">admin / admin123</span>（首次启动后请立即修改密码）
      </footer>
    </div>

    <!-- CREATE / EDIT MODAL -->
    <div v-if="modalOpen"
         class="fixed inset-0 z-50 flex items-center justify-center bg-warm-900/40 backdrop-blur-sm overlay-anim"
         @click.self="closeModal">
      <div class="modal-anim glass-strong rounded-2xl w-full max-w-md mx-4 overflow-hidden" @click.stop>
        <div class="px-6 py-4 border-b border-warm-800/8 flex items-center justify-between">
          <div>
            <h2 class="text-lg font-bold text-warm-900">{{ isEdit ? '编辑用户' : '新增用户' }}</h2>
            <p class="text-xs text-warm-800/50 mt-0.5 font-mono">{{ isEdit ? `edit · ${form.id?.slice(0, 8)}` : 'create a new account' }}</p>
          </div>
          <button @click="closeModal" class="icon-btn" aria-label="关闭">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <form @submit.prevent="onSubmitForm" class="px-6 py-5 space-y-4">
          <div>
            <label class="block text-xs font-semibold text-warm-800/70 mb-1.5 tracking-wide uppercase">用户名</label>
            <input v-model="form.username" type="text" required minlength="3" maxlength="32"
                   placeholder="3-32 字符" class="input" autocomplete="off" />
          </div>
          <div>
            <label class="block text-xs font-semibold text-warm-800/70 mb-1.5 tracking-wide uppercase">
              密码 <span class="text-warm-800/40 normal-case font-normal ml-1">（{{ isEdit ? '留空则不修改' : '6-64 字符' }}）</span>
            </label>
            <div class="relative">
              <input v-model="form.password" :type="showPassword ? 'text' : 'password'"
                     :required="!isEdit" minlength="6" maxlength="64"
                     placeholder="••••••" class="input pr-10" :autocomplete="isEdit ? 'new-password' : 'new-password'" />
              <button type="button" @click="showPassword = !showPassword"
                class="absolute right-2 top-1/2 -translate-y-1/2 icon-btn" aria-label="切换密码可见性">
                <svg v-if="!showPassword" class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
                <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                </svg>
              </button>
            </div>
          </div>
          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="block text-xs font-semibold text-warm-800/70 mb-1.5 tracking-wide uppercase">角色</label>
              <select v-model="form.role" class="input cursor-pointer">
                <option value="user">普通用户</option>
                <option value="admin">管理员</option>
              </select>
            </div>
            <div>
              <label class="block text-xs font-semibold text-warm-800/70 mb-1.5 tracking-wide uppercase">状态</label>
              <select v-model="form.is_active" class="input cursor-pointer">
                <option :value="true">启用</option>
                <option :value="false">禁用</option>
              </select>
            </div>
          </div>
          <div class="flex items-center justify-end gap-2 pt-3 border-t border-warm-800/8">
            <button type="button" @click="closeModal" class="btn btn-ghost">取消</button>
            <button type="submit" :disabled="formLoading" class="btn btn-primary">
              <svg v-if="formLoading" class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
              </svg>
              <span>{{ isEdit ? '保存' : '创建' }}</span>
            </button>
          </div>
        </form>
      </div>
    </div>

    <!-- DELETE MODAL -->
    <div v-if="deleteModalOpen"
         class="fixed inset-0 z-50 flex items-center justify-center bg-warm-900/40 backdrop-blur-sm overlay-anim"
         @click.self="closeDeleteModal">
      <div class="modal-anim glass-strong rounded-2xl w-full max-w-sm mx-4 overflow-hidden" @click.stop>
        <div class="px-6 py-5">
          <div class="flex items-start gap-3">
            <div class="shrink-0 w-10 h-10 rounded-full bg-red-100 flex items-center justify-center">
              <svg class="w-5 h-5 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
            <div>
              <h3 class="text-base font-bold text-warm-900">删除用户</h3>
              <p class="text-sm text-warm-800/65 mt-1">即将永久删除用户 <span class="font-semibold text-warm-900">{{ pendingDeleteUsername }}</span>，此操作不可撤销。</p>
            </div>
          </div>
          <div class="flex items-center justify-end gap-2 mt-5">
            <button @click="closeDeleteModal" class="btn btn-ghost">取消</button>
            <button @click="confirmDelete" :disabled="deleteLoading" class="btn btn-danger">
              <svg v-if="deleteLoading" class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
              </svg>
              <span>确认删除</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, onBeforeUnmount, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { apiFetch } from '@/api/client'
import { toast } from '@/composables/useToast'
import { formatTime, escapeHtml } from '@/utils'

const router = useRouter()
const auth = useAuthStore()

const users = ref([])
const initialLoading = ref(true)
const keyword = ref('')
const currentFilter = ref('all')
const lastSyncTime = ref('--')
const syncStatus = ref('已同步')

const stats = reactive({ total: 0, admins: 0, active: 0, disabled: 0 })
const filters = [
  { key: 'all',       label: '全部' },
  { key: 'admin',     label: '管理员' },
  { key: 'user',      label: '普通用户' },
  { key: 'disabled',  label: '已禁用' },
]

const filtered = computed(() => {
  let list = users.value
  if (currentFilter.value === 'admin') list = list.filter(u => u.role === 'admin')
  else if (currentFilter.value === 'user') list = list.filter(u => u.role === 'user')
  else if (currentFilter.value === 'disabled') list = list.filter(u => !u.is_active)
  const kw = keyword.value.trim().toLowerCase()
  if (kw) list = list.filter(u => u.username.toLowerCase().includes(kw))
  return list
})

const modalOpen = ref(false)
const isEdit = ref(false)
const formLoading = ref(false)
const showPassword = ref(false)
const form = reactive({ id: '', username: '', password: '', role: 'user', is_active: true })

const deleteModalOpen = ref(false)
const deleteLoading = ref(false)
const pendingDeleteId = ref(null)
const pendingDeleteUsername = ref('')

function onLogout() {
  auth.logout()
  router.replace('/auth')
}

async function loadUsers() {
  syncStatus.value = '加载中…'
  try {
    const data = await apiFetch('/admin/users')
    users.value = data.users || []
    if (data.stats) Object.assign(stats, data.stats)
    syncStatus.value = '已同步'
    lastSyncTime.value = new Date().toLocaleTimeString('zh-CN', { hour12: false })
  } catch (err) {
    syncStatus.value = '同步失败'
    toast(`加载失败: ${err.message}`, 'error')
  } finally {
    initialLoading.value = false
  }
}

function setFilter(key) { currentFilter.value = key }
function avatarClass(u) {
  if (u.role === 'admin') return 'bg-gold-400/20 text-gold-600'
  if (!u.is_active) return 'bg-red-100 text-red-600'
  return 'bg-brand-100 text-brand-600'
}
function avatarChar(name) { return (name || '?').charAt(0).toUpperCase() }

function openCreateModal() {
  isEdit.value = false
  Object.assign(form, { id: '', username: '', password: '', role: 'user', is_active: true })
  modalOpen.value = true
  nextTick(() => document.querySelector('input.input')?.focus())
}
function openEditModal(id) {
  const u = users.value.find(x => x.id === id)
  if (!u) return
  isEdit.value = true
  Object.assign(form, { id: u.id, username: u.username, password: '', role: u.role, is_active: !!u.is_active })
  modalOpen.value = true
}
function closeModal() { modalOpen.value = false }

async function onSubmitForm() {
  formLoading.value = true
  try {
    if (isEdit.value) {
      const body = { username: form.username, role: form.role, is_active: form.is_active }
      if (form.password) body.password = form.password
      await apiFetch(`/admin/users/${form.id}`, { method: 'PUT', body: JSON.stringify(body) })
      toast('用户已更新', 'success')
    } else {
      await apiFetch('/admin/users', { method: 'POST', body: JSON.stringify(form) })
      toast('用户已创建', 'success')
    }
    closeModal()
    await loadUsers()
  } catch (err) {
    toast(err.message || '保存失败', 'error')
  } finally {
    formLoading.value = false
  }
}

async function toggleRole(id) {
  const u = users.value.find(x => x.id === id)
  if (!u) return
  const next = u.role === 'admin' ? 'user' : 'admin'
  try {
    await apiFetch(`/admin/users/${id}/role`, { method: 'PATCH', body: JSON.stringify({ role: next }) })
    toast(`已切换为${next === 'admin' ? '管理员' : '普通用户'}`, 'success')
    await loadUsers()
  } catch (err) {
    toast(err.message, 'error')
  }
}
async function toggleStatus(id) {
  const u = users.value.find(x => x.id === id)
  if (!u) return
  const next = !u.is_active
  try {
    await apiFetch(`/admin/users/${id}/status`, { method: 'PATCH', body: JSON.stringify({ is_active: next }) })
    toast(next ? '已启用' : '已禁用', 'success')
    await loadUsers()
  } catch (err) {
    toast(err.message, 'error')
  }
}

function openDeleteModal(id) {
  const u = users.value.find(x => x.id === id)
  if (!u) return
  pendingDeleteId.value = id
  pendingDeleteUsername.value = u.username
  deleteModalOpen.value = true
}
function closeDeleteModal() { deleteModalOpen.value = false }

async function confirmDelete() {
  if (!pendingDeleteId.value) return
  deleteLoading.value = true
  try {
    await apiFetch(`/admin/users/${pendingDeleteId.value}`, { method: 'DELETE' })
    toast('用户已删除', 'success')
    closeDeleteModal()
    await loadUsers()
  } catch (err) {
    toast(err.message, 'error')
  } finally {
    deleteLoading.value = false
  }
}

function onEsc(e) {
  if (e.key === 'Escape') {
    if (modalOpen.value) closeModal()
    if (deleteModalOpen.value) closeDeleteModal()
  }
}

onMounted(() => {
  loadUsers()
  document.addEventListener('keydown', onEsc)
})
onBeforeUnmount(() => {
  document.removeEventListener('keydown', onEsc)
})
</script>
