<template>
  <div class="min-h-screen text-warm-900 bg-mesh relative flex items-center justify-center px-4">
    <div class="w-full max-w-md">
      <div class="flex flex-col items-center mb-6">
        <div class="text-5xl mb-2">🥢</div>
        <h1 class="text-2xl font-semibold text-warm-900">
          What2Eat <span class="text-brand-500 font-light">尝咸淡</span>
        </h1>
        <p class="text-sm text-warm-800/60 mt-1">基于知识图谱的智能烹饪助手</p>
      </div>

      <div class="glass-strong rounded-2xl p-6 sm:p-8">
        <div class="flex gap-2 p-1 rounded-xl bg-warm-100/60 mb-6">
          <button id="tab-login" @click="mode = 'login'"
            :class="['flex-1 py-2 rounded-lg text-sm font-medium transition-all duration-200 cursor-pointer',
                     mode === 'login' ? 'tab-active' : 'tab-inactive']">
            登录
          </button>
          <button id="tab-register" @click="mode = 'register'"
            :class="['flex-1 py-2 rounded-lg text-sm font-medium transition-all duration-200 cursor-pointer',
                     mode === 'register' ? 'tab-active' : 'tab-inactive']">
            注册
          </button>
        </div>

        <form @submit.prevent="onSubmit" class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-warm-800/80 mb-1.5">
              用户名
              <span class="text-xs text-warm-800/40 font-normal ml-1">(3-32 字符)</span>
            </label>
            <input v-model="username" type="text" autocomplete="username" required minlength="3" maxlength="32"
              placeholder="请输入用户名"
              class="w-full bg-white/70 rounded-xl border border-warm-800/10 outline-none text-sm sm:text-base text-warm-900 placeholder-warm-800/40 py-2.5 px-3 focus:border-brand-400/50 focus:bg-white/90 transition-colors" />
          </div>
          <div>
            <label class="block text-sm font-medium text-warm-800/80 mb-1.5">
              密码
              <span class="text-xs text-warm-800/40 font-normal ml-1">(6-64 字符)</span>
            </label>
            <input v-model="password" type="password"
              :autocomplete="mode === 'login' ? 'current-password' : 'new-password'"
              required minlength="6" maxlength="64"
              placeholder="请输入密码"
              class="w-full bg-white/70 rounded-xl border border-warm-800/10 outline-none text-sm sm:text-base text-warm-900 placeholder-warm-800/40 py-2.5 px-3 focus:border-brand-400/50 focus:bg-white/90 transition-colors" />
          </div>

          <button type="submit" :disabled="loading"
            class="w-full py-2.5 bg-brand-500 text-white rounded-xl hover:bg-brand-600 active:scale-[0.98] transition-all duration-200 cursor-pointer shadow-md shadow-brand-500/25 hover:shadow-lg hover:shadow-brand-500/30 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed">
            <span>{{ mode === 'login' ? '登录' : '注册' }}</span>
            <svg v-if="loading" class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"></path>
            </svg>
          </button>
        </form>

        <p class="text-xs text-warm-800/40 text-center mt-4">
          登录后会保留用户偏好与历史对话；关闭页面后本次会话自动结束。
        </p>
      </div>

      <p class="text-xs text-warm-800/40 text-center mt-4">
        <router-link to="/" class="hover:text-brand-500 transition-colors">← 返回首页</router-link>
      </p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { toast } from '@/composables/useToast'
import { apiFetch } from '@/api/client'

const mode = ref('login') // 'login' | 'register'
const username = ref('')
const password = ref('')
const loading = ref(false)

const router = useRouter()
const route = useRoute()
const auth = useAuthStore()

// 已登录则直接跳转
onMounted(async () => {
  if (!auth.isLoggedIn) return
  const ok = await auth.fetchMe()
  if (ok) router.replace((route.query.redirect || '/chat').toString())
})

async function onSubmit() {
  if (!username.value.trim() || !password.value) return
  loading.value = true
  try {
    if (mode.value === 'login') {
      await auth.login(username.value.trim(), password.value)
    } else {
      await auth.register(username.value.trim(), password.value)
    }
    toast((mode.value === 'login' ? '登录成功' : '注册成功') + '，正在跳转...', 'success')
    setTimeout(() => {
      router.replace((route.query.redirect || '/chat').toString())
    }, 600)
  } catch (err) {
    if (err.status === 401) {
      toast('登录已过期或账号不可用', 'error')
    } else {
      toast(err.message || (mode.value === 'login' ? '登录失败' : '注册失败'), 'error')
    }
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.tab-active {
  background: linear-gradient(135deg, #E85D3A 0%, #D14525 100%);
  color: white;
  box-shadow: 0 4px 12px rgba(232,93,58,0.25);
}
.tab-inactive {
  background: rgba(255,255,255,0.4);
  color: rgba(61,46,26,0.6);
}
.tab-inactive:hover { background: rgba(255,255,255,0.65); }
</style>
