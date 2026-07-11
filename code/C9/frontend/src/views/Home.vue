<template>
  <div class="min-h-screen text-warm-900 bg-mesh relative">
    <!-- ====== NAVIGATION ====== -->
    <nav class="fixed top-4 left-4 right-4 z-50" aria-label="主导航">
      <div class="max-w-6xl mx-auto glass-strong rounded-2xl px-6 py-3.5 flex items-center justify-between">
        <router-link to="/" class="flex items-center gap-2.5 no-underline" aria-label="What2Eat 首页">
          <span class="text-2xl">🥢</span>
          <span class="font-semibold text-lg text-warm-900">
            What2Eat <span class="text-brand-500 font-light">尝尝咸淡</span>
          </span>
        </router-link>
        <div class="hidden md:flex items-center gap-8 text-sm font-medium text-warm-800">
          <a href="#features" class="hover:text-brand-500 transition-colors duration-200 cursor-pointer">功能特性</a>
          <a href="#how-it-works" class="hover:text-brand-500 transition-colors duration-200 cursor-pointer">工作原理</a>
          <router-link
            v-if="!auth.isLoggedIn"
            to="/auth"
            class="px-5 py-2.5 bg-brand-500 text-white rounded-xl hover:bg-brand-600 transition-colors duration-200 cursor-pointer shadow-md shadow-brand-500/25"
          >
            登录 / 注册
          </router-link>
          <router-link
            v-else
            to="/chat"
            class="px-5 py-2.5 bg-brand-500 text-white rounded-xl hover:bg-brand-600 transition-colors duration-200 cursor-pointer shadow-md shadow-brand-500/25"
          >
            进入聊天
          </router-link>
        </div>
        <button class="md:hidden p-2 rounded-lg hover:bg-white/50" aria-label="菜单" @click="mobileMenuOpen = !mobileMenuOpen">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
      </div>
      <div
        v-show="mobileMenuOpen"
        class="md:hidden mt-3 glass-strong rounded-2xl px-6 py-4 flex flex-col gap-3 text-sm font-medium text-warm-800"
      >
        <a href="#features" class="py-2 hover:text-brand-500" @click="mobileMenuOpen = false">功能特性</a>
        <a href="#how-it-works" class="py-2 hover:text-brand-500" @click="mobileMenuOpen = false">工作原理</a>
        <router-link
          v-if="!auth.isLoggedIn"
          to="/auth"
          class="py-2.5 px-5 bg-brand-500 text-white rounded-xl text-center hover:bg-brand-600"
        >
          登录 / 注册
        </router-link>
        <router-link v-else to="/chat" class="py-2.5 px-5 bg-brand-500 text-white rounded-xl text-center hover:bg-brand-600">
          进入聊天
        </router-link>
      </div>
    </nav>

    <!-- ====== HERO ====== -->
    <section class="relative pt-36 pb-20 md:pt-48 md:pb-32 overflow-hidden" aria-labelledby="hero-heading">
      <div class="orb orb-1"></div>
      <div class="orb orb-2"></div>
      <div class="relative z-10 max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="max-w-3xl mx-auto text-center">
          <div class="inline-flex items-center gap-2 glass rounded-full px-4 py-2 text-sm text-brand-600 font-medium mb-8">
            <span class="w-1.5 h-1.5 rounded-full bg-brand-500 animate-pulse"></span>
            基于知识图谱的智能烹饪助手
          </div>
          <h1 id="hero-heading" class="text-4xl sm:text-5xl md:text-6xl font-bold tracking-tight leading-tight mb-6">
            懂食材，更懂
            <span class="bg-gradient-to-r from-brand-500 via-brand-400 to-gold-500 bg-clip-text text-transparent">
              你的口味
            </span>
          </h1>
          <p class="text-lg md:text-xl text-warm-800/80 leading-relaxed mb-10 max-w-2xl mx-auto">
            What2Eat 尝咸淡是一款基于知识图谱和 AI 的智能食谱问答助手。<br class="hidden sm:block" />
            搭载 RAG 检索增强引擎，为你提供精准、专业的中餐烹饪指导。
          </p>
          <div class="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16">
            <router-link to="/auth"
              class="w-full sm:w-auto px-8 py-4 bg-brand-500 text-white font-semibold rounded-2xl hover:bg-brand-600 active:scale-[0.98] transition-all duration-200 cursor-pointer shadow-lg shadow-brand-500/30 text-lg">
              立即登录
            </router-link>
            <a href="#features"
              class="w-full sm:w-auto px-8 py-4 glass-strong text-warm-800 font-semibold rounded-2xl hover:bg-white/80 transition-all duration-200 cursor-pointer text-lg">
              了解更多
            </a>
          </div>
          <div class="glass rounded-2xl p-6 sm:p-8 grid grid-cols-3 gap-4 sm:gap-8">
            <div class="text-center">
              <div class="text-3xl sm:text-4xl font-bold text-brand-500 mb-1">10万+</div>
              <div class="text-sm text-warm-800/70 font-medium">菜谱数据</div>
            </div>
            <div class="text-center">
              <div class="text-3xl sm:text-4xl font-bold text-brand-500 mb-1">3 种</div>
              <div class="text-sm text-warm-800/70 font-medium">智能检索策略</div>
            </div>
            <div class="text-center">
              <div class="text-3xl sm:text-4xl font-bold text-brand-500 mb-1">毫秒级</div>
              <div class="text-sm text-warm-800/70 font-medium">响应速度</div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- ====== FEATURES ====== -->
    <section id="features" class="relative py-20 md:py-28" aria-labelledby="features-heading">
      <div class="orb orb-3"></div>
      <div class="relative z-10 max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="text-center mb-14">
          <h2 id="features-heading" class="text-3xl sm:text-4xl font-bold mb-4">智能烹饪，从「问」开始</h2>
          <div class="section-divider mb-6"></div>
          <p class="text-lg text-warm-800/75 max-w-2xl mx-auto">
            What2Eat 将 AI 大模型与知识图谱深度融合，让每道菜的疑问都能得到精准回答
          </p>
        </div>
        <div class="grid md:grid-cols-3 gap-6 lg:gap-8">
          <div v-for="f in features" :key="f.title"
               class="glass rounded-2xl p-6 sm:p-8 hover:bg-white/70 transition-all duration-300 cursor-pointer group">
            <div class="icon-wrap mb-5" v-html="f.icon"></div>
            <h3 class="text-xl font-semibold mb-3">{{ f.title }}</h3>
            <p class="text-warm-800/70 leading-relaxed">{{ f.desc }}</p>
          </div>
        </div>
      </div>
    </section>

    <!-- ====== HOW IT WORKS ====== -->
    <section id="how-it-works" class="relative py-20 md:py-28" aria-labelledby="how-heading">
      <div class="relative z-10 max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="text-center mb-14">
          <h2 id="how-heading" class="text-3xl sm:text-4xl font-bold mb-4">三步，轻松掌握烹饪之道</h2>
          <div class="section-divider mb-6"></div>
          <p class="text-lg text-warm-800/75 max-w-2xl mx-auto">
            从菜谱解析到智能问答，What2Eat 用 AI 让烹饪变得更简单
          </p>
        </div>
        <div class="grid md:grid-cols-3 gap-6 lg:gap-8">
          <div v-for="(s, i) in steps" :key="s.title"
               class="glass rounded-2xl p-6 sm:p-8 text-center hover:bg-white/70 transition-all duration-300 cursor-pointer">
            <div class="step-num mx-auto mb-5">{{ i + 1 }}</div>
            <h3 class="text-lg font-semibold mb-3">{{ s.title }}</h3>
            <p class="text-warm-800/70 leading-relaxed text-sm">{{ s.desc }}</p>
          </div>
        </div>
      </div>
    </section>

    <!-- ====== TECH STACK ====== -->
    <section class="relative py-20 md:py-28" aria-labelledby="tech-heading">
      <div class="relative z-10 max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="text-center mb-14">
          <h2 id="tech-heading" class="text-3xl sm:text-4xl font-bold mb-4">技术架构</h2>
          <div class="section-divider mb-6"></div>
          <p class="text-lg text-warm-800/75 max-w-2xl mx-auto">
            前沿技术栈，构建新一代智能烹饪知识系统
          </p>
        </div>
        <div class="glass-strong rounded-2xl p-6 sm:p-10 overflow-x-auto">
          <div class="flex flex-wrap items-center justify-center gap-4 md:gap-6 min-w-fit">
            <div v-for="t in techs" :key="t.label"
                 class="glass rounded-xl px-5 py-3 flex items-center gap-2.5 text-sm font-medium whitespace-nowrap">
              <span :style="{ color: t.color }" class="inline-flex" v-html="t.icon"></span>
              {{ t.label }}
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- ====== CTA ====== -->
    <section id="cta" class="relative py-20 md:py-28" aria-labelledby="cta-heading">
      <div class="relative z-10 max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <div class="glass-strong rounded-3xl p-8 sm:p-12 md:p-16">
          <h2 id="cta-heading" class="text-3xl sm:text-4xl font-bold mb-4">开始你的智能烹饪之旅</h2>
          <div class="section-divider mb-6"></div>
          <p class="text-lg text-warm-800/75 mb-10 max-w-xl mx-auto leading-relaxed">
            只需一行命令，即刻启动 AI 烹饪助手。让每一道菜，都有 AI 为你保驾护航。
          </p>
          <div class="glass rounded-xl p-4 sm:p-5 mb-8 text-left overflow-x-auto">
            <code class="text-sm sm:text-base text-warm-800 font-mono whitespace-nowrap">
              <span class="text-brand-500">cd</span> code/C9
              <span class="text-warm-800/30">&amp;&amp;</span>
              <span class="text-brand-500">python</span> main.py
            </code>
          </div>
          <div class="flex flex-col sm:flex-row items-center justify-center gap-3">
            <a href="https://github.com" target="_blank" rel="noopener noreferrer"
               class="w-full sm:w-auto px-8 py-4 bg-brand-500 text-white font-semibold rounded-2xl hover:bg-brand-600 active:scale-[0.98] transition-all duration-200 cursor-pointer shadow-lg shadow-brand-500/30">
              GitHub 获取源码
            </a>
            <a href="#features"
               class="w-full sm:w-auto px-8 py-4 glass text-warm-800 font-semibold rounded-2xl hover:bg-white/60 transition-all duration-200 cursor-pointer">
              查看文档
            </a>
          </div>
        </div>
      </div>
    </section>

    <!-- ====== FOOTER ====== -->
    <footer class="relative pb-10" role="contentinfo">
      <div class="relative z-10 max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="glass rounded-2xl px-6 py-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-warm-800/60">
          <div class="flex items-center gap-2">
            <span class="text-lg">🥢</span>
            <span>What2Eat 尝尝咸淡 — AI 驱动的中餐食谱知识图谱问答系统</span>
          </div>
          <div class="flex items-center gap-6">
            <span>Python 3.12+</span>
            <span class="hidden sm:inline">·</span>
            <span>Neo4j 5.18</span>
            <span class="hidden sm:inline">·</span>
            <span>Milvus 2.5</span>
          </div>
        </div>
      </div>
    </footer>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useAuthStore } from '@/stores/auth'

const auth = useAuthStore()
const mobileMenuOpen = ref(false)

const features = [
  {
    title: '知识图谱引擎',
    desc: '基于 Neo4j 构建菜谱知识图谱，深度关联食材、步骤、口味，让 AI 真正理解烹饪逻辑而非简单匹配关键词。',
    icon: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>`,
  },
  {
    title: '双路混合检索',
    desc: 'Milvus 向量搜索 + BM25 关键词检索双引擎融合，毫秒级召回最相关的菜谱知识，回答既精准又全面。',
    icon: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/></svg>`,
  },
  {
    title: '智能问答对话',
    desc: '基于 Moonshot/Kimi 大模型，结合检索增强生成技术，支持多轮对话，像大厨一样为你答疑解惑。',
    icon: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"/></svg>`,
  },
]

const steps = [
  { title: '导入菜谱数据', desc: 'AI 自动解析 Markdown 菜谱，提取食材、步骤、口味等关键信息，构建结构化知识网络。' },
  { title: '输入你的问题',  desc: '「鱼香肉丝怎么做？」「家里有土豆和牛肉能做什么菜？」—— 用自然语言直接提问。' },
  { title: '获得精准答案',  desc: '智能路由选择最优检索策略，结合大模型为你生成步骤详尽、调味精准的烹饪指导。' },
]

const techs = [
  { label: 'Neo4j 图数据库', color: '#4581C3', icon: `<svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M9.613.183c-2.353.674-4.42 2.094-5.954 4.095L1.646 2.22l-.107.152c1.64 2.358 3.904 4.198 6.526 5.25l.153.048.15-.06c2.425-.922 4.566-2.55 6.157-4.665l.038-.067-.05-.08c-1.183-1.874-2.974-3.22-4.9-3.75v-.08h.04c.66.22 1.28.53 1.85.93l.09.07.06-.1c-.65-1.08-1.66-2.01-2.95-2.31l-.09-.02V.18l-.05.003z"/></svg>` },
  { label: 'Milvus 向量数据库', color: '#00A4D3', icon: `<svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>` },
  { label: 'Moonshot/Kimi 大模型', color: '#6366F1', icon: `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z"/></svg>` },
  { label: 'BGE 中文嵌入模型', color: '#059669', icon: `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4.26 10.147a60.436 60.436 0 00-.491 6.347A48.627 48.627 0 0112 20.904a48.627 48.627 0 018.232-4.41 60.46 60.46 0 00-.491-6.347m-15.482 0a50.57 50.57 0 00-2.658-.813A59.905 59.905 0 0112 3.493a59.902 59.902 0 0110.399 5.84c-.896.248-1.783.52-2.658.814m-15.482 0A50.697 50.697 0 0112 13.489a50.7 50.7 0 017.74-3.342"/></svg>` },
]
</script>
