<template>
  <div v-if="state.loading">
    <div class="text-xs text-warm-800/40 text-center py-2">加载中…</div>
  </div>
  <div v-else-if="!state.prefs">
    <div class="text-xs text-warm-800/40 text-center py-2">
      {{ state.message || '暂无记忆' }}
    </div>
  </div>
  <div v-else>
    <div v-for="row in rows" :key="row.label" class="mb-2">
      <span class="text-xs text-warm-800/50 font-medium">{{ row.label }}</span>
      <div class="flex flex-wrap gap-1 mt-0.5">
        <span v-for="v in row.values" :key="v"
              class="inline-block px-1.5 py-0.5 rounded-full text-xs bg-brand-50 text-brand-600 border border-brand-100">
          {{ v }}
        </span>
      </div>
    </div>
    <div v-if="!rows.length" class="text-xs text-warm-800/40 text-center py-2">记忆正在形成中…</div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { escapeHtml } from '@/utils'

const props = defineProps({ state: { type: Object, required: true } })

const GROUPS = [
  { key: 'cuisine_preferences',    label: '🍽️ 偏好菜系' },
  { key: 'taste_preferences',      label: '🌶️ 口味偏好' },
  { key: 'avoid',                  label: '🚫 忌口' },
  { key: 'skill_level',            label: '🍳 烹饪水平', single: true },
  { key: 'dietary_restrictions',   label: '📋 饮食限制' },
  { key: 'kitchen_equipment',      label: '🔧 厨房设备' },
  { key: 'favorite_dishes',        label: '❤️ 喜欢的菜' },
]

const rows = computed(() => {
  if (!props.state.prefs) return []
  const prefs = props.state.prefs.preferences || props.state.prefs || {}
  const out = []
  for (const g of GROUPS) {
    const raw = prefs[g.key]
    let values = []
    if (g.single) {
      if (raw) values = [raw]
    } else if (Array.isArray(raw)) {
      values = raw
    } else if (raw) {
      values = [raw]
    }
    if (values.length) {
      out.push({ label: g.label, values: values.map(v => escapeHtml(String(v))) })
    }
  }
  // 最近问过
  const recent = prefs.recently_asked
  if (Array.isArray(recent) && recent.length) {
    out.push({ label: '🔎 最近问过', values: recent.slice(-5).map(v => escapeHtml(String(v))) })
  }
  return out
})
</script>
