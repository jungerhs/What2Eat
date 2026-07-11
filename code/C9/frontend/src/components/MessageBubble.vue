<template>
  <!-- user message -->
  <div v-if="message.role === 'user'" class="flex justify-end">
    <div>
      <div class="user-message">{{ message.content }}</div>
      <div class="text-xs text-warm-800/40 mt-1 text-right">{{ message.time }}</div>
    </div>
  </div>

  <!-- assistant message -->
  <div v-else class="flex justify-start">
    <div>
      <div class="ai-message">
        <!-- header: intent/status/strategy -->
        <div v-if="message.intentHtml || message.statusPill" class="flex items-center gap-2 mb-2">
          <span v-if="message.intentHtml" v-html="message.intentHtml"></span>
          <span v-if="message.statusPill"
                class="text-xs px-2 py-0.5 rounded-full font-medium bg-warm-100 text-warm-600 ml-1">
            {{ message.statusPill }}
          </span>
        </div>

        <!-- body -->
        <div v-if="message.content || message.streaming"
             class="msg-content text-sm sm:text-base text-warm-800 leading-relaxed"
             :class="{ 'streaming-cursor': message.streaming }"
             v-html="renderedContent"></div>

        <!-- error banner -->
        <div v-if="message.error" class="mt-2 text-red-500 text-sm">⚠ {{ message.error }}</div>

        <!-- stages -->
        <details v-if="message.stages" class="mt-2 text-xs text-warm-800/50">
          <summary class="cursor-pointer hover:text-warm-800/80 select-none">⏱ 阶段耗时详情</summary>
          <div class="mt-1 ml-2 font-mono leading-relaxed">{{ message.stages }}</div>
        </details>
      </div>
      <div class="text-xs text-warm-800/40 mt-1">{{ message.time }}</div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { renderMarkdown } from '@/composables/useMarkdown'
import { useDishIndexStore } from '@/stores/dishIndex'

const props = defineProps({ message: { type: Object, required: true } })
const dishIndex = useDishIndexStore()

const renderedContent = computed(() => renderMarkdown(props.message.content || '', dishIndex.state))
</script>
