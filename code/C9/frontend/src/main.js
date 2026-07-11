import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import { router } from './router'
import { installDishPreview } from './composables/useDishPreview'
import { applyTailwindConfig } from './assets/tailwind-config.js'

import './assets/main.css'

applyTailwindConfig()

const app = createApp(App)
app.use(createPinia())
app.use(router)
app.mount('#app')

// 全局副作用：菜名图片预览单例（DOM 事件委托）
installDishPreview()
