/**
 * Vue Router 配置。
 * 守卫：未登录跳转 /auth；非 admin 试图进入 /admin /retrieve 时跳 /chat。
 */
import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const Home = () => import('@/views/Home.vue')
const Auth = () => import('@/views/Auth.vue')
const Chat = () => import('@/views/Chat.vue')
const Admin = () => import('@/views/Admin.vue')
const Retrieve = () => import('@/views/Retrieve.vue')
const NotFound = () => import('@/views/NotFound.vue')

const routes = [
  { path: '/', name: 'home', component: Home, meta: { requiresAuth: false } },
  { path: '/auth', name: 'auth', component: Auth, meta: { requiresAuth: false } },
  { path: '/chat', name: 'chat', component: Chat, meta: { requiresAuth: true } },
  { path: '/admin', name: 'admin', component: Admin, meta: { requiresAuth: true, requiresAdmin: true } },
  { path: '/retrieve', name: 'retrieve', component: Retrieve, meta: { requiresAuth: true, requiresAdmin: true } },
  { path: '/:pathMatch(.*)*', name: 'not-found', component: NotFound },
]

export const router = createRouter({
  history: createWebHistory('/'),
  routes,
  scrollBehavior: () => ({ top: 0 }),
})

let _firstGuardRun = true
router.beforeEach((to) => {
  const auth = useAuthStore()
  // 首次载入校验一次 token，过期则清掉
  if (_firstGuardRun) {
    _firstGuardRun = false
    auth.fetchMe()
  }
  if (to.meta.requiresAuth && !auth.isLoggedIn) {
    return { path: '/auth', query: { redirect: to.fullPath } }
  }
  if (to.meta.requiresAdmin && !auth.isAdmin) {
    return { path: '/chat' }
  }
  if (to.name === 'auth' && auth.isLoggedIn) {
    return { path: '/chat' }
  }
  return true
})
