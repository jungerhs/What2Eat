# C9 Landing Page · Vue 重构

本目录是用 **Vue 3 + Vue Router + Pinia + Vite** 重写的 C9 前端。
原版（HTML + Tailwind CDN + 原生 JS）位于 `../landing-page-legacy/`，可在验证通过前保留作对照。

## 目录结构

```
landing-page/
├── index.html         # Vite 入口
├── vite.config.js     # 配置 + dev proxy
├── package.json
├── src/
│   ├── main.js        # 启动入口
│   ├── App.vue        # router-view 容器
│   ├── router/        # 路由 + 守卫
│   ├── stores/        # Pinia: auth / stats / profile / dishIndex
│   ├── api/           # apiFetch 统一封装
│   ├── utils/         # escape / 时间格式化
│   ├── composables/   # useToast / useMarkdown / useDishPreview
│   ├── components/    # MessageBubble / ProfilePanel
│   ├── views/         # Home / Auth / Chat / Admin / Retrieve / NotFound
│   └── assets/        # main.css + tailwind-config.js
```

## 路由

| 路径        | 组件       | 权限 |
| ----------- | ---------- | ---- |
| `/`         | Home       | 公开 |
| `/auth`     | Auth       | 公开 |
| `/chat`     | Chat       | 登录 |
| `/admin`    | Admin      | admin |
| `/retrieve` | Retrieve   | admin |

## 开发

```powershell
cd "F:\cook项目\cook\code\C9\landing-page"
npm install
npm run dev     # http://localhost:5173
```

`vite.config.js` 默认把 `/api/*` 代理到 `http://localhost:8000`。

## 生产构建

```powershell
npm run build   # 产物在 dist/
```

`api_server.py` 已自动挂载 `landing-page/dist/` 作为静态目录，并把 `/{full_path:path}` 的 GET 走 SPA fallback（Vue Router 用 history 模式必须 fallback 到 `index.html`）。
`/api/*` 路径继续由 FastAPI 路由处理，互不干扰。

启动：

```powershell
python api_server.py     # http://localhost:8000 即可访问整套 SPA + /api
```

> 开发态仍可用 `npm run dev` 单独启 Vite（5173），但务必让 `npm install && npm run build` 至少运行过一次，否则启动 `api_server.py` 会提示「前端未构建」。

## 与原版的差异（仅前端）

- 全部逻辑切到 Vue 3 Composition API；状态用 Pinia。
- 路由统一管理，未登录跳 `/auth`、非 admin 跳 `/chat`。
- Markdown 渲染：`marked` + `dompurify` 改为 npm 包（取代 CDN），菜名链接与浮层预览逻辑 1:1 复用。
- Toast / 模态 / 时间统计等由 Vue 组件统一处理（仍保持视觉风格）。
- 流式响应：直接用 `fetch` + ReadableStream，未引入额外 SSE 库，保持与原 `app.html` 行为一致。

## 与原 HTML 的对应

| 原文件 | 现文件 |
| --- | --- |
| `landing-page-legacy/index.html`   | `src/views/Home.vue` |
| `landing-page-legacy/auth.html`    | `src/views/Auth.vue` |
| `landing-page-legacy/app.html`     | `src/views/Chat.vue` + `src/components/MessageBubble.vue` |
| `landing-page-legacy/admin.html`   | `src/views/Admin.vue` |
| `landing-page-legacy/retrieve.html`| `src/views/Retrieve.vue` |
| `landing-page-legacy/js/*`         | 切到 `src/composables/*` / `src/stores/*` |

## 已知未对接/可选后端（与原前端一致）

- `/api/dish-images` 当前仅在 `dishIndex` store 内调用；后端若未实现则静默忽略。
- `/api/user/profile` 同样静默处理（profile 卡片留空）。

## 后端契约（保持不变）

聊天主链路、用户/管理后台、检索测试等端点保持原 FastAPI 契约，无需改动后端。
