import { createRouter, createWebHashHistory } from 'vue-router'

export const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    // / 路由的 component 不重要：App.vue 根据 route.name 自己渲染 ChatPanel，
    // 这里挂个占位空组件避免 vue-router 报 missing component。
    { path: '/', name: 'chat', component: { template: '<div />' } },
    { path: '/docs/:kbId', name: 'docs', component: () => import('@/views/DocsView.vue') },
    { path: '/kb', name: 'kb', component: () => import('@/views/KBView.vue') },
    { path: '/search', name: 'search', component: () => import('@/views/SearchView.vue') },
    { path: '/settings', name: 'settings', component: () => import('@/views/SettingsView.vue') },
  ],
})
