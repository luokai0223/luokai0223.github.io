import { defineConfig } from 'vitepress'
import autoSidebar from './sidebar-auto-generated.json'

export default defineConfig({
  // 站点配置
  title: "K's blog",
  description: '个人笔记记录',
  lang: 'zh-CN',

  // 外观
  appearance: 'light',

  // SEO 和头部配置
  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }],
    ['meta', { name: 'theme-color', content: '#3c8772' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:locale', content: 'zh-CN' }],
    ['meta', { property: 'og:title', content: "K's blog" }],
    ['meta', { property: 'og:site_name', content: "K's blog" }],
    ['meta', { property: 'og:url', content: 'https://luokai0223.github.io/' }],
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
    // Google Analytics
    ['script', { src: 'https://www.googletagmanager.com/gtag/js?id=G-355180992', async: true }],
    ['script', {}, `
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-355180992');
    `]
  ],

  // 主题配置
  themeConfig: {
    // 导航栏
    nav: [
      { text: '首页', link: '/' },
      { text: '关于', link: '/about' },
      {
        text: 'GitHub',
        link: 'https://github.com/luokai0223/luokai0223.github.io'
      }
    ],

    // 侧边栏 - 自动生成
    // 运行 pnpm generate-sidebar 或 pnpm docs:dev 会自动更新
    sidebar: autoSidebar,

    // 社交链接
    // socialLinks: [
    //   { icon: 'github', link: 'https://github.com/luokai0223' }
    // ],

    // 页脚
    footer: {
      message: '基于 MIT 许可发布',
      copyright: 'Copyright © 2023-present K'
    },

    // 编辑链接
    // editLink: {
    //   pattern: 'https://github.com/luokai0223/luokai0223.github.io/edit/main/docs/:path',
    //   text: '在 GitHub 上编辑此页'
    // },

    // 最后更新
    lastUpdated: {
      text: '最后更新',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'medium'
      }
    },

    // 目录
    outline: {
      level: [2, 3],
      label: '页面导航'
    }
  },

  // Markdown 配置
  markdown: {
    // 行号
    lineNumbers: true,

    // 代码高亮
    theme: 'github-light',

    // 语言
    languages: ['python', 'cpp', 'javascript', 'bash', 'yaml', 'markdown']
  },

  // 构建配置
  base: '/',
  outDir: '../.vitepress/dist',

  // 干净的 URL
  cleanUrls: true,

  // Sitemap
  sitemap: {
    hostname: 'https://luokai0223.github.io'
  },

  // 最后更新的 Git 配置
  lastUpdated: true,

  // Vite 配置
  vite: {
    server: {
      host: '0.0.0.0',
      port: 5173
    }
  }
})
