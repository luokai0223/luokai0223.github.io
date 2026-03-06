import { defineConfig } from 'vitepress'

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
      { text: '分类', link: '/categories' },
      { text: '标签', link: '/tags' },
      { text: '关于', link: '/about' },
      {
        text: 'GitHub',
        link: 'https://github.com/luokai0223/luokai0223.github.io'
      }
    ],

    // 侧边栏 - 按分类组织
    sidebar: [
      {
        text: 'PyTorch 源码阅读笔记',
        collapsible: true,
        collapsed: false,
        items: [
          { text: '自动微分张量库', link: '/pytorch_read/autograd' },
          { text: '编译构建', link: '/pytorch_read/build' },
          { text: '编译构建 v2', link: '/pytorch_read/build_v2' },
          { text: '派发器', link: '/pytorch_read/dispatcher' },
          { text: '算子调用', link: '/pytorch_read/operators_call' },
          { text: '算子注册', link: '/pytorch_read/operators_register' },
          { text: 'TorchScript', link: '/pytorch_read/torchscript' },
          { text: 'TorchDynamo', link: '/pytorch_read/torchdynamo' }
        ]
      },
      {
        text: 'Python',
        collapsible: true,
        collapsed: true,
        items: [
          { text: '代码编译运行过程', link: '/python/python_compile' },
          { text: 'Python 虚拟机', link: '/python/python_vm' }
        ]
      },
      {
        text: '深度学习',
        collapsible: true,
        collapsed: true,
        items: [
          { text: 'ChatGLM 并发改造', link: '/llm/chatglm-concurrent' }
        ]
      },
      {
        text: '论文阅读',
        collapsible: true,
        collapsed: true,
        items: [
          { text: '遮挡环境下的面部识别', link: '/paper/occlusion_face' }
        ]
      },
      {
        text: '其他',
        collapsible: true,
        collapsed: true,
        items: [
          { text: 'Stable Diffusion LoRA 训练', link: '/other/stable_diffusion_lora' },
          { text: 'AMD CPU ML 环境安装', link: '/other/amdcpu_ml_install' },
          { text: 'VSCode Server 部署', link: '/other/vscode_server_deploy' }
        ]
      }
    ],

    // 社交链接
    socialLinks: [
      { icon: 'github', link: 'https://github.com/luokai0223' }
    ],

    // 页脚
    footer: {
      message: '基于 MIT 许可发布',
      copyright: 'Copyright © 2023-present K'
    },

    // 编辑链接
    editLink: {
      pattern: 'https://github.com/luokai0223/luokai0223.github.io/edit/main/docs/:path',
      text: '在 GitHub 上编辑此页'
    },

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

  // Vite 服务器配置
  vite: {
    server: {
      host: '0.0.0.0',
      port: 4000
    }
  }
})
