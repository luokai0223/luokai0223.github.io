---
layout: home
hero:
  name: "K's blog"
  text: 个人技术博客
  actions:
    - theme: brand
      text: 浏览文章
      link: /latest
    - theme: alt
      text: 关于我
      link: /about
---

<script setup>
import { onMounted, ref } from 'vue'

const posts = ref([])
const latestPostLink = ref('/llm/chatglm-concurrent') // 默认值

// 解析文章日期
function parseDate(dateStr) {
  // 匹配 "发布日期：2024-06-24" 格式
  const match = dateStr.match(/发布日期：(\d{4}-\d{2}-\d{2})/)
  if (match) {
    return new Date(match[1])
  }
  return new Date(0) // 如果没有日期，返回最早日期
}

// 获取最新文章链接
async function fetchLatestPostLink() {
  try {
    const response = await fetch('/.vitepress/sidebar-auto-generated.json')
    const sidebar = await response.json()

    const allPosts = []
    const categories = sidebar['/'] || []

    for (const category of categories) {
      if (category.items) {
        for (const post of category.items) {
          allPosts.push({
            title: post.text,
            link: post.link,
            category: category.text
          })
        }
      }
    }

    // 获取每篇文章的内容并解析日期
    const postsWithDates = await Promise.all(
      allPosts.map(async (post) => {
        try {
          const postResponse = await fetch(`${post.link}.md`)
          const content = await postResponse.text()
          const date = parseDate(content)
          return { ...post, date }
        } catch {
          return { ...post, date: new Date(0) }
        }
      })
    )

    // 按日期排序，获取最新的
    postsWithDates.sort((a, b) => b.date - a.date)

    if (postsWithDates.length > 0) {
      return postsWithDates[0].link
    }
  } catch (error) {
    console.error('Failed to fetch latest post:', error)
  }
  return '/llm/chatglm-concurrent' // 出错时返回默认值
}

onMounted(async () => {
  try {
    const response = await fetch('/.vitepress/sidebar-auto-generated.json')
    const sidebar = await response.json()

    const allPosts = []
    const categories = sidebar['/'] || []

    for (const category of categories) {
      if (category.items) {
        for (const post of category.items) {
          allPosts.push({
            title: post.text,
            link: post.link,
            category: category.text
          })
        }
      }
    }

    posts.value = allPosts.slice(0, 9)

    // 获取最新文章链接
    latestPostLink.value = await fetchLatestPostLink()

    // 更新hero action的链接
    const heroAction = document.querySelector('.VPButton[href="/llm/chatglm-concurrent"]')
    if (heroAction && latestPostLink.value) {
      heroAction.setAttribute('href', latestPostLink.value)
    }
  } catch (error) {
    console.error('Failed to load posts:', error)
  }
})
</script>

<style>
.VPHome {
  display: block;
}

.VPHero {
  display: block;
}

.posts-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 2rem;
  max-width: 1152px;
  margin: 0 auto;
  padding: 0 1.5rem 4rem;
}

.post-card,
.post-card:link,
.post-card:visited,
.post-card:hover,
.post-card:active {
  text-decoration: none;
}

.post-card {
  display: flex;
  flex-direction: column;
  padding: 2rem;
  background: linear-gradient(135deg, var(--vp-c-bg-soft) 0%, var(--vp-c-bg) 100%);
  border-radius: 16px;
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  border: 1px solid var(--vp-c-divider);
  height: 100%;
  position: relative;
  overflow: hidden;
}

.post-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--vp-c-brand), var(--vp-c-brand-light));
  opacity: 0;
  transition: opacity 0.4s ease;
}

.post-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 24px rgba(0, 0, 0, 0.1);
  border-color: transparent;
  text-decoration: none;
}

.post-card:hover::before {
  opacity: 1;
}

.post-title {
  font-size: 1.15rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin-bottom: 1rem;
  line-height: 1.7;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  flex-grow: 1;
  text-decoration: none;
}

.post-category {
  font-size: 0.8rem;
  color: var(--vp-c-brand);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  padding: 0.4rem 0.8rem;
  background: var(--vp-c-brand-soft);
  border-radius: 6px;
  display: inline-block;
  align-self: flex-start;
  text-decoration: none;
}

.post-card:hover .post-title {
  color: var(--vp-c-brand);
}

@media (max-width: 768px) {
  .posts-grid {
    grid-template-columns: 1fr;
    padding: 0 1rem 2rem;
    gap: 1.5rem;
  }

  .post-card {
    padding: 1.5rem;
  }
}

@media (min-width: 769px) and (max-width: 1024px) {
  .posts-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>

<div class="posts-grid">
  <a v-for="post in posts" :key="post.link" :href="post.link" class="post-card">
    <div class="post-title">{{ post.title }}</div>
    <div class="post-category">{{ post.category }}</div>
  </a>
</div>