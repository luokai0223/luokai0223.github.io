---
layout: page
title: 最新文章
---

<script setup>
import { onMounted } from 'vue'

// 解析文章日期
function parseDate(dateStr) {
  // 匹配 "发布日期：2024-06-24" 格式
  const match = dateStr.match(/发布日期：(\d{4}-\d{2}-\d{2})/)
  if (match) {
    return new Date(match[1])
  }
  return new Date(0) // 如果没有日期，返回最早日期
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
      // 重定向到最新文章
      window.location.href = postsWithDates[0].link
    }
  } catch (error) {
    console.error('Failed to redirect to latest post:', error)
    // 出错时跳转到默认文章
    window.location.href = '/llm/chatglm-concurrent'
  }
})
</script>

<div style="display: flex; justify-content: center; align-items: center; min-height: 50vh;">
  <p>正在跳转到最新文章...</p>
</div>
