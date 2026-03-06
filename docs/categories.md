---
title: 文章分类
description: 按分类浏览所有文章
---

# 文章分类

<div class="category-grid">
  <a href="/pytorch_read/" class="category-card">
    <h3>PyTorch 源码阅读笔记</h3>
    <p>8 篇文章</p>
  </a>
  <a href="/python/" class="category-card">
    <h3>Python</h3>
    <p>2 篇文章</p>
  </a>
  <a href="/llm/" class="category-card">
    <h3>深度学习</h3>
    <p>1 篇文章</p>
  </a>
  <a href="/paper/" class="category-card">
    <h3>论文阅读</h3>
    <p>1 篇文章</p>
  </a>
  <a href="/other/" class="category-card">
    <h3>其他</h3>
    <p>3 篇文章</p>
  </a>
</div>

<style>
.category-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin-top: 2rem;
}

.category-card {
  display: block;
  padding: 1.5rem;
  border: 1px solid var(--vp-c-border);
  border-radius: 8px;
  transition: all 0.3s ease;
  text-decoration: none;
  color: var(--vp-c-text-1);
}

.category-card:hover {
  border-color: var(--vp-c-brand);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.category-card h3 {
  margin: 0 0 0.5rem 0;
  color: var(--vp-c-brand);
}

.category-card p {
  margin: 0;
  color: var(--vp-c-text-2);
}
</style>
