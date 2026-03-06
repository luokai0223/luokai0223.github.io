---
title: 首页
description: K's blog - 个人技术博客
---

# K's blog

<div class="hero">
  <p class="tagline">个人笔记记录</p>
  <p class="description">
    记录 PyTorch 源码阅读、Python 学习、深度学习、论文阅读等技术内容
  </p>
</div>

<div class="stats">
  <div class="stat-item">
    <div class="stat-number">15</div>
    <div class="stat-label">篇文章</div>
  </div>
  <div class="stat-item">
    <div class="stat-number">5</div>
    <div class="stat-label">个分类</div>
  </div>
  <div class="stat-item">
    <div class="stat-number">PyTorch</div>
    <div class="stat-label">主要关注</div>
  </div>
</div>

## 最新文章

<VPLatestPosts />

<style>
.hero {
  text-align: center;
  padding: 3rem 0;
}

.tagline {
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--vp-c-brand);
  margin-bottom: 1rem;
}

.description {
  font-size: 1.1rem;
  color: var(--vp-c-text-2);
  max-width: 600px;
  margin: 0 auto;
}

.stats {
  display: flex;
  justify-content: center;
  gap: 3rem;
  margin: 3rem 0;
  padding: 2rem;
  background: var(--vp-c-default-soft);
  border-radius: 12px;
}

.stat-item {
  text-align: center;
}

.stat-number {
  font-size: 2rem;
  font-weight: 700;
  color: var(--vp-c-brand);
}

.stat-label {
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
  margin-top: 0.5rem;
}
</style>
