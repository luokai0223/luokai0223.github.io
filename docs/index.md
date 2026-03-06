---
title: 首页
description: K's blog - 个人技术博客
---

# K's blog

<div class="hero">
  <!-- <p class="tagline">个人笔记记录</p> -->
  <p class="description">
    个人技术笔记
  </p>
</div>

<!-- <div class="stats">
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
</div> -->

## 文章时间线

<script setup>
import { onMounted, ref } from 'vue'

const timelineData = ref([])

onMounted(async () => {
  try {
    const response = await fetch('/timeline-data.json')
    timelineData.value = await response.json()
  } catch (error) {
    console.error('Failed to load timeline data:', error)
  }
})
</script>

<div class="timeline" v-if="timelineData.length > 0">
  <div class="timeline-item" v-for="item in timelineData" :key="item.link">
    <div class="timeline-date">{{ item.date }}</div>
    <div class="timeline-content">
      <a :href="item.link" class="timeline-title">{{ item.title }}</a>
    </div>
  </div>
</div>

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

.timeline {
  max-width: 800px;
  margin: 2rem auto;
  padding: 0 1rem;
}

.timeline-item {
  display: flex;
  gap: 1.5rem;
  padding: 1rem 0;
  border-bottom: 1px solid var(--vp-c-divider);
  transition: background-color 0.2s;
}

.timeline-item:last-child {
  border-bottom: none;
}

.timeline-item:hover {
  background-color: var(--vp-c-default-soft);
  margin: 0 -1rem;
  padding: 1rem;
  border-radius: 8px;
}

.timeline-date {
  flex-shrink: 0;
  width: 100px;
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
  font-family: var(--vp-font-family-mono);
  padding-top: 2px;
}

.timeline-content {
  flex: 1;
  min-width: 0;
}

.timeline-title {
  font-size: 1.1rem;
  font-weight: 500;
  color: var(--vp-c-text-1);
  text-decoration: none;
  display: block;
  line-height: 1.6;
  transition: color 0.2s;
}

.timeline-title:hover {
  color: var(--vp-c-brand);
}
</style>
