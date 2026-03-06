import { readFileSync, readdirSync, statSync, existsSync, writeFileSync } from 'fs'
import { join } from 'path'
import matter from 'gray-matter'

const DOCS_DIR = new URL('../docs', import.meta.url).pathname

/**
 * 从 markdown 文件中提取标题和日期
 */
function extractTitleAndDate(filePath) {
  try {
    const content = readFileSync(filePath, 'utf-8')
    const { data, content: markdownContent } = matter(content)

    // 优先使用 frontmatter 中的 title 和 date
    let title = data.title
    let date = data.date

    // 如果 frontmatter 中没有，从正文中提取
    if (!title) {
      const h1Match = markdownContent.match(/^#\s+(.+)$/m)
      if (h1Match) {
        title = h1Match[1].trim()
      } else {
        title = filePath.split('/').pop().replace(/\.md$/, '')
      }
    }

    // 从正文中提取发布日期
    if (!date) {
      const dateMatch = content.match(/发布日期[：:]\s*(\d{4}-\d{2}-\d{2})/)
      if (dateMatch) {
        date = dateMatch[1]
      }
    }

    return { title, date }
  } catch (error) {
    console.error(`Error reading ${filePath}:`, error.message)
    return { title: null, date: null }
  }
}

/**
 * 获取所有 markdown 文件
 */
function getAllMarkdownFiles(dirPath) {
  const files = []
  const entries = readdirSync(dirPath)

  for (const entry of entries) {
    const fullPath = join(dirPath, entry)
    const stat = statSync(fullPath)

    if (stat.isDirectory() && !entry.startsWith('.') && entry !== '.vitepress' && entry !== 'public') {
      files.push(...getAllMarkdownFiles(fullPath))
    } else if (entry.endsWith('.md') && entry !== 'index.md' && entry !== 'about.md') {
      files.push(fullPath)
    }
  }

  return files
}

/**
 * 生成时间线数据
 */
function generateTimeline() {
  const allFiles = getAllMarkdownFiles(DOCS_DIR)
  const timeline = []

  for (const filePath of allFiles) {
    const { title, date } = extractTitleAndDate(filePath)

    if (title && date) {
      // 获取相对路径
      const relativePath = filePath.replace(DOCS_DIR, '').replace(/\.md$/, '')

      timeline.push({
        title,
        date,
        link: relativePath,
        timestamp: new Date(date).getTime()
      })
    }
  }

  // 按日期降序排序
  timeline.sort((a, b) => b.timestamp - a.timestamp)

  return timeline
}

// 生成并导出配置
const timeline = generateTimeline()

// 将配置写入文件
const outputPath = new URL('../docs/public/timeline-data.json', import.meta.url)
writeFileSync(outputPath, JSON.stringify(timeline, null, 2), 'utf-8')

console.log('✅ 文章时间线已生成:', outputPath.pathname)
console.log('📝 文章总数:', timeline.length)

// 按年份统计
const yearCount = {}
timeline.forEach(item => {
  const year = item.date.substring(0, 4)
  yearCount[year] = (yearCount[year] || 0) + 1
})
console.log('📅 年份分布:', yearCount)
