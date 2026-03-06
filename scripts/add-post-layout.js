import { readFileSync, writeFileSync, readdirSync, statSync } from 'fs'
import { join } from 'path'
import matter from 'gray-matter'

const DOCS_DIR = new URL('../docs', import.meta.url).pathname

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
    } else if (entry.endsWith('.md') && entry !== 'index.md') {
      files.push(fullPath)
    }
  }

  return files
}

/**
 * 为文章添加布局配置并移除日期显示块
 */
function updateArticleLayout(filePath) {
  const content = readFileSync(filePath, 'utf-8')
  const { data, content: markdownContent } = matter(content)

  let updated = false

  // 添加自定义布局
  if (!data.layout && data.date) {
    data.layout = 'PostLayout'
    updated = true
  }

  // 移除之前添加的日期显示块（如果在内容开头）
  let cleanContent = markdownContent
  const dateBlockPattern = /^::: info 📅 文章发布日期\n.*?\n:::\s*\n*/
  if (dateBlockPattern.test(cleanContent)) {
    cleanContent = cleanContent.replace(dateBlockPattern, '')
    updated = true
  }

  if (updated) {
    const newContent = matter.stringify(cleanContent, data)
    writeFileSync(filePath, newContent, 'utf-8')
    console.log(`✓ ${filePath}`)
    return true
  }
  return false
}

// 处理所有 markdown 文件
const allFiles = getAllMarkdownFiles(DOCS_DIR)
console.log(`找到 ${allFiles.length} 个 markdown 文件`)

let updatedCount = 0
for (const filePath of allFiles) {
  try {
    if (updateArticleLayout(filePath)) {
      updatedCount++
    }
  } catch (error) {
    console.error(`处理文件失败: ${filePath}`, error.message)
  }
}

console.log(`\n✅ 已更新 ${updatedCount} 个文件`)
