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
 * 检查文件是否已有日期显示块
 */
function hasDateBlock(content) {
  return content.includes('::: info 📅 文章发布日期')
}

/**
 * 为文件添加日期显示块
 */
function addDateBlock(filePath) {
  const content = readFileSync(filePath, 'utf-8')

  // 如果已经有日期显示块，跳过
  if (hasDateBlock(content)) {
    return false
  }

  const { data, content: markdownContent } = matter(content)

  // 检查是否有日期字段
  if (!data.date) {
    return false
  }

  // 重建文件内容：frontmatter + 日期显示块 + markdown内容
  const date = data.date
  const dateBlock = `::: info 📅 文章发布日期\n${date}\n:::\n\n`
  const newContent = matter.stringify(dateBlock + markdownContent, data)

  writeFileSync(filePath, newContent, 'utf-8')
  console.log(`✓ ${filePath}`)
  return true
}

// 处理所有 markdown 文件
const allFiles = getAllMarkdownFiles(DOCS_DIR)
console.log(`找到 ${allFiles.length} 个 markdown 文件`)

let updatedCount = 0
for (const filePath of allFiles) {
  try {
    if (addDateBlock(filePath)) {
      updatedCount++
    }
  } catch (error) {
    console.error(`处理文件失败: ${filePath}`, error.message)
  }
}

console.log(`\n✅ 已为 ${updatedCount} 个文件添加日期显示`)
