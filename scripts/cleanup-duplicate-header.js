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
 * 移除重复添加的标题和日期
 */
function removeDuplicateHeader(filePath) {
  const content = readFileSync(filePath, 'utf-8')
  const { data, content: markdownContent } = matter(content)

  // 移除开头的 # 标题
  let cleanContent = markdownContent.replace(/^# .+\n\n📅 发布日期：.+\n\n/, '')

  // 如果内容被修改了，写回文件
  if (cleanContent !== markdownContent) {
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
    if (removeDuplicateHeader(filePath)) {
      updatedCount++
    }
  } catch (error) {
    console.error(`处理文件失败: ${filePath}`, error.message)
  }
}

console.log(`\n✅ 已清理 ${updatedCount} 个文件`)
