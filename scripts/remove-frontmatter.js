import { readFileSync, writeFileSync, readdirSync, statSync } from 'fs'
import { join } from 'path'

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
 * 移除 frontmatter
 */
function removeFrontmatter(filePath) {
  const content = readFileSync(filePath, 'utf-8')

  // 移除开头的 frontmatter（从第一个 --- 到第二个 ---）
  const cleaned = content.replace(/^---\n[\s\S]*?---\n/, '')

  if (content !== cleaned) {
    writeFileSync(filePath, cleaned, 'utf-8')
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
    if (removeFrontmatter(filePath)) {
      updatedCount++
    }
  } catch (error) {
    console.error(`处理文件失败: ${filePath}`, error.message)
  }
}

console.log(`\n✅ 已清理 ${updatedCount} 个文件的 frontmatter`)
