import { readFileSync, writeFileSync, readdirSync, statSync } from 'fs'
import { join } from 'path'
import matter from 'gray-matter'

const DOCS_DIR = new URL('../docs', import.meta.url).pathname

/**
 * 转换日期格式为 YYYY-MM-DD
 */
function convertDateFormat(dateValue) {
  if (!dateValue) return null

  // 转换为字符串
  const dateStr = String(dateValue)

  // 如果已经是 YYYY-MM-DD 格式，直接返回
  if (/^\d{4}-\d{2}-\d{2}$/.test(dateStr.trim())) {
    return dateStr.trim()
  }

  // 尝试解析各种日期格式
  try {
    const date = new Date(dateStr)
    if (isNaN(date.getTime())) {
      console.warn(`无法解析日期: ${dateStr}`)
      return dateStr
    }

    // 转换为 YYYY-MM-DD 格式
    const year = date.getFullYear()
    const month = String(date.getMonth() + 1).padStart(2, '0')
    const day = String(date.getDate()).padStart(2, '0')

    return `${year}-${month}-${day}`
  } catch (error) {
    console.warn(`日期转换失败: ${dateStr}`, error.message)
    return dateStr
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
    } else if (entry.endsWith('.md') && entry !== 'index.md') {
      files.push(fullPath)
    }
  }

  return files
}

/**
 * 更新文件的日期格式
 */
function updateFileDate(filePath) {
  const content = readFileSync(filePath, 'utf-8')
  const { data, content: markdownContent } = matter(content)

  let updated = false

  // 更新日期格式
  if (data.date) {
    const oldDate = data.date
    const newDate = convertDateFormat(oldDate)

    if (oldDate !== newDate) {
      data.date = newDate
      updated = true
      console.log(`✓ ${filePath}: ${oldDate} -> ${newDate}`)
    }
  }

  // 如果有更新，写回文件
  if (updated) {
    const newContent = matter.stringify(markdownContent, data)
    writeFileSync(filePath, newContent, 'utf-8')
  }
}

// 处理所有 markdown 文件
const allFiles = getAllMarkdownFiles(DOCS_DIR)
console.log(`找到 ${allFiles.length} 个 markdown 文件`)

let updatedCount = 0
for (const filePath of allFiles) {
  try {
    updateFileDate(filePath)
    updatedCount++
  } catch (error) {
    console.error(`处理文件失败: ${filePath}`, error.message)
  }
}

console.log(`\n✅ 已处理 ${updatedCount} 个文件`)
