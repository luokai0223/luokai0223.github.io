import { readFileSync, readdirSync, statSync, existsSync } from 'fs'
import { join, relative } from 'path'
import matter from 'gray-matter'

const DOCS_DIR = new URL('../docs', import.meta.url).pathname

/**
 * 从 markdown 文件中提取标题
 */
function extractTitle(filePath) {
  try {
    const content = readFileSync(filePath, 'utf-8')
    const { data, content: markdownContent } = matter(content)

    // 优先使用 frontmatter 中的 title
    if (data.title) {
      return data.title
    }

    // 其次尝试从第一个 h1 标题提取
    const h1Match = markdownContent.match(/^#\s+(.+)$/m)
    if (h1Match) {
      return h1Match[1].trim()
    }

    // 最后使用文件名
    return filePath.split('/').pop().replace(/\.md$/, '')
  } catch (error) {
    console.error(`Error reading ${filePath}:`, error.message)
    return filePath.split('/').pop().replace(/\.md$/, '')
  }
}

/**
 * 获取目录下的所有 markdown 文件（排除 index.md）
 */
function getMarkdownFiles(dirPath) {
  const files = []

  if (!existsSync(dirPath)) {
    return files
  }

  const entries = readdirSync(dirPath)

  for (const entry of entries) {
    const fullPath = join(dirPath, entry)
    const stat = statSync(fullPath)

    if (stat.isDirectory()) {
      // 递归处理子目录
      files.push(...getMarkdownFiles(fullPath))
    } else if (entry.endsWith('.md') && entry !== 'index.md') {
      files.push(fullPath)
    }
  }

  return files.sort()
}

/**
 * 生成单个目录的侧边栏配置
 */
function generateCategorySidebar(docsDir, categoryDir) {
  const categoryPath = join(docsDir, categoryDir)
  const mdFiles = getMarkdownFiles(categoryPath)

  const items = []

  // 不再添加概述链接

  // 添加所有文章
  for (const filePath of mdFiles) {
    const relativePath = relative(categoryPath, filePath)  // 相对于分类目录
    const link = '/' + relativePath.replace(/\.md$/, '')
    const title = extractTitle(filePath)

    items.push({
      text: title,
      link: link
    })
  }

  return {
    base: `/${categoryDir}/`,
    items
  }
}

/**
 * 将侧边栏配置转换为可折叠的分组
 */
function createCollapsibleGroup(docsDir, categoryDir, categoryTitle) {
  const categoryPath = join(docsDir, categoryDir)
  const mdFiles = getMarkdownFiles(categoryPath)

  const items = []

  // 不再添加概述链接

  // 添加该分类下的所有文章
  for (const filePath of mdFiles) {
    const relativePath = relative(categoryPath, filePath)
    const link = `/${categoryDir}/` + relativePath.replace(/\.md$/, '')
    const title = extractTitle(filePath)

    items.push({
      text: title,
      link: link
    })
  }

  return {
    text: categoryTitle,
    collapsed: false,
    collapsible: true,
    items: items
  }
}

/**
 * 获取所有分类目录
 */
function getCategories(docsDir) {
  const categories = []
  const entries = readdirSync(docsDir)

  for (const entry of entries) {
    const fullPath = join(docsDir, entry)
    const stat = statSync(fullPath)

    if (stat.isDirectory() && !entry.startsWith('.') && entry !== '.vitepress') {
      // 检查目录中是否有 markdown 文件
      const mdFiles = getMarkdownFiles(fullPath)
      const hasIndex = existsSync(join(fullPath, 'index.md'))
      if (mdFiles.length > 0 || hasIndex) {
        categories.push(entry)
      }
    }
  }

  return categories.sort()
}

/**
 * 生成侧边栏配置
 */
function generateSidebar() {
  const categories = getCategories(DOCS_DIR)

  // 根路径侧边栏 - 显示所有分类（可折叠）
  const rootItems = []

  // 添加所有分类目录
  for (const category of categories) {
    const categoryPath = join(DOCS_DIR, category)
    const indexPath = join(categoryPath, 'index.md')

    // 尝试获取分类名称
    let categoryTitle = category
    if (existsSync(indexPath)) {
      categoryTitle = extractTitle(indexPath) || category
    }

    // 创建可折叠的分类组，包含所有文章
    rootItems.push(createCollapsibleGroup(DOCS_DIR, category, categoryTitle))
  }

  // 不再添加"关于"页面到侧边栏
  // const aboutPath = join(DOCS_DIR, 'about.md')
  // if (existsSync(aboutPath)) {
  //   rootItems.push({
  //     text: '关于',
  //     link: '/about'
  //   })
  // }

  // 所有路径都使用同一个侧边栏配置（根路径的折叠目录）
  return {
    '/': rootItems
  }
}

// 生成并导出配置
const sidebar = generateSidebar()

// 将配置写入文件
const outputPath = new URL('../docs/.vitepress/sidebar-auto-generated.json', import.meta.url)
import { writeFileSync } from 'fs'
writeFileSync(outputPath, JSON.stringify(sidebar, null, 2), 'utf-8')

// 计算分类数量
const categoryCount = sidebar['/'].filter(item => item.collapsible).length

console.log('✅ 侧边栏配置已自动生成:', outputPath.pathname)
console.log('📁 检测到的分类数量:', categoryCount)
console.log('📝 所有路径使用统一的侧边栏配置（折叠目录）')
