---
title: 'md2pdf——一个极致简洁轻量化的md-pdf格式转换工具'
date: 2026-08-18 03:14:41
tags: ["技术"]

categories: ["原创"]
---

在技术写作和文档管理中，Markdown 已成为事实上的标准格式。然而，将 Markdown 导出为 PDF 时，常见的工具往往引入过多额外元素——封面、目录、页眉页脚、页码——这些附加物在某些场景下是必要的，但在另一些场景下却成为干扰。

**md2pdf** 正是为此而生：一个极简的 Markdown → PDF 命令行工具，遵循"内容即输出"的原则，不添加任何多余元素。

Github: [https://github.com/Hanerson/md2pdf](https://github.com/Hanerson/md2pdf)

[点击此处查看本文档经过该工具转换后的产出效果PDF](https://ref.hanerson.com/#/post/20260818)

---

## 设计原则

| 原则 | 说明 |
| --- | --- |
| **一一对应** | 内容按文档流连续渲染，不插入任何附加页 |
| **无附加物** | 没有封面、目录、页眉、页脚、页码 |
| **极简命令** | `md2pdf doc.md` 一条命令搞定 |
| **经典字体** | 中文 → 宋体 (SimSun)，英文 → Times New Roman |
| **最小参数** | 仅 5 个可选参数，学习成本为零 |

---
<!-- more -->
## 快速开始

### 前置要求

- Node.js >= 20

### 安装

```bash
# 克隆或进入项目目录
cd md2pdf

# 安装依赖
npm install

# 全局安装（获得 md2pdf 命令）
npm install -g .

# 首次使用前安装 Chromium
npx playwright install chromium
```

> **中国大陆用户**：可设置镜像加速下载
> ```bash
> PLAYWRIGHT_DOWNLOAD_HOST=https://npmmirror.com/mirrors/playwright npx playwright install chromium
> ```

### 基本使用

```bash
# 最简用法：生成与输入同名的 PDF
md2pdf doc.md

# 指定输出路径
md2pdf doc.md -o out.pdf

# 自定义页边距与页面尺寸
md2pdf doc.md -m "25 20 25 20" --page-size A4

# 更换字体
md2pdf doc.md -f "SimHei"

# 调试：输出 HTML 而非 PDF
md2pdf doc.md --format html
```

### CLI 参数一览

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `<input>` | 输入 Markdown 文件（必填） | — |
| `-o, --output <path>` | 输出 PDF 路径 | 与输入同名 `.pdf` |
| `-s, --style <path>` | 自定义主题 CSS | 内置 `default` |
| `-f, --font <fonts>` | 字体族（逗号分隔） | `Times New Roman, SimSun` |
| `--page-size <size>` | A4 / Letter / A5 / A3 | `A4` |
| `-m, --margin <value>` | 页边距（mm） | `18` |
| `--format <format>` | pdf 或 html | `pdf` |

---

## 工作原理

```
Markdown ──▶ markdown-it 解析 ──▶ HTML + 排版 CSS ──▶ Chromium 打印 ──▶ PDF
              (锚点/脚注/任务列表)    (宋体/Times 栈)      (无页眉页脚)
              + highlight.js 高亮    + 本地图片 base64
```

- **代码高亮**：在构建期由 highlight.js 完成（支持 30+ 语言），PDF 渲染时无需网络
- **图片处理**：本地图片自动转为 base64 嵌入 HTML，网络图片直连引用
- **字体**：使用系统字体，不打包任何字体文件
- **渲染引擎**：基于 Playwright (Chromium)，确保精确的 CSS 打印排版

---

## 编程式 API

也可在 Node.js 代码中直接调用：

```js
import { convert } from 'md2pdf';

// 基本用法
const { pdf, title } = await convert({ inputPath: './doc.md' });
await fs.writeFile('doc.pdf', pdf);

// 传入选项
const result = await convert({
  inputPath: './doc.md',
  options: {
    pageSize: 'A4',
    margin: '25 20 25 20',
    font: 'SimHei, SimSun',
  },
});

// 仅获取 HTML 预览
const { html } = await convert({
  inputPath: './doc.md',
  options: { format: 'html' },
});
```

### 导出模块

所有模块统一从 `md2pdf` 入口导出：

| 入口 | 导出 |
| --- | --- |
| `md2pdf` | `convert` |
| `md2pdf` | `renderMarkdown`, `slugify` |
| `md2pdf` | `buildHtml`, `loadThemeCss` |
| `md2pdf` | `renderPdf`, `parseMargin` |
| `md2pdf` | `resolveOptions`, `DEFAULTS` |

---

## Web 服务

md2pdf 内置轻量级 Web 服务，提供在线编辑器和 API 接口：

```bash
# 启动 Web 服务（默认 http://localhost:3222）
npm run web

# 开发模式（文件变动自动重启）
npm run web:dev
```

### API 端点

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/render` | 提交 Markdown，返回 HTML |
| `POST` | `/api/pdf` | 提交 Markdown，返回 PDF 下载 |
| `GET` | `/api/theme` | 返回内置主题 CSS |
| `GET` | `/api/config` | 返回默认配置 |

**示例**——通过 API 生成 PDF：

```bash
curl -X POST http://localhost:3222/api/pdf \
  -H "Content-Type: application/json" \
  -d '{"markdown": "# Hello World\n\n你好，世界！"}' \
  -o hello.pdf
```

---

## 自定义主题

主题为 CSS 文件，可完全重写排版。内置 `default` 主题的关键点：

- **字体栈**：`"Times New Roman", SimSun, "Songti SC", "STSong", "Source Han Serif SC", "Noto Serif CJK SC", serif`
  （拉丁字符走 Times，中文字符自动回退宋体）
- **正文**：11pt、行高 1.75、两端对齐（`text-justify: inter-ideograph`）
- **表格**：跨页重复表头、代码块防分页断裂
- **字体覆盖**：可通过 CSS 变量 `--md2pdf-font` 或 CLI `-f` 覆盖

```bash
# 使用自定义主题
md2pdf doc.md -s path/to/your-theme.css
```

自定义时建议从 `src/themes/default.css` 复制一份再修改。

---

## 开发

```bash
# 运行测试（node:test，无额外依赖）
npm test

# 生成示例 PDF
npm run example

# 生成示例 HTML（便于调试排版）
npm run example:html

# 启动 Web 服务
npm run web
```

### 测试覆盖

- 中文锚点、语法高亮、安全转义
- 端到端 PDF 生成（含 PDF 文本提取校验）
- 一一对应渲染（无封面/目录/页脚）
- 字体嵌入（宋体 + Times New Roman）
- 页边距解析（1/2/4 参数）

### 项目结构

```
md2pdf/
├── bin/md2pdf.js         CLI 入口
├── src/
│   ├── cli.js            参数解析与编排
│   ├── config.js         默认值与合并
│   ├── markdown.js       markdown-it 解析 + 高亮
│   ├── template.js       HTML 模板 / 主题加载
│   ├── themes/default.css 内置主题
│   ├── pdf.js            Playwright 打印
│   ├── convert.js        转换管线
│   ├── util.js           通用工具函数
│   └── index.js          编程式 API 入口
├── web/
│   ├── server.js         Express Web 服务
│   └── public/           Web 前端静态资源
├── examples/
│   ├── sample.md         示例文档
│   └── sample.pdf        生成的示例 PDF
├── test/run.mjs          测试
├── package.json
└── README.md
```

---

## License

[MIT](LICENSE) © 2025 md2pdf contributors