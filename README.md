# Hanerson 的学术车辙

个人博客源码：Hexo 8 + [Stellar 2](https://xaoxuu.com/wiki/stellar/)，发布在 <https://blog.hanerson.com>。

## 目录结构

```
.
├─ source/
│  ├─ _posts/                 文章，按主题分目录
│  │  ├─ tech/                技术笔记：heterogeneous（异构计算）/ tooling（工具）/ algorithms（算法）
│  │  ├─ essays/              随笔：film（影评）/ notes（杂记）
│  │  └─ courses/             课程复习：bi / compiler / hla / os / se / history / guide
│  ├─ about/index.md          关于页
│  ├─ img/                    图片，按文章主题分目录，正文用 /img/... 绝对路径引用
│  └─ _data/
│     └─ legacy-redirects.yml 旧地址 -> 新地址映射（由重定向脚本读取）
├─ scripts/
│  ├─ legacy-redirects.js     为旧地址生成跳转页
│  └─ math-marker.js          按 Front Matter 给数学公式页加加载标记
├─ _config.yml                Hexo 配置（站点、permalink、生成器）
├─ _config.stellar.yml        Stellar 主题配置（导航、外观、侧栏）
└─ scaffolds/post.md          新文章模板
```

文章的目录只影响源码组织，不进入 URL：permalink 使用 `:year/:month/:day/:name/`，地址只由日期和文件名决定。

## 本地开发

```bash
npm install          # 安装依赖
npm run server       # 本地预览 http://localhost:4000
npm run build        # 生成静态站点到 public/
npm run clean        # 清理缓存与 public/
npx hexo stellar doctor   # 检查主题配置与内容
```

## 写文章

```bash
npx hexo new post "文章标题"   # 生成 source/_posts/文章标题.md
```

Front Matter 约定：

```yaml
---
title: 文章标题
date: 2026-10-30 12:00:00
categories: ["技术笔记", "异构计算"]   # 两级分类，见下表
tags: ["CUDA", "GPU"]                 # 2-4 个可复用标签
description: 用于列表卡片与 SEO 的一句话摘要（可选）
cover: /img/xxx/cover.png             # 卡片封面（可选）
render:
  math: mathjax                       # 含 $...$ 公式时开启
article:
  style: story                        # 叙事类文章用 story，其余默认 tech
---
```

- 摘要：不写 `description` 时，列表卡片自动截取正文前 140 字。
- 图片：放在 `source/img/<主题>/`，正文用 `/img/<主题>/<文件名>` 引用，不要写 `../` 相对路径。
- 公式：在 Front Matter 写 `render.math: mathjax`，正文用 `$...$` / `$$...$$`。`scripts/math-marker.js` 会只在启用了公式的页面插入 MathJax 标记，其余页面不加载 MathJax。

### 分类

| 一级 | 二级 | 说明 |
| --- | --- | --- |
| 技术笔记 | 异构计算 / 工具与工程 / 算法与数据结构 | 原创技术长文 |
| 随笔 | 影评 / 杂记 | 非技术写作 |
| 课程复习 | 商务智能 / 编译原理 / 高级算法 / 计算机操作系统 / 软件工程与计算II / 中国近现代史纲要 / 课程导航 | 课程笔记 |

### 标签

标签只用于跨分类的主题聚合，优先复用已有词（`GPU`、`CUDA`、`算法`、`复习提纲`……），不要新建与分类同名的标签，也不要给所有文章打同一个标签。

## 配置

- `_config.yml`：站点信息、permalink、首页分页、归档生成。
- `_config.stellar.yml`：只写需要覆盖的字段，其余使用主题默认值；字段说明见[主题配置参考](https://xaoxuu.com/wiki/stellar/reference/theme/)。
- 主题本身装在 `node_modules/hexo-theme-stellar`，不要直接改主题文件，定制写在 `_config.stellar.yml` 或 `source/_data/`。

## 旧链接

整理内容时目录与 URL 都变了。`source/_data/legacy-redirects.yml` 记录了全部旧地址到新地址的映射，`scripts/legacy-redirects.js` 在生成站点时为每条映射输出一个带 `meta refresh` 的跳转页，保证旧链接仍然可用。

再次调整 permalink 或移动文章时，请同步更新这个映射文件。
