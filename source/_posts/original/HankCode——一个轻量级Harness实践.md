---
title: HankCode——一个轻量级Harness实践
date: 2026-08-03 23:05:57
tags: ["学术"]

categories: ["原创"]
---

# HankCode

一个轻量级的 Coding Agent Harness，聚焦治理护栏（Guardrails），让 LLM 的编码行为受可验证的确定性代码约束。

[Github](https://github.com/Hanerson/HankCode)

[GitLab](https://git.nju.edu.cn/HANS/hankcode)

---

## 项目简介

HankCode 是一个**从零构建**的 Agent Harness 框架，核心等式为 **Agent = LLM + Harness**。它实现了 6 个核心维度：

| 维度 | 实现 | 说明 |
|------|------|------|
| 决策封装 | Agent 主循环 | 组织上下文 → 调用 LLM → 解析动作 → 分发执行 → 回灌结果 → 停机 |
| 动作/工具 | 6 个工具 | read_file, write_file, list_files, run_shell, web_search, run_tests |
| 治理护栏 | 5 道护栏流水线 | 路径沙箱 → 危险命令拦截 → 敏感脱敏 → 写入范围 → HITL |
| 反馈闭环 | 语法检查 + 测试运行 | 客观信号驱动自我修正 |
| 记忆 | 会话级记忆 | 20 条历史记录，支持截取 |
| 配置 | .env + CLI | 声明式规则约束 Agent 行为 |

**重点深入维度：治理护栏**。所有护栏的判定逻辑均为确定性代码（正则匹配、路径比较、glob 匹配），不依赖 LLM 的智能。

---

<!-- more -->

## 快速开始

### 安装

```bash
# 克隆仓库
git clone <repo-url>
cd hankcode

# 安装依赖（Python 3.11+）
pip install -r requirements.txt

# 安装包（开发模式）
pip install -e .
```

### 配置 API Key

```bash
# 安全录入 Key（隐藏输入，不进入 shell history）
hankcode keys set

# 查看 Key 状态（只显示预览，不回显明文）
hankcode keys show

# 更新 Key（重新执行 set 覆盖旧值）
hankcode keys set

# 清除 Key
hankcode keys clear
```

### 运行

```bash
# 启动交互模式
hankcode

# 选择人格风格
hankcode --persona master

# 指定工作目录
hankcode --workspace /path/to/project

# 单次执行
hankcode run "创建一个 Python 计算器"

# 查看状态
hankcode status
```

### 人格风格

| 人格 | 效果 | 命令 |
|------|------|------|
| `default` | 标准语气，默认 | `hankcode` |
| `friend` | 友好轻松 | `hankcode --persona friend` |
| `minimal` | 极简输出 | `hankcode --persona minimal` |
| `master` | 自信专业 | `hankcode --persona master` |

---

## 护栏系统

5 道护栏按以下顺序执行，采用短接（short-circuit）模式——任一护栏拦截即终止：

```
┌─ 用户输入 ─┐
      │
      ▼
┌─ Agent 主循环 ─────────────────────┐
│ ① 组织上下文 → ② 调用 LLM → ③ 解析 JSON │
└──────────────┬─────────────────────┘
               │
               ▼
┌─ 护栏流水线 ──────────────────────────┐
│ ④ 路径沙箱 (PathSandbox)              │
│    检查文件路径是否在 workspace 内       │
├───────────────────────────────────────┤
│ ⑤ 危险命令拦截 (CommandBlocker)        │
│    匹配 rm -rf /、format、dd 等黑名单   │
├───────────────────────────────────────┤
│ ⑥ 敏感信息脱敏 (SensitiveDataRedactor) │
│    检测并替换 API Key、Token、私钥       │
├───────────────────────────────────────┤
│ ⑦ 写入范围检查 (WriteScope)            │
│    glob 模式限定写入路径                │
├───────────────────────────────────────┤
│ ⑧ HITL 人工审批 (HumanInTheLoop)      │
│    高风险操作暂停等待确认               │
└────────────────┬──────────────────────┘
                 │ 放行
                 ▼
┌─ 工具执行 → 反馈校验 ─┐
│ ⑨ 语法检查 / 测试运行  │
└───────────────────────┘
```

---

## 目录结构

```
hankcode/
├── hankcode/                 # 核心包
│   ├── __init__.py           # 版本信息
│   ├── cli.py                # CLI 入口 + 命令分发
│   ├── agent.py              # Agent 主循环 ★
│   ├── llm.py                # LLM 抽象层（MockLLM + OpenAILLM）
│   ├── config.py             # 配置加载（.env）
│   ├── models.py             # 核心数据模型
│   ├── persona.py            # 人格系统（纯文本风格）
│   ├── tools/                # 工具系统（6 个工具）
│   │   ├── base.py           # Tool 基类 + ToolRegistry
│   │   ├── file_tools.py     # read_file, write_file, list_files
│   │   ├── shell.py          # run_shell
│   │   ├── web.py            # web_search
│   │   └── tests.py          # run_tests
│   ├── gates/                # 护栏系统（5 道护栏）★
│   │   ├── base.py           # Gate 基类 + GatePipeline
│   │   ├── path_sandbox.py   # 路径沙箱
│   │   ├── command_blocker.py# 危险命令拦截
│   │   ├── redactor.py       # 敏感信息脱敏
│   │   ├── write_scope.py    # 写入范围检查
│   │   └── hitl.py           # HITL 人工审批
│   ├── checks/               # 反馈闭环
│   │   ├── syntax.py         # 语法检查
│   │   └── test_runner.py    # 测试运行器
│   └── memory/               # 记忆系统
│       └── session.py        # 会话级记忆
├── tests/                    # 单元测试（69 个，全部使用 MockLLM）
│   ├── test_agent.py         # Agent 循环测试
│   ├── test_gates.py         # 护栏测试 ★
│   ├── test_gates_extra.py   # 额外护栏测试
│   ├── test_tools.py         # 工具测试
│   ├── test_checks.py        # 校验器测试
│   ├── test_llm.py           # LLM 抽象层测试
│   ├── test_memory.py        # 记忆系统测试
│   ├── test_persona.py       # 人格系统测试
│   ├── test_config.py        # 配置测试
│   ├── test_cli.py           # CLI 测试
│   └── test_models.py        # 数据模型测试
├── demo/                     # 机制演示
│   ├── run_demo.py           # 一键演示脚本
│   └── test_demo.py          # 演示测试（3 个场景）
├── docs/                     # 设计文档
│   ├── SPEC.md               # 设计规约
│   ├── plan.md               # 实现计划
│   ├── SPEC_PROCESS.md       # 规约过程文档
│   ├── AGENT_LOG.md          # Agent 交互日志
│   └── REFLECTION.md         # 反思报告
├── workspace/                # 默认工作目录（不提交 Git）
├── .env.example              # 环境变量模板
├── .gitignore
├── .gitlab-ci.yml            # CI 配置
├── Dockerfile                # 容器分发
├── setup.py                  # 包安装配置
├── requirements.txt          # Python 依赖
├── SPEC.md                   # 设计规约（根目录副本）
├── PLAN.md                   # 实现计划（根目录副本）
├── README.md                 # 本文件
└── LICENSE
```

---

## 运行测试

```bash
# 一键运行所有测试（无需网络，无需 API Key）
python -m pytest tests/ demo/ -v

# 仅运行护栏测试
python -m pytest tests/test_gates.py tests/test_gates_extra.py -v

# 运行机制演示
python -m pytest demo/ -v

# 运行演示脚本
python demo/run_demo.py
```

所有 69 个测试使用 MockLLM，不依赖网络和真实 API Key，测试套件在 1-2 秒内完成。

---

## 分发

### Docker 容器

```bash
# 构建镜像
docker build -t hankcode .

# 运行交互模式（挂载工作目录）
docker run -it --rm \
  -v $(pwd)/workspace:/app/workspace \
  -e HANKCODE_API_KEY=your-key-here \
  hankcode --persona master

# 或者通过挂载 .env 文件配置 Key（更安全，避免 Key 进入 shell history）
docker run -it --rm \
  -v $(pwd)/workspace:/app/workspace \
  -v $(pwd)/.env:/app/.env \
  hankcode

# 单次执行
docker run --rm \
  -v $(pwd)/workspace:/app/workspace \
  -e HANKCODE_API_KEY=your-key-here \
  hankcode run "创建一个 Python 计算器"
```

**注意**：Docker 镜像基于 `python:3.11-slim`，适用于 Linux/amd64 和 Linux/arm64 平台。Windows 用户需使用 Docker Desktop，并注意路径格式（使用 `%cd%` 替代 `$(pwd)`）。

### PyPI 安装（可选）

```bash
pip install hankcode
```

---

## 安全边界

### API Key 安全

| 风险 | 对策 |
|------|------|
| Key 硬编码进源码 | 通过 `.env` 加载，不硬编码、不提交 Git |
| Key 在终端日志泄露 | 护栏敏感信息脱敏自动检测并替换 |
| Key 在 shell history 泄露 | 使用 `hankcode keys set`（`getpass` 隐藏输入） |
| Key 在进程环境可见 | 在 README 中说明风险（`/proc/pid/environ` 可读） |

**重要安全说明**：
- `.env` 文件是**明文存储**的，如果系统被入侵，Key 可能被读取
- 进程环境变量可通过 `/proc/pid/environ` 读取
- 生产环境建议使用操作系统钥匙串（macOS Keychain / Windows Credential Manager / Linux Secret Service）或密钥管理服务
- **提交前自查**：确保 `.env` 不在 Git 跟踪中，`git status` 确认无 Key 泄露

### 运行安全

| 机制 | 保护范围 |
|------|---------|
| 路径沙箱 | 防止 Agent 读写 workspace 外的文件 |
| 危险命令拦截 | 防止 `rm -rf /`、`format`、`dd` 等破坏性命令 |
| 敏感信息脱敏 | 检测并替换输出中的 API Key、Token、私钥 |
| 写入范围 | 限定写入路径的 glob 模式 |
| HITL | 高风险操作（如删除文件）暂停等待人工确认 |

---

## 已知限制

1. **Windows 兼容性**：部分路径处理逻辑在 Windows 上可能行为不同（`Path.resolve()` 的符号链接解析行为），建议在 Linux/macOS 上运行
2. **TestRunner 未完全接入**：TestRunner 类已实现但未被 agent 主循环自动调用，需 agent 主动使用 `run_tests` 工具
3. **凭据存储**：当前使用 `.env` 明文存储，未集成操作系统钥匙串
4. **单会话**：记忆系统仅支持会话级，不支持跨会话持久化
5. **无速率限制**：未实现 API 调用速率限制和 token 预算控制
6. **Docker 交互模式**：Docker 中运行交互模式需要 `-it` 参数，且 volume 挂载路径在 Windows 上需使用 `%cd%`

---

## CI/CD

项目使用 GitLab CI，配置在 `.gitlab-ci.yml` 中：

- **unit-test job**：每次 push 自动运行所有测试（必须通过）
- **docker-build job**：main 分支构建 Docker 镜像

```bash
# 本地运行 CI 检查
python -m pytest tests/ demo/ -v
```

---

## 技术栈

| 维度 | 选择 | 理由 |
|------|------|------|
| 语言 | Python 3.11+ | 课程熟悉度高；标准库丰富 |
| LLM 供应商 | OpenAI 兼容 API | 最广泛兼容；支持 DeepSeek、SiliconFlow、Volc 等 |
| LLM SDK | `openai` Python 包 | 官方维护，支持 OpenAI 兼容 API |
| 配置 | `.env` | 零配置，广泛使用 |
| 测试 | `pytest` | 课程标准，功能强大 |
| 分发 | Docker + PyPI | 平台无关，一键运行 |
| CLI 框架 | `argparse`（标准库） | 零依赖，够用 |

---

## 机制演示

HankCode 提供 3 个机制演示，验证核心能力：

1. **护栏拦截**：`CommandBlocker` 拦截危险命令 `rm -rf /`
2. **反馈闭环**：`SyntaxChecker` 检测语法错误并回灌
3. **路径沙箱**：`PathSandbox` 阻止路径遍历逃逸

```bash
# 运行演示
python -m pytest demo/ -v
```

所有演示使用 MockLLM，无需真实 API。

---

## 许可证

MIT

---

## 学术规范

- 本项目使用 Superpowers 框架（[https://github.com/obra/superpowers](https://github.com/obra/superpowers)）进行开发
- 第三方依赖的许可证见各包文档
- 反思报告（`REFLECTION.md`）由学生本人撰写，AI 辅助润色文字表达