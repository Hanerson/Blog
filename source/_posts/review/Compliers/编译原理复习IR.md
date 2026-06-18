---
title: 编译原理复习_Intermediate Representation
date: 2026-06-14 19:08:29
tags: ["复习"]

categories: ["编译原理"]
---

本文部分内容由AI辅助整理

## 中间表示（IR）

### 什么是IR
- **定义**：编译器前端与后端之间使用的、与具体源语言和目标机器都相对独立的程序表示形式。
- **作用**：承载程序语义、驱动优化，并作为目标语言生成的基础。

### IR的层次分类
| 类型 | 抽象程度 | 代表 | 优点 | 缺点 |
|------|----------|------|------|------|
| 高层IR | 高 | AST、语法树 | 便于高层语义优化（内联、异常处理） | 距机器远，难以做寄存器分配 |
| 中层IR | 中 | 三地址码（TAC）、四元式、SSA | 权衡了可分析性与机器接近度 | 常见于编译器核心 |
| 低层IR | 低 | 显式寄存器/栈操作 | 易映射到目标指令 | 优化空间受限 |

<!-- more -->

### 常见IR范式
- **树形IR**：AST → 适合早期高层变换
- **三地址码**：`x = y op z` → 最常用
- **SSA形态IR**：在TAC基础上引入Φ函数 → 优化友好
- **堆栈式IR**：如JVM字节码 → 简单紧凑
- **LLVM IR**：工业级IR

---

## 三地址码（TAC）

### 基本格式
```
result = arg1 op arg2
```
- 每条指令最多含一个运算符、三个操作数
- “地址”可以是：常量、变量、临时变量、标签

### 常见指令类型
| 类型 | 格式示例 |
|------|----------|
| 算术/逻辑运算 | `x = y + z` |
| 赋值/拷贝 | `x = y` 或 `x = -y` |
| 无条件跳转 | `goto L` |
| 条件跳转 | `if x relop y goto L` |
| 函数调用 | `param x` `call f, n` |
| 数组访问 | `x = y[i]` `x[i] = y` |
| 标签 | `L:` |

### 实现方式（四元式 vs 三元式）

#### 四元式 `(op, arg1, arg2, result)`
- 优点：容易移动/优化
- 例子：`t1 = b * -c` → `(*, b, t_minus_c, t1)`

#### 三元式 `(op, arg1, arg2)`
- 用三元式的位置引用结果
- 问题：优化移动时需修改引用

#### 间接三元式
- 包含指向三元式的指针列表
- 操作时不修改三元式本身

---

## 静态单赋值（SSA）

### 核心规则
- 每个变量只赋值一次 → 添加下标区分版本
- 控制流合并点使用 **Φ函数** 合并不同路径的定值

### 例子

**原始代码：**
```c
if (flag) x = -1; else x = 1;
y = x * a;
```

**SSA形式：**
```c
if (flag) x1 = -1; else x2 = 1;
x3 = Φ(x1, x2);
y = x3 * a;
```

### Φ函数的本质
- 数学意义：多路选择器（switch）
- 语义：在控制流合流点，根据控制路径决定选哪个值
- 不是CPU指令 → 生成汇编时必须消除（φ-elimination）

---

## 控制流语句翻译（基于继承属性）

### 翻译核心机制
- **基本方法**：递归下降 + 临时变量生成 + 标签跳转
- **关键属性**：
    - `B.true` / `B.false`：布尔表达式真假出口标签
    - `S.next`：语句执行完后跳转到的标签
    - `label(X)`：设置标签
    - `newlabel()`：生成新标签

### 关键语法规则（SDT）

#### if 语句
```
S → if (B) S1
B.true = newlabel()
B.false = S1.next = S.next
S.code = B.code || label(B.true) || S1.code
```

#### if-else 语句
```
S → if (B) S1 else S2
B.true = newlabel()
B.false = newlabel()
S1.next = S2.next = S.next
S.code = B.code || label(B.true) || S1.code
        || gen('goto' S.next) || label(B.false) || S2.code
```

#### while 循环
```
S → while (B) S1
begin = newlabel()
B.true = newlabel()
B.false = S.next
S1.next = begin
S.code = label(begin) || B.code
        || label(B.true) || S1.code || gen('goto' begin)
```

#### 顺序语句
```
S → S1 S2
S1.next = newlabel()
S2.next = S.next
S.code = S1.code || label(S1.next) || S2.code
```

### 4.3 布尔表达式翻译（短路求值）

| 产生式 | 语义规则 |
|--------|----------|
| `B → B1 || B2` | `B1.true = B.true`; `B1.false = newlabel()`; 当B1为假时进入B2 |
| `B → B1 && B2` | `B1.true = newlabel()`; `B1.false = B.false`; 当B1为真时进入B2 |
| `B → !B1` | `B1.true = B.false`; `B1.false = B.true` |
| `B → E1 rel E2` | `gen('if' E1 rel E2 'goto' B.true)`; `gen('goto' B.false)` |
| `B → true` | `gen('goto' B.true)` |
| `B → false` | `gen('goto' B.false)` |

### 穿越（Fall-through）优化
- **思想**：当条件为真时，不生成跳转，而是顺序进入下一句（fall-through）
- **效果**：减少冗余goto指令
- **实现**：引入特殊标号 `fall`，表示不生成跳转指令

#### 示例：改进后的 `B → E1 rel E2` 规则
```
test = E1.addr rel.op E2.addr
s = if B.true ≠ fall and B.false ≠ fall then
        gen('if' test 'goto' B.true) || gen('goto' B.false)
    else if B.true ≠ fall then gen('if' test 'goto' B.true)
    else if B.false ≠ fall then gen('ifFalse' test 'goto' B.false)
    else ''
```

---

## 回填技术（Backpatching）

### 为什么需要回填？
- 控制流语句（if/while）的跳转目标在生成时尚未确定
- 无法在一次扫描中完成所有标签绑定

### 核心思想
1. 先生成**不完整的跳转指令**（占位 `goto _`）
2. 记录这些指令的位置到列表（`truelist` / `falselist`）
3. 当目标位置确定时，用 `backpatch()` 回填

### 关键辅助函数
- `makelist(i)`：创建包含指令位置 i 的列表
- `merge(p1, p2)`：合并两个列表
- `backpatch(p, L)`：将列表 p 中所有指令的目标设为 L
- `M.instr = nextinstr`：记录当前位置（用于回填）

### 回填版布尔表达式规则

| 产生式 | 语义规则 |
|--------|----------|
| `B → B1 || M B2` | `backpatch(B1.falselist, M.instr)`; `B.truelist = merge(B1.truelist, B2.truelist)`; `B.falselist = B2.falselist` |
| `B → B1 && M B2` | `backpatch(B1.truelist, M.instr)`; `B.truelist = B2.truelist`; `B.falselist = merge(B1.falselist, B2.falselist)` |
| `B → !B1` | `B.truelist = B1.falselist`; `B.falselist = B1.truelist` |
| `B → E1 rel E2` | `B.truelist = makelist(nextinstr)`; `B.falselist = makelist(nextinstr+1)`; `gen('if E1 rel E2 goto _')`; `gen('goto _')` |
| `B → true` | `B.truelist = makelist(nextinstr)`; `gen('goto _')` |
| `B → false` | `B.falselist = makelist(nextinstr)`; `gen('goto _')` |
| `M → ε` | `M.instr = nextinstr` |

### 回填版控制流语句规则

| 产生式 | 语义规则 |
|--------|----------|
| `S → if (B) M S1` | `backpatch(B.truelist, M.instr)`; `S.nextlist = merge(B.falselist, S1.nextlist)` |
| `S → if (B) M1 S1 N else M2 S2` | `backpatch(B.truelist, M1.instr)`; `backpatch(B.falselist, M2.instr)`; `S.nextlist = merge(merge(S1.nextlist, N.nextlist), S2.nextlist)` |
| `S → while M1 (B) M2 S1` | `backpatch(S1.nextlist, M1.instr)`; `backpatch(B.truelist, M2.instr)`; `S.nextlist = B.falselist`; `gen('goto M1.instr')` |
| `N → ε` | `N.nextlist = makelist(nextinstr)`; `gen('goto _')` |

---

## for 循环翻译（重点）

### 结构等价
```
for (S1; !B; S2) S3
```
等价于：
```
S1;
while (!B) {
    S3;
    S2;
}
```

### 翻译规则（非回填版）
```
begin = newlabel()
B.false = newlabel()
B.true = S.next
S1.next = begin
S2.next = begin
S3.next = newlabel()

S.code = S1.code
      || label(begin)
      || B.code
      || label(B.false)
      || S3.code
      || label(S3.next)
      || S2.code
      || gen('goto' begin)
```

### 逻辑解释
- `B.false` 是 `!B` 为真时（即 `B` 为假，满足循环条件）跳转去 `S3`
- `B.true` 是 `!B` 为假时（即 `B` 为真，退出循环）跳转到 `S.next`
- `S3.next` 指向 `S2` 入口，`S2.next` 指向 `begin`

---

## 完整示例：for 循环翻译

**源代码：**
```c
for (i = 0; i < n && sum < 1000; i = i + 1) {
    sum = sum + i;
}
```

**最终中间代码：**
```
i = 0

L1: if i < n goto L2
    goto L5
L2: if sum < 1000 goto L3
    goto L5
L3: sum = sum + i
    goto L4
L4: i = i + 1
    goto L1
L5:
```

**语法树属性标注：**
- `B.true = L3`（循环体入口）
- `B.false = L5`（循环出口）
- `S.next = L5`
- `S3.next = L4`
- `S2.next = L1`

---

## 常见题型总结

| 题型 | 要求 |
|------|------|
| 补充SDT规则 | 根据已有模式补充新控制结构的语义规则 |
| 语法树+属性标注 | 画出AST，在每个节点标注 `true`/`false`/`next` |
| 生成中间代码 | 按规则翻译成TAC，注意标号与跳转 |
| 回填过程分析 | 给出指令列表，说明何时用 `backpatch` |

