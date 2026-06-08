---
title: TileLang Puzzle 第二课
date: 2026-06-08 10:30:58
tags: ["CANN"]

categories: ["TileLang Puzzle"]
---
## 编程模型

### Grid-Block-Thread

![](../img/TileLang/1.png)

<!-- more -->

![](../img/TileLang/2.png)

- 同一个warp当中的thread是共享部分资源的

![GPU存储子系统架构](../img/TileLang/3.png)

![内存层次模型](../img/TileLang/4.png)


## 体系结构相关的优化案例

![overall view](../img/TileLang/5.png)

### Occupancy-资源占用
![occupancy](../img/TileLang/occupancy.png)

#### case 1: active block

![occupancy](../img/TileLang/activeBlock.png)

> 追求相对较小的task分割大小，以获得更高的block利用率（当然最好是切成block的倍数）

> 简单的数学就可以证明，由于每一个计算单元的计算速度是固定的，所以只要空转时间越短，整体效率就会越高

![occupancy](../img/TileLang/active2.png)

#### case 2: active warp

![occupancy](../img/TileLang/activeWarp.png)

#### case 3: private memory

> 简单来说，就是使用的临时变量太多，导致最快的寄存器分配器无法满足，会出现溢写到private memory当中的情况，而private memory的访问周期往往是寄存器的上百倍，导致显著的性能损失

![occupancy](../img/TileLang/privateMemory.png)
![occupancy](../img/TileLang/privateMemory2.png)

##### 一个优化示例

![occupancy](../img/TileLang/memory3.png)


### Coalescing-合并
> coalescing 是指将连续的访问合并成一个访问，从而提高访存效率

#### case 4: Global Memory Access Coalescing

![occupancy](../img/TileLang/globalCoalescing.png)


#### case 5: Partial Write降低HBM写带宽
![occupancy](../img/TileLang/coalescing2.png)

### Bank Conflict - bank冲突
#### case 6: Shared Memory Bank Conflicts

![occupancy](../img/TileLang/confilt.png)

##### 一个优化示例

![occupancy](../img/TileLang/Cexample.png)

### Latency Hiding - 延迟隐藏
#### Latency Hiding

![occupancy](../img/TileLang/latecyHiding.png)

