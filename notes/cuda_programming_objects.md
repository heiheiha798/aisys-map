# CUDA Programming Objects

这份笔记专门澄清 CUDA kernel 编程里最基础、也最容易混淆的对象。

重点不是教材式定义，而是先把这些对象之间的关系讲清楚：

- thread
- warp
- block
- grid
- kernel
- SM

---

## 先看总图

可以先用一句话记：

- `kernel` 是你发给 GPU 执行的一段函数
- `grid` 是这次启动出来的所有线程块
- `block` 是一组线程
- `warp` 是硬件实际成组执行线程的重要单位
- `thread` 是你写代码时看到的最小逻辑线程
- `SM` 是 GPU 上实际执行这些 block/warp 的计算单元

一个粗糙但好用的关系图是：

`kernel -> grid -> block -> thread`

以及：

`block` 在硬件上会被拆成多个 `warp`，再放到 `SM` 上执行

---

## 1. kernel 是什么

`kernel` 就是一段在 GPU 上执行的函数。

当你写 CUDA 时，常见形式是：

```cpp
my_kernel<<<grid_dim, block_dim>>>(...);
```

这里的意思不是“调用一次普通函数”，而是：

- 启动一个 kernel
- 让 GPU 创建很多线程
- 这些线程一起执行同一段代码

所以 kernel 可以理解成：

- 同一份程序
- 被很多线程并行执行

### 关键直觉

- kernel 不是一个线程
- kernel 也不是一个 block
- kernel 是“一次并行启动的整体”

---

## 2. thread 是什么

`thread` 是 CUDA 编程里最小的逻辑执行单位。

你写 kernel 时，通常会让每个 thread 处理一小部分数据，例如：

- 一个元素
- 一行中的一个位置
- 一个 tile 中的一个片段

每个 thread 都有自己的：

- thread index
- 局部变量
- 寄存器状态

### 关键直觉

- thread 是你写并行逻辑时最细粒度的对象
- 但硬件不是按单个 thread 随意执行的，而是按 warp 成组执行

---

## 3. warp 是什么

`warp` 是 NVIDIA GPU 上硬件执行线程的重要单位。

通常：

- `1 warp = 32 个线程`

这意味着，虽然你写代码时面对的是一个个 thread，  
但 GPU 往往会把连续的 32 个 thread 当成一组来调度和执行。

### 为什么 warp 重要

很多 GPU 性能现象都和 warp 有关：

- 分支发散
- 内存访问是否 coalesced
- 调度和 latency hiding

### 分支发散是什么

如果一个 warp 里的线程走了不同分支，例如：

- 一部分线程走 `if`
- 一部分线程走 `else`

GPU 往往需要分开执行这些路径，效率就会下降。

所以 warp 也是理解“为什么分支会慢”的关键对象。

### 关键直觉

- thread 是逻辑最小单位
- warp 是硬件实际批量执行 thread 的小队

---

## 4. block 是什么

`block` 是一组 thread 的组织单位。

你可以在 launch kernel 时指定每个 block 有多少个线程，例如：

```cpp
my_kernel<<<num_blocks, 256>>>(...);
```

这里的 `256` 就表示：

- 每个 block 有 256 个线程

### block 的重要性

block 很重要，因为：

- 一个 block 内的线程可以共享 `shared memory`
- 一个 block 内的线程可以做 block-level synchronization
- block 是调度到 SM 上执行的基本单位之一

### block 和 warp 的关系

一个 block 会被拆成多个 warp。

例如：

- `blockDim = 128`
- 那么这个 block 会被拆成 `4` 个 warp

因为：

- `128 / 32 = 4`

### 关键直觉

- block 是程序员组织线程合作的重要单位
- warp 是硬件执行这些线程的重要单位

---

## 5. grid 是什么

`grid` 是一次 kernel launch 中的所有 block 的集合。

例如：

```cpp
my_kernel<<<80, 256>>>(...);
```

可以理解成：

- 这次 launch 有 80 个 block
- 每个 block 有 256 个线程

所以整个 grid 中一共启动了：

- `80 * 256` 个线程

### 关键直觉

- grid 是这次 kernel 启动的全体线程组织
- 一个 grid 里有很多 block
- block 之间默认不能像 block 内线程那样随意协作

---

## 6. SM 是什么

`SM` 是 Streaming Multiprocessor，可以理解成 GPU 上重复出现的计算核心簇。

SM 上会执行：

- block
- warp
- thread 对应的指令流

每个 SM 一般有：

- 计算单元
- 寄存器文件
- shared memory
- warp scheduler

### block 和 SM 的关系

一个 block 会被分配到某个 SM 上执行。  
同一个 SM 上通常可以同时驻留多个 block，前提是资源够用。

这些资源包括：

- 寄存器
- shared memory
- 最大线程数
- 最大 block 数

### 关键直觉

- SM 是“真正干活”的地方
- block 会被放到 SM 上
- warp 会在 SM 上被调度执行

---

## 7. 为什么 block 不能太大也不能太小

如果 block 太小：

- 并行度可能不够
- shared memory / warp 利用可能不充分

如果 block 太大：

- 每个 block 可能吃掉太多寄存器
- 吃掉太多 shared memory
- 反而让一个 SM 同时放不下太多 block

这会影响 occupancy 和 latency hiding。

所以 block size 是一个典型的系统权衡点，不是越大越好。

---

## 8. synchronization 的边界

### block 内同步

一个 block 内的线程可以通过类似 `__syncthreads()` 的方式同步。

这是因为它们在同一个 block 内，有明确的协作边界。

### block 间同步

不同 block 默认不能在同一个 kernel 中随意做全局同步。

这点非常重要，因为很多初学者会误以为：

- 只要都在同一个 grid 里，就能方便同步

其实不是。

### 关键直觉

- block 内容易协作
- block 间默认隔离更强

---

## 9. 编程对象之间的常见混淆

### thread 和 warp 混淆

错误直觉：

- “GPU 是一个个 thread 单独执行的”

更准确的理解：

- 程序员按 thread 写逻辑
- 硬件经常按 warp 批量执行

### warp 和 block 混淆

错误直觉：

- “warp 就是一小个 block”

更准确的理解：

- warp 是硬件执行单位
- block 是程序组织和协作单位

### block 和 grid 混淆

错误直觉：

- “grid 就是更大的 block”

更准确的理解：

- grid 是全部 block 的集合
- 不同 block 之间默认没有 block 内那种紧密协作能力

### kernel 和 thread 混淆

错误直觉：

- “kernel 就是一段在一个线程上跑的函数”

更准确的理解：

- kernel 是同一段函数，被大量线程并行执行

---

## 10. 一个最小例子怎么理解

如果你写：

```cpp
my_kernel<<<2, 64>>>();
```

可以这样理解：

- 启动 1 个 kernel
- 这个 kernel 有 1 个 grid
- 这个 grid 里有 2 个 block
- 每个 block 里有 64 个 thread
- 每个 block 会被拆成 2 个 warp
- 总共有 128 个 thread

如果 GPU 上有足够资源，这些 block 会被分配到一个或多个 SM 上执行。

---

## 11. 先记住的最小结论

如果你现在只想先记最重要的内容，可以记这几句：

1. `kernel` 是一次 GPU 并行启动
2. `grid` 是这次启动的所有 block
3. `block` 是一组可以共享 shared memory 的线程
4. `thread` 是最小逻辑线程
5. `warp` 是硬件常用的 32 线程执行单位
6. `SM` 是实际执行 block/warp 的地方

---

## 12. 下一步自然会遇到的问题

- thread index、block index 应该怎么用？
- 1D / 2D / 3D grid 和 block 是什么？
- occupancy 和 block size 到底怎么关联？
- register pressure 为什么会限制并发？
- shared memory 大小为什么会影响一个 SM 上能同时放多少 block？
- warp divergence 到底会怎么拖慢程序？
