# CUDA Kernel Advanced

这份笔记接在 `cuda_programming_objects.md` 后面。  
如果前一份笔记解决的是：

- `thread`
- `warp`
- `block`
- `grid`
- `SM`

这些对象“是什么”，

那么这一份主要解决的是：

- 这些对象在写 kernel 时到底怎么用
- 它们为什么会影响性能
- 初学者最容易在哪些地方踩坑

---

## 1. thread index、block index 应该怎么用

写 CUDA kernel 时，一个最核心的问题是：

`每个 thread 到底负责哪一部分数据？`

这就是 index 存在的原因。

### 最常见的几个对象

- `threadIdx`
- `blockIdx`
- `blockDim`
- `gridDim`

可以先这样理解：

- `threadIdx`：当前线程在自己的 block 里的编号
- `blockIdx`：当前 block 在整个 grid 里的编号
- `blockDim`：每个 block 有多少线程
- `gridDim`：整个 grid 有多少个 block

### 最经典的 1D 线性索引

如果你处理的是一个一维数组，最常见的写法是：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

这句的意思是：

- `blockIdx.x * blockDim.x` 先跳过前面所有 block 的线程数
- `threadIdx.x` 再定位到当前 block 内的具体线程

所以这个 `idx` 就是：

- 当前线程在线性数据空间中的全局编号

### 为什么这很重要

因为 CUDA kernel 最常见的基本模式就是：

- 算出自己负责的数据下标
- 只处理自己那一小块

例如：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < n) {
    y[idx] = x[idx] * 2.0f;
}
```

这里每个 thread 负责一个元素。

### 为什么要写 `if (idx < n)`

因为 launch 的线程数常常会大于真实数据长度。

例如：

- 数据长度是 `1000`
- block size 是 `256`
- 那你可能会开 `4` 个 block
- 总线程数是 `1024`

这时最后 24 个线程是“多出来的”，必须防止越界。

### 一个更重要的理解

索引不只是“找到位置”，它还决定：

- memory access pattern
- 数据切分方式
- 一个 warp 里的线程是否访问连续内存

所以 index 设计不仅关系到对不对，也关系到快不快。

---

## 2. 1D / 2D / 3D grid 和 block 是什么

CUDA 里的 block 和 grid 不一定只能是一维。

你可以把它们组织成：

- 1D
- 2D
- 3D

### 什么时候用 1D

如果数据天然是一维的，例如：

- 向量
- 展平后的数组
- token 序列

那常常直接用 1D 就够了。

### 什么时候用 2D

如果数据天然有二维结构，例如：

- 矩阵
- 图像
- attention score matrix

那 2D launch 往往更直观。

例如：

```cpp
dim3 block(16, 16);
dim3 grid((W + 15) / 16, (H + 15) / 16);
```

这时：

- `threadIdx.x` 可以对应列
- `threadIdx.y` 可以对应行

### 什么时候用 3D

3D 比较少见，但在这些场景会用到：

- 三维张量
- batched 结构
- 额外多一个维度很自然的情况

### 关键直觉

1D/2D/3D 的本质不是“更高级”，而是：

- 让数据映射更自然
- 让代码更容易表达
- 有时也更容易形成好的访问模式

### 一个重要提醒

“维度组织方式”不只是代码风格。  
它常常会影响：

- 一个 warp 中线程的访问是否连续
- shared memory 的 tile 组织方式
- branch divergence 的形状

---

## 3. occupancy 和 block size 到底怎么关联

`occupancy` 可以粗糙理解成：

`一个 SM 上活跃线程/warp 的充实程度`

它在直觉上回答的是：

- 这个 SM 有没有足够多的活跃工作可做

### 为什么 occupancy 有意义

GPU 经常不是靠单个线程快，而是靠：

- 一堆 warp 同时在等、在切换、在执行

当某些 warp 因为访存而停住时，SM 可以切去执行别的 warp，这就是 latency hiding。

如果活跃 warp 太少，隐藏 latency 的能力就差。

### block size 为什么影响 occupancy

一个 block 会占用 SM 上的资源，例如：

- 寄存器
- shared memory
- warp 槽位
- block 槽位

如果 block size 太大：

- 单个 block 就可能吃掉太多资源
- 一个 SM 上能同时驻留的 block 数就少
- 活跃 warp 数也可能下降

如果 block size 太小：

- 每个 block 提供的 warp 太少
- 调度和资源利用可能又不充分

所以 block size 会直接影响一个 SM 能塞下多少工作。

### occupancy 高就一定快吗

不一定。

这是非常重要的一点。

高 occupancy 只是说明：

- 你“有机会”更好地隐藏 latency

但真正性能还要看：

- memory access pattern
- 是否发生 branch divergence
- shared memory 是否冲突
- 寄存器使用是否合理
- kernel 本身是不是 compute-bound

所以 occupancy 是一个重要指标，但不是唯一指标。

---

## 4. register pressure 为什么会限制并发

前面说过，寄存器是很快但很少的资源。

### 什么是 register pressure

如果一个 kernel 里：

- 每个 thread 需要很多寄存器

那么一个 block 总共需要的寄存器数就会很大。

例如：

- 每个 thread 需要 80 个寄存器
- 一个 block 有 256 个 thread

那这个 block 消耗的寄存器总量就会很高。

### 为什么这会限制并发

因为一个 SM 上可用的寄存器总量是有限的。

如果一个 block 太“吃寄存器”，

那么一个 SM 上就放不下太多 block 或太多 warp。

这会导致：

- 活跃 warp 变少
- occupancy 下降
- latency hiding 变差

### 更糟的情况：spill 到 local memory

如果寄存器真的不够，编译器可能会把一些本该放在寄存器里的东西“溢出”到 `local memory`。

而 `local memory` 往往会走更慢的内存路径。

这意味着：

- 访问变慢
- memory traffic 增加
- 性能可能明显下降

### 关键直觉

寄存器不是越用越好。  
寄存器使用多，说明 thread 局部数据放得更近；  
但如果太多，就会反过来压缩并发。

这就是典型的系统权衡。

---

## 5. shared memory 大小为什么会影响一个 SM 上能同时放多少 block

shared memory 是 SM 上的一块有限资源。

### 发生了什么

如果一个 kernel 的每个 block 都声明了很多 shared memory，

例如：

- 一个 block 需要 48 KB shared memory

而一个 SM 的 shared memory 容量有限，

那么这个 SM 上就可能只能同时驻留很少几个 block。

### 结果是什么

这会带来两个后果：

1. 一个 SM 上活跃 block 数下降
2. 活跃 warp 总数也可能下降

这本质上和寄存器压力很像：

- 你用 shared memory 换取更好的数据复用
- 但它也可能牺牲并发能力

### 为什么这很常见

很多高性能 kernel 都会大量使用 shared memory 来做 tile 和重用。  
这通常是值得的，但代价就是：

- 不能无限开大 tile
- 不能无限堆 shared memory

否则 occupancy 可能掉得很厉害。

### 关键直觉

shared memory 不是“免费的小缓存”。  
它是一个很强的工具，但用多了也会挤压并发。

---

## 6. `__syncthreads()` 是什么

`__syncthreads()` 是 block 内线程的同步原语。

可以粗糙理解成：

- 一个 block 里的线程在这里集合
- 大家都到齐之后，才能继续往下走

### 为什么需要它

最常见的原因是 shared memory。

例如：

1. 每个 thread 先把一部分数据写进 shared memory
2. 然后所有 thread 再去读 shared memory 里别的 thread 写的数据

如果中间没有同步，可能有人还没写完，别人就已经开始读了。

### 为什么它重要

因为 block 内线程虽然在协作，但它们不是自动严格同步前进的。

所以：

- 需要显式同步来保证数据准备好了

### 为什么它不能乱用

同步是有代价的。

如果你过多使用同步：

- 会让线程等待
- 会减少执行效率

所以同步应该只在真的需要的地方用。

### 重要边界

`__syncthreads()` 只能同步同一个 block 内的线程。  
它不能解决 block 之间的同步问题。

---

## 7. warp divergence 到底会怎么拖慢程序

`warp divergence` 指的是：

- 同一个 warp 里的线程走了不同控制流

例如：

```cpp
if (x > 0) {
    ...
} else {
    ...
}
```

如果一个 warp 里：

- 一部分线程 `x > 0`
- 一部分线程 `x <= 0`

那这个 warp 往往不能像“所有线程都走同一路径”那样高效执行。

### 为什么会慢

因为 warp 是成组执行的。  
当一组线程走不同路径时，硬件通常需要分开处理不同路径。

结果就是：

- 一部分线程执行时，另一部分线程相当于在等
- 有效并行度下降

### 什么场景容易 divergence

- 数据相关分支
- 不规则稀疏计算
- 不规则长度处理
- 边界判断很多的代码

### 为什么这和 AI infra 有关

因为很多系统优化最后都希望：

- 让同一个 warp 做更规则的事
- 减少无规则分支
- 让访问和控制流更整齐

这也是为什么 dense tensor 计算通常更“吃硬件”，而稀疏和不规则 workload 更难榨干 GPU。

---

## 8. 一个最小性能直觉总结

当你写一个 CUDA kernel 时，最基础的几个性能问题其实是：

1. 每个 thread 负责哪块数据？
2. 一个 warp 里的线程是不是在做相似的事？
3. 它们访问内存是不是连续、规整？
4. block size 会不会太大或太小？
5. shared memory 和寄存器是不是吃太多？
6. 有没有不必要的同步？

如果这几个问题都没想清楚，kernel 大概率不会太好。

---

## 9. 初学时最该先建立的直觉

你现在不需要先追求“会调最优参数”，而是先建立下面这些直觉：

### 直觉 1

索引不仅决定“算谁”，还决定访存模式。

### 直觉 2

block 是协作单位，warp 是硬件执行单位。

### 直觉 3

寄存器和 shared memory 用多了，不一定是好事，因为它们会压缩并发。

### 直觉 4

occupancy 重要，但不是唯一目标。

### 直觉 5

不规则分支和不规则访存，通常都会让 GPU 更难跑满。

---

## 10. 继续往下应该学什么

如果这份笔记里的东西你已经基本接受了，下一步最自然的是：

- `memory_access_patterns.md`
  - coalesced access
  - bank conflict
  - stride access
  - locality

- `cuda_sync_and_memory_model.md`
  - synchronization
  - memory fence
  - visibility

- `kernel_performance_tuning.md`
  - occupancy
  - register pressure
  - shared memory tradeoff
  - roofline
