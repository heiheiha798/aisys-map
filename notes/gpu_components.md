# GPU Components

这份笔记只负责 GPU 硬件组织和存储层次的最小地图：

- `SM` 是什么
- `register / shared memory / local memory / L1-TEX / L2 / VRAM` 分别是什么
- 哪些由程序显式管理，哪些由硬件自动管理
- 哪些是 cache，哪些不是

`CUDA core / Tensor Core / FMA / MMA / WMMA` 放到 [cuda_tensor_core_wmma.md](./cuda_tensor_core_wmma.md)。

---

## 1. 最短地图

可以先按距离计算资源的远近记：

```text
register
-> shared memory / L1-TEX
-> L2
-> VRAM(global memory)
```

但这条链不是“全都是 cache”：

| 层级 | 谁管理 | 是不是 cache |
|---|---|---|
| register | 程序/编译器 | 不是 |
| shared memory | 程序显式搬运和同步 | 不是 |
| L1/TEX | 硬件自动管理 | 是 cache / cache-like 通路 |
| L2 | 硬件自动管理 | 是 cache |
| VRAM / global memory | 程序分配，硬件访问 | 不是 cache |

---

## 2. `SM`

`SM` 可以理解成 GPU 上反复复制的执行容器。每个 `SM` 通常包含：

- 执行线程的计算资源
- 寄存器文件
- shared memory
- warp scheduler
- load/store 与控制逻辑

最容易记错的是：`SM` 不是一个单独的算术单元，而是一组能调度 warp、执行指令、访问片上资源的计算核心簇。

---

## 3. `VRAM / global memory`

日常说的 GPU 显存可以叫 `VRAM`；现代 GPU 常见实现包括 `GDDR` 和 `HBM`。CUDA 讨论里的 `global memory`，性能上通常可以近似理解成走显存路径。

特点：

- 容量最大
- 带宽高
- 离 `SM` 更远
- 访问代价最高

适合放模型权重、大张量、KV cache，以及放不进片上小存储的数据。

---

## 4. `register`

`register` 是离计算最近、最快的一层线程私有存储。

特点：

- 很快
- 很小
- 线程私有
- 不是 cache

如果中间值能一直留在寄存器里，访问成本最低。GEMM 里的 accumulator 例如 `float acc[TM][TN]`，通常就希望尽量留在寄存器中。

寄存器不是越多越好：单线程寄存器用得太多，会降低一个 `SM` 上能同时驻留的 warp/block 数，甚至导致 spill。

---

## 5. `local memory`

`local memory` 这个名字很误导。它逻辑上属于线程私有，但物理上通常不在寄存器里，往往走更慢的内存路径，常常接近显存访问代价。

常见来源：

- 寄存器不够导致 spill
- 局部数组太大
- 编译器无法把某些局部对象保在寄存器里

性能直觉：`local memory` 更像“线程私有但很慢的存储”，不是“local 所以快”。

---

## 6. `shared memory`

`shared memory` 是位于 `SM` 上的小而快的 block 级共享存储。

特点：

- 比显存快很多
- 容量小
- 一个 block 内线程共享
- 程序显式搬运、显式同步
- 不是 cache

常见用途：

- 缓存高复用数据
- 降低 global memory 重复访问
- 重排访问模式
- 作为 block 内线程交换中间结果的中转站

把 shared memory 理解成程序员手动管理的小仓库，比理解成 cache 更稳。

---

## 7. `L1/TEX`

`L1/TEX` 在 `ncu` 里经常是一个合并口径，可以先理解成靠近 `SM` 的 cache / load-store 通路，把 `L1 cache` 和 texture 相关路径合在一起看。

边界：

- 它由硬件自动管理
- 它不是 `shared memory`
- `L1/TEX throughput` 高，通常说明靠近 `SM` 的 load/cache 路径很忙

不要把 `L1/TEX` 指标直接读成 shared memory 很忙。

---

## 8. `L2 cache`

`L2 cache` 是比 `L1/TEX` 更大、更远、更共享的一层 cache。

特点：

- 硬件自动管理
- 比 `L1/TEX` 更大
- 比 `L1/TEX` 更远
- 通常被多个 `SM` 共享

作用是缓冲最近访问过的数据，减少直接打到显存的次数。

---

## 9. 最容易混淆的边界

| 对象 | 最重要的边界 |
|---|---|
| `register` | 线程私有，不是 cache |
| `shared memory` | block 共享，程序显式管理，不是 cache |
| `local memory` | 线程私有的逻辑语义，不代表物理上快 |
| `L1/TEX` | 近端硬件 cache / 通路，不是 shared memory |
| `L2` | 更大、更共享的硬件 cache |
| `global memory` | 性能讨论里通常近似等于显存路径 |

最重要的分界线是：

- `register / shared memory`：程序显式使用或编译器直接分配的存储
- `L1/TEX / L2`：硬件自动管理的 cache / 访存层

---

## 10. 为什么这些层次重要

很多 GPU 性能问题，本质不是算术公式错了，而是：

- 数据是不是总从慢层取
- 高复用数据有没有被留在近处
- 访问模式是不是友好
- `SM` 有没有被持续喂饱

所以很多 kernel 优化最后都在做同几件事：

- 把热点值留在寄存器
- 把高复用块搬到 shared memory
- 让 global memory 访问对 cache 和带宽更友好
- 降低直接访问显存的压力

后续可以继续补：register pressure、bank conflict、coalesced memory access、warp scheduler 与 load/store 路径。
