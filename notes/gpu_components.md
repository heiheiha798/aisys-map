# GPU Components

这份笔记只负责一件事：

- 建立 GPU 硬件组织和存储层次的最小地图

它主要回答：

- `SM` 是什么
- `register / shared memory / local memory / L1/TEX / L2 / VRAM` 分别是什么
- 哪些是程序显式管理的，哪些是硬件自动管理的
- 哪些是 cache，哪些不是

它**不负责**展开这些主题：

- `CUDA core`
- `Tensor Core`
- `FMA`
- `MMA`
- `WMMA`
- 低精度 GEMM 为什么快

这些放到 [cuda_tensor_core_wmma.md](./cuda_tensor_core_wmma.md)。

---

## 1. 先看最短地图

如果先只记一个最小顺序，可以记成：

```text
register
-> shared memory / L1-TEX
-> L2
-> VRAM(global memory)
```

但这里有一个很重要的边界：

- `register` 不是 cache
- `shared memory` 也不是 cache
- `L1/TEX` 和 `L2` 才是硬件自动管理的 cache / cache-like 通路

所以这条链不是“全都是 cache”，而是：

- 从离计算更近，到离计算更远的一组存储/访存层

---

## 2. `SM` 是什么

`SM` 可以理解成 GPU 上反复复制的计算核心簇。

每个 `SM` 里通常有：

- 执行线程的计算资源
- 寄存器文件
- shared memory
- warp scheduler
- 一些 load/store 与控制逻辑

最容易记错的点是：

- `SM` 不是一个单独的算术单元
- 它更像是一整个“执行容器”

如果你要区分：

- `SM`
- `CUDA core`
- `Tensor Core`

那这份笔记只负责 `SM` 这一层。  
`CUDA core / Tensor Core` 的边界放到 [cuda_tensor_core_wmma.md](./cuda_tensor_core_wmma.md)。

---

## 3. `VRAM / global memory` 是什么

最粗略地看：

- GPU 板载显存在日常语境里更适合直接叫：
  - `VRAM`
  - 或者“显存”
- 现代 GPU 上常见的显存实现是：
  - `GDDR`
  - `HBM`
- CUDA 讨论里常说的 `global memory`，性能上通常可以近似理解成走显存路径

它的特点是：

- 容量最大
- 带宽高
- 但离 `SM` 更远
- 访问代价最高

它适合放：

- 模型权重
- 大张量
- KV cache
- 放不进片上小存储的数据

---

## 4. `register` 是什么

`register` 是离计算最直接、最快的一层线程私有存储。

它的特点是：

- 很快
- 很小
- 线程私有

如果一个中间结果一直留在寄存器里，访问成本最低。

但要注意：

- `register` 不是 cache
- 不是硬件偷偷帮你缓存的
- 它更像线程当前手里直接拿着的值

例如 GEMM 里的：

```cpp
float acc[TM][TN];
```

通常就尽量希望放在寄存器里。

---

## 5. `local memory` 是什么

`local memory` 这个名字非常容易误导。

它不是：

- 离线程很近
- 很快
- 某种“线程小缓存”

更准确地说：

- 它逻辑上属于线程私有
- 但物理上通常不在寄存器里
- 往往还是走更慢的内存路径，常常接近显存访问代价

它常见于：

- 寄存器不够
- 局部数组太大
- 编译器无法把某些局部对象保在寄存器里

所以从性能角度看：

- `local memory` 更像“线程私有但很慢的存储”

---

## 6. `shared memory` 是什么

`shared memory` 是位于 `SM` 上的一块小而快的 block 级共享存储。

它的特点是：

- 比显存快很多
- 容量小
- 一个 block 内线程共享
- 需要程序显式搬运和显式同步

它常见的用途是：

- 搬运高复用数据
- 降低 global memory 重复访问
- 重排访问模式
- 作为 block 内线程交换中间结果的中转站

最重要的边界是：

- `shared memory` 不是 cache
- 它不是硬件自动帮你填的
- 它更像程序员手动管理的小仓库

---

## 7. `L1/TEX` 是什么

`L1/TEX` 在 `ncu` 里经常是一个合并口径。

你可以先把它理解成：

- 靠近 `SM` 的一层 cache / load-store 通路
- 把 `L1 cache` 和 `texture` 相关路径合在一起看

这里最容易混淆的点是：

- `L1/TEX` 是硬件自动管理的近端 cache / 访存通路
- 它不是 `shared memory`

所以当你看到：

- `L1/TEX throughput` 很高

更直接的意思通常是：

- 靠近 `SM` 的 load/cache 路径已经很忙

而不是：

- “shared memory 很忙”
- 或者 “texture 在忙”

---

## 8. `L2 cache` 是什么

`L2 cache` 是比 `L1/TEX` 更大、更远、更共享的一层 cache。

它的特点是：

- 硬件自动管理
- 比 `L1/TEX` 更大
- 比 `L1/TEX` 更远
- 通常对多个 `SM` 更共享

它的作用是：

- 缓冲最近访问过的数据
- 减少直接打到显存的次数

所以一个很粗略但够用的直觉是：

- `L1/TEX` 更近、更小
- `L2` 更远、更大

---

## 9. `register`、`shared memory`、`L1/TEX`、`L2` 到底怎么区分

这是最容易混的一组边界。

### `register`

- 线程私有
- 不是 cache
- 程序/编译器直接使用

### `shared memory`

- block 共享
- 不是 cache
- 程序显式管理

### `L1/TEX`

- 靠近 `SM`
- 是硬件自动管理的 cache / 通路
- 不是程序显式分配的存储

### `L2`

- 更大、更共享
- 也是硬件自动管理

所以最重要的分界线就是：

- `register / shared memory`：程序显式使用的存储
- `L1/TEX / L2`：硬件自动管理的 cache / 访存层

---

## 10. 为什么这些层次重要

很多 GPU 性能问题，本质上都不是“算术公式错了”，而是：

- 数据是不是总从慢层取
- 高复用数据有没有被留在近处
- 访问模式是不是友好
- `SM` 有没有被持续喂饱

所以后面你看到很多优化，最终都在做下面这些事：

- 尽量把热点值留在寄存器
- 尽量把高复用块搬到 shared memory
- 尽量让 global memory 访问对 cache 友好
- 尽量降低直接访问显存的压力

---

## 11. 当前最容易混淆的点

- `global memory` 在性能讨论里，经常可以近似理解成走显存路径
- `local memory` 不是“local 所以快”
- `shared memory` 不是 cache
- `L1/TEX` 不是 shared memory
- `register` 虽然最靠近计算，但它也不是 cache
- “靠近计算单元”不等于“就是寄存器”

---

## 12. 后续可继续补的方向

- register pressure 为什么会影响 occupancy
- bank conflict 到底是什么
- coalesced memory access 为什么重要
- warp scheduler 和 load/store 路径怎么配合
