# CUDA Kernel Advanced

这份笔记接在 [cuda_programming_objects.md](./cuda_programming_objects.md) 后面，只保留写教学 kernel 和读 NCU 时最常用的性能直觉。

## 1. 索引不只是正确性问题

索引决定两件事：

- 每个 thread 算哪块数据
- 一个 warp 内线程如何访问内存

常见一维索引：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

如果相邻 thread 访问相邻地址，通常更容易形成 coalesced access；如果访问由随机索引决定，就更容易 memory-bound。

## 2. 1D / 2D / 3D launch 的意义

维度不是“更高级”，而是让数据映射更自然：

| 数据形状 | 常见 launch |
|---|---|
| 向量、token 序列、展平数组 | 1D |
| 矩阵、图像、attention score | 2D |
| batched tensor、三维数据 | 3D |

重点不是维度本身，而是它是否让 warp 访问更规整、边界分支更少、tile 组织更清晰。

## 3. occupancy 是机会，不是目标

`occupancy` 可以粗略理解成一个 SM 上活跃 warp/block 的充实程度。它有助于隐藏 latency，但高 occupancy 不保证快。

会压低 occupancy 的常见原因：

- 每个 thread 用太多 register
- 每个 block 用太多 shared memory
- block size 过大或过小
- kernel 本身资源占用过高

判断时要问：当前 kernel 是缺并发隐藏 latency，还是已经被带宽/计算/同步限制？

## 4. register pressure 和 spill

寄存器快，但每个 SM 的寄存器总量有限。

- 每个 thread 用更多 register，可以减少内存访问。
- 但 register 太多，会减少可驻留 warp/block。
- 如果编译器放不下，还可能 spill 到 local memory，走慢路径。

所以“更多寄存器”不是绝对好事。

## 5. shared memory 的收益和代价

shared memory 适合：

- 缓存会被重复使用的数据
- 改善 global memory 访问模式
- 做 block 内线程交换和 reduction workspace

代价是：

- 容量小
- 需要显式搬运
- 需要 `__syncthreads()`
- 用太多会减少一个 SM 上能驻留的 block 数

## 6. warp divergence

同一个 warp 里的线程走不同分支时，硬件通常要分路径执行，有效并行度下降。

常见来源：

- 数据相关分支
- 不规则长度
- sparse / indexed 访问
- 大量边界判断

优化方向通常是让同一个 warp 做更相似、更规整的事情。

## 7. 最小性能 checklist

写或读一个 kernel 时，先问：

1. 每个 thread / warp / block 分别负责什么？
2. warp 内访问是否连续、规整？
3. 数据是否有复用，值得放到 shared memory 或 register？
4. register 和 shared memory 是否压低了 occupancy？
5. 是否有不必要同步？
6. bottleneck 更像 compute、memory、sync，还是 runtime overhead？
