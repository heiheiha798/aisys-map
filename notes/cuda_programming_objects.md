# CUDA Programming Objects

这份笔记只保留 CUDA 编程对象之间的关系，用来支撑 `02_kernel_intro/cuda_kernels/`。

## 总图

```text
kernel launch
  -> grid
    -> block
      -> warp
        -> thread
SM 执行 block/warp，并提供 register、shared memory、scheduler 等资源
```

| 对象 | 含义 | 最容易混淆的点 |
|---|---|---|
| kernel | 由 host launch 的 GPU 函数 | 不是一个 thread，也不是一个 block |
| grid | 一次 launch 的所有 block | 决定全局并行空间 |
| block | 一组可协作线程 | block 内可 shared memory + `__syncthreads()`，block 间不能直接同步 |
| warp | 硬件执行基本组，通常 32 个线程 | thread 不是完全独立随意执行，而是按 warp 成组推进 |
| thread | 最小编程视角 | 负责一小块数据，但执行受 warp/block 组织约束 |
| SM | 执行 block/warp 的硬件核心簇 | 一个 SM 可驻留多个 block，受寄存器、shared memory、warp 槽位限制 |

## 最常见索引关系

一维数组最常见写法：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < n) {
    y[idx] = x[idx] * 2.0f;
}
```

这段代码里：

- `threadIdx.x`：线程在 block 内的位置
- `blockIdx.x`：block 在 grid 内的位置
- `blockDim.x`：每个 block 的线程数
- `idx`：映射到全局数据下标
- `if (idx < n)`：防止多 launch 的线程越界

## 同步边界

| 同步对象 | 能做什么 | 不能做什么 |
|---|---|---|
| `__syncthreads()` | 同一个 block 内线程等待，常用于 shared memory 数据准备后 | 不能同步不同 block |
| kernel launch 边界 | 一个 kernel 完成后再启动下一个 kernel | 成本高，不能拿来替代细粒度同步 |
| stream/event | host/runtime 层控制顺序和依赖 | 不改变单个 kernel 内的 block 间同步限制 |

## 为什么 block size 重要

block size 同时影响：

- 一个 block 有多少 warp
- 每个 SM 能驻留多少 block
- register / shared memory 使用量
- latency hiding 能力
- 边界线程和分支形状

所以 block size 不是越大越好，也不是越小越好；它是资源占用和并发之间的权衡。

## 读代码时先问

1. 一个 thread 负责哪个数据元素或 tile？
2. 一个 block 负责哪个数据区域？
3. 一个 warp 内线程访问是否连续？
4. 是否使用 shared memory？如果用，哪里需要同步？
5. block 之间是否存在隐含依赖？如果有，设计大概率有问题。
