# NCU Notes: row_softmax

## 命令

```bash
/usr/local/cuda-12.4/bin/ncu \
  --target-processes all \
  --set full \
  --kernel-name row_softmax_kernel \
  ./row_softmax
```

## 结论

这个 kernel 当前最显著的问题不是“更偏 compute 还是更偏 memory”，而是：

- **grid 太小**
- **occupancy 太低**
- **GPU 整体没有被真正喂饱**

所以它更像一个教学版 reduction kernel，而不是一个已经进入真实性能区间的 softmax kernel。

## 关键依据

### 1. Compute Throughput 和 Memory Throughput 都不高

- `Memory Throughput`: `13.56%`
- `Compute (SM) Throughput`: `13.56%`

这说明：

- 不是某一个子系统特别顶满了
- 而是整个 kernel 的整体利用率都偏低

如果它真是典型 memory-bound，通常会看到：

- memory throughput 很高
- compute throughput 明显更低

这里不是这种图像。

### 2. NCU 直接指出 grid 太小

`ncu` 的提示是：

- `This kernel grid is too small to fill the available resources on this device`
- `only 0.2 full waves across all SMs`

这已经非常直接了。

当前 launch 是：

- `Grid Size = 128`
- `# SMs = 128`

也就是说：

- 总共才 128 个 block
- 平均每个 SM 只分到差不多 1 个 block

这对一个含有同步的 kernel 来说很吃亏。

### 3. Achieved Occupancy 非常低

- `Theoretical Occupancy`: `100%`
- `Achieved Occupancy`: `16.07%`

这不是寄存器、shared memory 把 occupancy 压得很惨，而更像是：

- 根本没有足够多的 block / warp 同时在跑

所以很多硬件资源闲着。

### 4. Scheduler 利用率很差

关键指标是：

- `Issue Slots Busy`: `12.98%`
- `SM Busy`: `12.98%`
- `No Eligible`: `86.51%`
- `Issued Warp Per Scheduler`: `0.13`

这说明：

- scheduler 很多周期根本发不出指令
- 不是因为 ALU 满了
- 也不是因为 memory subsystem 顶满了
- 而是没足够多 ready 的 warps 来维持吞吐

### 5. `__syncthreads()` 在这种小 grid 下更吃亏

`ncu` 的提示里明确说：

- 如果执行 `__syncthreads()`，最好让每个 SM 上不只 1 个 block

原因很直观：

- 一个 block 在 barrier 等待时
- 如果同一个 SM 上没有别的 block 可以切换
- 硬件就更容易空转

而这个 kernel 正好：

- 有 block 内 reduction
- 有多轮 `__syncthreads()`
- block 数又不多

这会把“小 grid”的问题进一步放大。

## 关键指标摘录

- `Duration`: `3.14 us`
- `Grid Size`: `128`
- `Block Size`: `256`
- `Waves Per SM`: `0.17`
- `Memory Throughput`: `43.63 GB/s`
- `Memory Throughput %`: `13.56%`
- `Compute (SM) Throughput %`: `13.56%`
- `Issue Slots Busy`: `12.98%`
- `SM Busy`: `12.98%`
- `Achieved Occupancy`: `16.07%`

## 为什么现在不适合直接下“memory-bound”或“compute-bound”结论

这个 kernel 当然有：

- global memory 读写
- 两次 reduction
- 多次同步

但从 profile 看，它的第一问题仍然是：

- 规模太小
- 无法形成足够并发

所以如果现在硬要把它归成：

- `memory-bound`
- 或 `compute-bound`

都会有点失焦。

更准确的说法应该是：

- **这个实现当前主要是 under-occupied / under-filled**

## 这次实验最该记住的结论

这个 `row_softmax` 最适合用来建立：

- reduction 结构直觉
- shared memory 协作直觉
- `__syncthreads()` 为什么重要

但它的 profile 不代表“softmax 天生就是这样”。

这次最该记住的是：

- 同一个 softmax，教学版写法可能首先暴露的是“没有把 GPU 喂饱”
- 在这种情况下，讨论 compute-bound 还是 memory-bound 不是主问题
- 真正该先看的，是 grid、waves、occupancy、scheduler 利用率
