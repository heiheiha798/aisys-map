# NCU Notes: row_softmax_online

## 命令

```bash
/usr/local/cuda-12.4/bin/ncu \
  --target-processes all \
  --set full \
  --kernel-name row_softmax_online_kernel \
  ./row_softmax_online
```

## 结论

这个 kernel 已经明显不是“GPU 根本没吃满”的状态了。  
它更接近一个：

- `compute-heavy`
- 同时带有明显 `synchronization` 开销
- 也包含一定 memory 压力

的混合型 kernel。

如果必须粗略归类，它比教学版明显更偏：

- `compute-bound / sync-sensitive`

而不是纯粹 `memory-bound`。

## 关键依据

### 1. Compute Throughput 明显高于 Memory Throughput

- `Compute (SM) Throughput`: `61.58%`
- `Memory Throughput`: `37.34%`

而且 `ncu` 直接提示：

- `Compute is more heavily utilized than Memory`

这已经是最重要的信号。

说明这个 kernel 的主要矛盾不再是：

- 单纯等内存

而是：

- 计算本身
- warp-level 合并
- block 级同步

这些部分都已经开始占明显比重。

### 2. 这次 occupancy 已经很高

- `Grid Size = 4096`
- `Waves Per SM = 5.33`
- `Achieved Occupancy = 87.16%`

这和教学版的：

- `Grid Size = 128`
- `Achieved Occupancy = 16.07%`

完全不是一个状态。

所以这次可以更放心地看：

- kernel 本身的结构问题

而不是被“小 grid”这个问题掩盖掉。

### 3. Scheduler 已经比较忙

关键指标是：

- `Issue Slots Busy`: `70.39%`
- `SM Busy`: `70.39%`
- `One or More Eligible`: `70.95%`
- `Issued Warp Per Scheduler`: `0.71`

这说明：

- 已经有相当数量的 warp 在持续推进
- scheduler 不再像教学版那样大量空转

也就是说：

- 这个 kernel 已经真正进入“看内部结构”的阶段了

### 4. Warp Stall 里 CTA barrier 很显眼

`ncu` 明确提示：

- 每个 warp 平均有 `4.5 cycles` 在等待 sibling warps at a `CTA barrier`
- 这一类 stall 大约占总 issue 间隔的 `30.2%`

这说明：

- `__syncthreads()` 仍然是重要开销来源
- 虽然这个版本已经减少了 shared memory tree reduction
- 但 warp 间汇总时仍然需要 block 级同步

所以这版最准确的描述不是：

- “没有同步”

而是：

- **同步更少了，但同步仍然重要**

### 5. 这版更接近真实高性能 softmax 的结构

原因不是它“已经最优”，而是它具有这些特征：

- 每个 thread 先维护自己的 `(m, l)` 局部状态
- warp 内用 `__shfl_down_sync` 做 reduction
- shared memory 只保存 warp 级结果
- block 最后只做一次较轻的汇总

这种结构会自然带来：

- 更多算术和状态合并逻辑
- 更少 shared memory 中转
- 更高 occupancy

因此它的 profile 也自然和教学版很不一样。

## 关键指标摘录

- `Duration`: `12.83 us`
- `Grid Size`: `4096`
- `Block Size`: `256`
- `Waves Per SM`: `5.33`
- `Memory Throughput`: `327.86 GB/s`
- `Memory Throughput %`: `37.34%`
- `Compute (SM) Throughput %`: `61.58%`
- `Issue Slots Busy`: `70.39%`
- `SM Busy`: `70.39%`
- `Achieved Occupancy`: `87.16%`
- `L1/TEX Hit Rate`: `33.26%`
- `L2 Hit Rate`: `49.83%`

## 为什么它不像典型 elementwise 那样 memory-bound

如果只是简单 elementwise kernel，通常会看到：

- 每个元素计算量很少
- 大量时间在等数据
- memory throughput 很高而 compute 很低

但这里不是。

`row_softmax_online` 虽然也有大量读写，但还额外做了很多事情：

- `(m, l)` 状态更新
- `expf`
- warp-level merge
- block 级最终汇总

这些都会提高：

- 算术密度
- 控制流复杂度
- 同步敏感性

所以它更像一个真正的 mixed kernel，而不是纯搬运型 kernel。

## 和教学版对比，最该怎么看

对比这两个 softmax，最应该先看这几组指标：

- `Grid Size`: `128` -> `4096`
- `Achieved Occupancy`: `16.07%` -> `87.16%`
- `Compute Throughput`: `13.56%` -> `61.58%`
- `Memory Throughput`: `13.56%` -> `37.34%`

这说明：

1. 教学版首先暴露的是“规模太小、并发不足”。
2. online 版才开始暴露真正的 kernel 结构代价。
3. 一旦 GPU 被喂饱，online 版的算术和同步成本就会明显浮现出来。

## 这次实验最该记住的结论

`online softmax` 的价值不只是：

- 减少 shared memory tree reduction

更重要的是：

- 让 softmax 统计量变成可在线合并的 `(m, l)` 状态
- 并且让实现路径更自然地落到 warp-level primitive 上

从 profile 上看，这样的 kernel：

- 不再像教学版那样主要败在 occupancy
- 也不再像 elementwise 那样主要败在 memory
- 而是进入真正的 compute / sync / memory 混合权衡

这正是它更接近 FlashAttention 那条路的原因。
