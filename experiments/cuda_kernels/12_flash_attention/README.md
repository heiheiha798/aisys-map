# FlashAttention

这个目录保留一个最小可运行的 `FlashAttention-2` 风格 CUDA 学习用例：

- `flash_attention.cu`

这个目录的重点是：

- 用一个最小例子说明 `FlashAttention v1` 和 `FlashAttention-2` 的工作划分差别
- 讲清楚为什么 `sliced-Q` 比 `sliced-K` 更合理
- 让读者对 `Q / K / V` 的循环组织留下明确印象

这里最重要的内容其实不是 `.cu` 文件，而是这份 README。

主要参考：

- Hazy Research, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*  
  <https://hazyresearch.stanford.edu/blog/2023-07-17-flash2>

---

## 1. 先把定位说清楚

这个目录是一个学习用例，不是性能样例。

原因很简单：

- 真正高性能的 FlashAttention kernel，依赖非常细的线程映射、寄存器 blocking、shared memory 布局、向量化访存和硬件特化
- 这些东西一旦展开，教学价值反而会迅速下降

所以这个目录只保留两件事：

1. 一个最小可运行的 `flash_attention.cu`
2. 一份把 FA2 核心思想讲清楚的 README

如果你读完只记住了：

- FA1 的问题在 `sliced-K`
- FA2 的关键改进是 `sliced-Q`
- “Q 在外面更好”真正指的是“按 query rows 划分 warp 的责任边界”

那这个目录就达成目的了。

---

## 2. attention 数学没有变

不管是普通 attention、FA1 还是 FA2，数学上都还是：

```text
S = QK^T
P = softmax(S)
O = PV
```

真正变化的不是公式，而是：

- 这个公式如何在 GPU 上切分成可执行的工作单元

FlashAttention 这一类方法的共同目标是：

- 不显式写出完整的 score matrix
- 按 tile 处理 `Q/K/V`
- 用 online softmax 在 tile 之间维护数值稳定性

所以要把这件事分两层看：

1. online softmax 解决的是“不物化整张 `S` 还能精确算”
2. `sliced-K` / `sliced-Q` 解决的是“GPU 上该怎么分工更合理”

---

## 3. FA1: `sliced-K`

FA1 的块内分工可以粗略理解成：

- 多个 warp 共享同一个 `Q` tile
- 每个 warp 处理不同的 `K/V` slice

也就是：

```text
same Q
warp 0 -> K/V slice 0
warp 1 -> K/V slice 1
warp 2 -> K/V slice 2
warp 3 -> K/V slice 3
```

这会带来一个根本问题：

- 一行 query 的最终输出，被拆散到了多个 warp 上

于是每个 warp 只能先算出：

- 一部分 `QK^T`
- 一部分 softmax 状态
- 一部分 `PV` 输出

最后还必须做：

- 跨 warp 写回
- 同步
- 归约合并

所以 FA1 的低效点不是数学，而是：

- **同一行输出由多个 warp 共同生产，因此必须通信**

---

## 4. FA2: `sliced-Q`

FA2 把责任边界改到了 query rows 上。

它的思路是：

- 整个 block 共享同一个 `K/V tile`
- 不同 warp 负责不同的 `Q` rows

也就是：

```text
same K/V
warp 0 -> Q slice 0
warp 1 -> Q slice 1
warp 2 -> Q slice 2
warp 3 -> Q slice 3
```

这时每个 warp 都可以对自己那几行 query 负责到底：

1. 用自己的 `Q rows` 去乘当前 `K tile`
2. 更新自己的 online softmax 状态
3. 用 softmax 权重乘 `V tile`
4. 继续累计自己的输出

于是最重要的变化就是：

- 一行输出只属于一个 warp

这意味着大部分情况下都不再需要：

- 多 warp 共同拼同一行输出
- 频繁把部分结果写到 shared memory 再读回来
- 大量跨 warp synchronization

一句话概括：

- `sliced-K` 把一行输出拆给多个 warp
- `sliced-Q` 让一行输出从头到尾都只归一个 warp

---

## 5. “Q 在外层”到底是什么意思

这里最容易被说糊涂。

“Q 在外层更好”并不是在说：

- 你写伪代码时，一定要把 `for q` 写在最外层

它真正指的是：

- **GPU 工作划分时，把 query rows 作为责任边界**

判断标准非常简单：

- 一行输出是否只属于一个 warp

如果不是，那通常就会有：

- 合并
- 同步
- shared memory 中转

如果是，那通常就能自然变成：

- 一个 warp 顺序扫过所有 `K/V tiles`
- 自己维护自己的 `m / l / out`
- 最后直接写回自己的输出

所以“Q 在外层更好”最准确的意思是：

- **让每个 warp 对自己的 query rows 负责到底**

---

## 6. 一个最小例子

假设：

- `Q` 有 4 行：`q0 q1 q2 q3`
- `K/V` 被切成两块：`tile0`、`tile1`
- 一个 block 里有 2 个 warp

### 6.1 FA1 / `sliced-K`

```text
warp 0: 用 tile0 去算 q0 q1 q2 q3
warp 1: 用 tile1 去算 q0 q1 q2 q3
```

于是：

- `q0` 的最终输出要把 warp 0 和 warp 1 的结果合起来
- `q1` 也是
- `q2` 也是
- `q3` 也是

也就是说：

- 每一行都被拆散了

### 6.2 FA2 / `sliced-Q`

```text
warp 0: 负责 q0 q1，顺序扫过 tile0, tile1
warp 1: 负责 q2 q3，顺序扫过 tile0, tile1
```

于是：

- `q0 q1` 的完整输出都在 warp 0 内闭合
- `q2 q3` 的完整输出都在 warp 1 内闭合

也就是说：

- 每一行都没有被拆散

这就是最该记住的区别。

---

## 7. 为什么这更符合 GPU 硬件

### 7.1 warp 是天然执行单位

GPU 不是按单个线程调度，而是按 warp 调度。

因此更理想的结构通常是：

- 给一个 warp 一份尽量完整、闭合的工作
- 让它自己算完
- 少依赖别的 warp

FA2 更接近这个目标。

### 7.2 warp 之间通信很贵

只要多个 warp 共同生产同一行输出，就很容易引入：

- shared memory 中转
- `__syncthreads()`
- 额外归约

这些都不是免费的。

### 7.3 shared memory 不是免费午餐

shared memory 比 HBM 快，但仍然会带来：

- 读写指令成本
- bank conflict
- 同步成本

所以如果一个切分方式天然要求“先分散算，再合并”，那就会很容易把自己推向大量 shared memory 通信。

### 7.4 现代 GPU 更喜欢把时间花在 matmul 上

现代 GPU 在矩阵乘法上的吞吐非常强。  
相比之下，额外的：

- rescale
- 合并
- 同步
- 中间结果搬运

都更像“管理成本”。

FA2 的改进，本质上就是：

- 尽量把时间留给真正的 `QK^T` 和 `PV`
- 尽量减少为了并行切分不合理而付出的额外代价

---

## 8. 这个目录里的 `.cu` 文件该怎么理解

`flash_attention.cu` 只是一个最小可运行教学版。

它的作用是：

- 让这个目录里有一个真实可跑的 FlashAttention 风格 kernel

它不是：

- 官方实现复刻
- 生产级实现
- 性能对标样例

所以这里不要反过来理解成：

- “只要这份 `.cu` 比 vanilla attention 快，README 才成立”

恰恰相反：

- **README 对 FA2 工作划分的说明，才是这个目录的主体价值**

---

## 9. 这个目录最该记住什么

如果最后只保留 5 句话，那就是：

1. attention 数学没变，变的是 GPU 上的工作切法。
2. FlashAttention 的共同基础是 tiling + online softmax。
3. FA1 的主要问题在 `sliced-K`。
4. FA2 的关键改进是 `sliced-Q`。
5. `sliced-Q` 的本质收益是：一行输出只属于一个 warp，因此更少通信、更少同步、更少 shared memory 中转。
