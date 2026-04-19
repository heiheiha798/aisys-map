# Online Softmax

这个目录放一个比教学版 `row_softmax` 更进一步的实验：

- `row_softmax_online.cu`

它的重点不是“把 softmax 再写一遍”，而是：

- 引入 `online softmax` 的状态合并思想
- 引入 warp-level reduction
- 减少对 shared memory tree reduction 的依赖

这份说明会继续把读者当成：

- 会写 Python
- 知道 softmax 是什么
- 已经看过基础版 `row_softmax`
- 但还没有真正把 `online softmax` 和 `FlashAttention` 的关系想清楚

---

## 1. 这份实验跟基础版 softmax 有什么不同

基础版 `row_softmax` 的思路是：

1. 整行先做一次 `max reduction`
2. 再算 `exp(x - row_max)`
3. 再做一次 `sum reduction`
4. 最后归一化

这个版本非常适合教学，因为它把 softmax 拆成了几个清楚的阶段。

但它也有很明显的特点：

- 阶段边界很硬
- 整个 block 大量依赖 shared memory tree reduction
- 你会先把“找最大值”和“算和”当成两个完全分开的大步骤

`online softmax` 不一样。

它的核心思想是：

- 你在扫描数据时
- 可以一边看新元素
- 一边维护一个稳定的 softmax 统计状态

这个状态不是单个数，而是两个量：

- `m`: 到目前为止看到的最大值
- `l`: 以这个最大值为参考系的累计和

所以这份实验最关键的不是某句 CUDA 语法，而是：

**为什么 softmax 可以用 `(m, l)` 这种状态表示，并且这些状态还能继续合并。**

---

## 2. 先回顾一下 stable softmax

普通 softmax 写成：

```text
softmax(x_i) = exp(x_i) / sum_j exp(x_j)
```

但直接这样算容易数值不稳定，所以通常会改写成：

```text
softmax(x_i) = exp(x_i - m) / sum_j exp(x_j - m)
```

其中：

- `m = max_j x_j`

这样做的原因是：

- 避免 `exp(x)` 溢出

这一步你已经熟了。  
但 `online softmax` 比这更进一步，它问的是：

> 如果我不是一次看到整行，而是分块看到一些元素，能不能始终维护一个正确的 softmax 统计状态？

答案是可以。

---

## 3. `(m, l)` 到底代表什么

假设你已经扫过一部分元素：

```text
x_0, x_1, ..., x_k
```

那我们可以把当前状态记成：

- `m = max(x_0, ..., x_k)`
- `l = sum_t exp(x_t - m)`

注意这里的 `l` 不是原始 `exp(x_t)` 的和，而是：

- 在当前最大值 `m` 这个参考系下的和

这样定义的好处是：

- 数值稳定
- 后面可以和另一部分状态合并

例如：

- 第一段数据有状态 `a = (m_a, l_a)`
- 第二段数据有状态 `b = (m_b, l_b)`

那它们的总状态就是：

```text
m = max(m_a, m_b)
l = l_a * exp(m_a - m) + l_b * exp(m_b - m)
```

这就是 online softmax 最核心的公式。

---

## 4. 为什么这个合并公式是对的

这是这份实验里最值得真正理解的地方。

假设第一段的和是：

```text
l_a = sum_i exp(x_i - m_a)
```

第二段的和是：

```text
l_b = sum_j exp(y_j - m_b)
```

如果总最大值变成：

```text
m = max(m_a, m_b)
```

那第一段原来是在 `m_a` 坐标系下表示的和，想改写到 `m` 坐标系下，就要乘：

```text
exp(m_a - m)
```

因为：

```text
exp(x_i - m_a) * exp(m_a - m) = exp(x_i - m)
```

第二段同理。

所以总和就变成：

```text
l = l_a * exp(m_a - m) + l_b * exp(m_b - m)
```

这就是为什么两个局部状态能继续合并。

你可以把它理解成：

- 每一段都用自己的最大值做归一化
- 合并时，把它们重新换算到同一个最大值坐标系

---

## 5. 代码里这个思想体现在哪

最关键的结构体是：

```cpp
struct OnlineSoftmaxState {
  float m;
  float l;
};
```

它表示的就是：

- 当前最大值
- 当前累计和

而最关键的函数是：

```cpp
__device__ __forceinline__ OnlineSoftmaxState merge_states(
    OnlineSoftmaxState a, OnlineSoftmaxState b)
```

逻辑是：

1. 如果某个状态还是空的，就直接返回另一个
2. 否则取新的最大值
3. 把两个状态都换算到这个最大值坐标系下
4. 把它们的和加起来

核心代码是：

```cpp
out.m = fmaxf(a.m, b.m);
out.l = a.l * expf(a.m - out.m) + b.l * expf(b.m - out.m);
```

如果你把整份代码只记一行，应该记这一行。

---

## 6. kernel 的整体分工还是“一行一个 block”

这份实验虽然内部逻辑更高级，但整体分工仍然很清楚：

- 一个 `block` 处理一整行
- 一个 `thread` 处理这一行里的一个或多个元素

当前参数是：

- `rows = 4096`
- `cols = 256`
- `threads_per_block = 256`

所以：

- 一共会有 4096 个 block
- 每个 block 256 个线程

和基础版相比，这里的 `rows` 明显更大。  
这样做的主要目的是：

- 让 GPU 真正有足够多的 block 可以调度
- 避免因为 grid 太小导致 occupancy 太低

---

## 7. 先看 kernel 一开始的几个索引

代码是：

```cpp
int row = blockIdx.x;
int tid = threadIdx.x;
int lane = tid & 31;
int warp_id = tid >> 5;
int num_warps = blockDim.x >> 5;
```

这里除了你熟悉的：

- `row`
- `tid`

还多了三个和 warp 相关的量。

### `lane`

表示当前 thread 在 warp 内的位置。

因为一个 warp 有 32 个线程，所以：

- `lane` 的范围是 `0 ~ 31`

### `warp_id`

表示当前 thread 属于 block 内的第几个 warp。

如果 `blockDim.x = 256`，那：

- 一共 8 个 warp

所以：

- `warp_id` 的范围是 `0 ~ 7`

### `num_warps`

表示这个 block 里一共有多少个 warp。

在当前例子里就是：

- `256 / 32 = 8`

这些量的作用是：

- 后面要先做 warp 内 reduction
- 再做 warp 间汇总

---

## 8. 为什么 shared memory 只剩下 8 个槽

代码里写的是：

```cpp
__shared__ float warp_max[8];
__shared__ float warp_sum[8];
```

这和基础版很不一样。

基础版里 shared memory 往往要存：

- 每个 thread 的中间结果

但这里不是。

这里 shared memory 只用来存：

- 每个 warp 的中间结果

为什么是 8 个？

因为：

- 256 个线程
- 每 32 个线程一个 warp
- 所以一共 8 个 warp

你可以把它理解成：

- 先在每个 warp 内部把事情谈妥
- 只把 warp 级别的结果写到 shared memory

这比“所有线程都把结果写到 shared memory 再做整块 reduction”更轻。

---

## 9. 每个 thread 是怎么维护自己的 online 状态的

代码是：

```cpp
OnlineSoftmaxState state;
state.m = -INFINITY;
state.l = 0.0f;

for (int col = tid; col < cols; col += blockDim.x) {
  float v = x[row * cols + col];
  OnlineSoftmaxState cur;
  cur.m = v;
  cur.l = 1.0f;
  state = merge_states(state, cur);
}
```

这里要特别注意：

- 单个元素 `v` 也可以看成一个状态

因为对一个元素来说：

- 最大值就是它自己
- 在这个最大值坐标系下，`exp(v - v) = 1`

所以单个元素对应的状态就是：

- `(v, 1)`

然后每个 thread 在扫自己负责的元素时，不断把：

- 当前累计状态 `state`

和：

- 新元素对应的状态 `cur`

合并起来。

这就是在线更新的过程。

---

## 10. warp 内 reduction 为什么可以不用 shared memory

代码是：

```cpp
state = warp_reduce_state(state);
```

而 `warp_reduce_state` 里面用的是：

```cpp
__shfl_down_sync(...)
```

这是 warp-level primitive。

它的含义可以先粗糙理解成：

- 让一个 warp 内的线程直接交换寄存器里的值

这样做的好处是：

- 不用先把数据写到 shared memory
- 不用像 block-level tree reduction 那样每一步都 `__syncthreads()`
- warp 内通信更轻

函数里的循环：

```cpp
for (int offset = 16; offset > 0; offset >>= 1)
```

逻辑和 tree reduction 很像：

- 先和相距 16 的线程合并
- 再和相距 8 的线程合并
- 再 4、2、1

最后 warp 的 lane 0 会拿到整个 warp 的合并结果。

---

## 11. 为什么 warp-level reduction 还能直接复用 merge 逻辑

这里很漂亮的一点是：

- 不管你是在扫元素
- 还是在 warp 内合并线程结果
- 还是在 warp 之间继续合并

本质上都只是：

- 把两个 `(m, l)` 状态并成一个更大的 `(m, l)` 状态

也就是说：

- `merge_states` 是统一的组合规则

这让代码结构很干净，也更接近 FlashAttention 里那种：

- 分块算局部统计量
- 再不断合并局部统计量

---

## 12. 为什么 lane 0 要把结果写到 shared memory

代码是：

```cpp
if (lane == 0) {
  warp_max[warp_id] = state.m;
  warp_sum[warp_id] = state.l;
}
```

每个 warp 在完成 warp 内 reduction 后：

- 只有 lane 0 持有这个 warp 的最终状态

所以这里只需要 lane 0 把结果写出去。

写完后：

- `warp_max[0..7]`
- `warp_sum[0..7]`

分别就存了 8 个 warp 的结果。

这一步之后才需要：

```cpp
__syncthreads();
```

因为接下来 warp 0 要读取所有 warp 的结果。

---

## 13. block 级别的最终合并是怎么做的

代码是：

```cpp
OnlineSoftmaxState block_state;
if (warp_id == 0) {
  if (lane < num_warps) {
    block_state.m = warp_max[lane];
    block_state.l = warp_sum[lane];
  } else {
    block_state.m = -INFINITY;
    block_state.l = 0.0f;
  }

  block_state = warp_reduce_state(block_state);

  if (lane == 0) {
    warp_max[0] = block_state.m;
    warp_sum[0] = block_state.l;
  }
}
```

大白话解释：

- 现在 block 里一共有 8 个 warp 结果
- 不需要所有 warp 都来参与最终汇总
- 只让 warp 0 来做就够了

warp 0 的前 8 个 lane 会分别读取：

- 第 0 个 warp 的状态
- 第 1 个 warp 的状态
- ...
- 第 7 个 warp 的状态

剩余 lane 没有有效数据，就填空状态：

- `(-INFINITY, 0.0f)`

然后 warp 0 再做一次 `warp_reduce_state`，就得到整个 block，也就是整行的最终状态。

最后：

- `warp_max[0]` 里放行最大值
- `warp_sum[0]` 里放行和

---

## 14. 为什么这里还有两次 `__syncthreads()`

很多人看到 warp-level primitive，会误以为：

- 从此不用同步了

这不对。

warp 内部很多时候确实不需要 block 级同步，但只要你：

- 把 warp 结果写到 shared memory
- 再让别的 warp 去读

那就还是需要 `__syncthreads()`。

这里的两个同步点分别对应：

### 第一个同步点

- 确保所有 warp 的 lane 0 都把结果写到 `warp_max/warp_sum` 了
- 然后 warp 0 才能安全读取

### 第二个同步点

- 确保 warp 0 已经把最终的 `row_max/row_sum` 写回 `warp_max[0]/warp_sum[0]`
- 这样整个 block 的其他线程才能安全读取

所以这里并不是“没有同步”，而是：

- 同步点更少了
- shared memory 的使用范围更小了

---

## 15. 最后为什么还要再算一次 `exp`

代码是：

```cpp
for (int col = tid; col < cols; col += blockDim.x) {
  float exp_val = expf(x[row * cols + col] - row_max);
  y[row * cols + col] = exp_val / row_sum;
}
```

看到这里很多人会问：

- 既然前面已经维护了 `l`
- 为什么不顺手把最终输出也一起算出来？

原因是：

- 前面维护的是统计量
- 不是每个元素的最终归一化结果

`row_max` 和 `row_sum` 虽然已经有了，但每个元素自己的：

```text
exp(x_i - row_max)
```

还是得各自算出来。

所以这份实验没有像 FlashAttention 那样把：

- softmax 统计更新
- 输出累计

完全揉在一起。

它做的是更基础的一步：

- 先把 online softmax 的统计量维护搞清楚

---

## 16. 这份代码和 FlashAttention 的关系到底是什么

最容易犯的错是：

- 把 online softmax 当成 FlashAttention 本身

这不对。

更准确地说：

- `online softmax` 是 FlashAttention 能成立的关键数学部件之一
- 但 FlashAttention 还包含更完整的 tile 调度、片上存储复用、输出累计方式

你可以把这份实验理解成：

- 先把 “softmax 统计量可以在线合并” 这件事单独练清楚

这样你后面再看 FlashAttention，就不会只看到一堆复杂 tile，而忽略它最核心的数值逻辑。

---

## 17. host 侧代码在做什么

`main()` 的结构和基础版类似，主要还是：

1. 在 CPU 上准备输入
2. 分配 GPU 内存
3. 把输入拷到 GPU
4. 启动 kernel
5. 拿回结果
6. 用 CPU 版本对拍 correctness

这里额外值得注意的点是：

- `rows = 4096`
- `cols = 256`

这个规模比基础版大很多。

这么做不是因为 softmax 数学上需要这么大，而是因为：

- 如果 block 数太少
- 很多 GPU 性能现象根本看不出来

也就是说，这里参数的一部分目的是：

- 让 profile 更接近真实 kernel 在 GPU 上“跑起来”的样子

---

## 18. 这份实验为什么比基础版更像“性能版”

不是说它已经是工业级实现，而是它明显更接近高性能思路：

### 1. 用 warp-level reduction

比“所有线程都走 shared-memory tree reduction”更接近真实优化路径。

### 2. shared memory 压得更小

这里只存每个 warp 的结果，而不是每个 thread 的结果。

### 3. occupancy 更容易做高

因为实验参数更大，GPU 更容易真正被喂饱。

### 4. kernel 画像更复杂

它不再只是“最基础 reduction 教学题”，而开始同时体现：

- 算术开销
- warp 内通信
- block 级同步
- memory 读写

所以它比基础版更接近你后面会遇到的真实 kernel。

---

## 19. 这份实现仍然故意保留了哪些简化

虽然它比基础版更进一步，但它仍然不是完整高性能 softmax。

这里故意没有做的事包括：

- 没有 vectorized load/store
- 没有半精度或 Tensor Core 相关路径
- 没有进一步减少最后那次 `exp`
- 没有处理更复杂的 mask
- 没有把 softmax 和后续 `V` 累计融合起来

这些简化是合理的，因为这份实验的目标仍然是：

- 先把 `online softmax` 这个核心思想单独讲明白

---

## 20. 一句话总结这份 kernel

这份 `row_softmax_online` 的核心思路可以压成一句话：

> 每个 thread 先把自己负责的元素在线合并成一个 `(m, l)` 状态；每个 warp 再把线程状态合并成 warp 状态；最后整个 block 再把所有 warp 状态合并成整行状态，然后再完成最终归一化。

---

## 21. 现在你最该记住的几个点

1. `online softmax` 的核心不是“少一个 pass”，而是 `(m, l)` 状态可以稳定合并。
2. 单个元素本身也可以看成一个 online softmax 状态：`(v, 1)`。
3. warp-level reduction 让 block 内协作不必每一步都依赖 shared memory tree reduction。
4. 这份实验已经比教学版更接近 FlashAttention 的思维方式，但还不是 FlashAttention。
5. `online softmax` 解决的是统计量如何分块合并，`FlashAttention` 解决的是整个 attention 的 IO 和执行顺序重排。

---

## 22. 建议怎么使用这份实验

最好的使用顺序是：

1. 先把基础版 `04_softmax/README.md` 看懂
2. 再看这份 `07_online_softmax/README.md`
3. 边看边对照 `merge_states` 和 `warp_reduce_state`
4. 最后再回到 `notes/attention_flash_bridge.md`，把它接到 `FlashAttention`

这样路径最稳。

如果反过来直接去看 FlashAttention 论文或实现，你很容易：

- 看懂 tile 结构
- 但没真正抓住为什么 softmax 统计量能边算边合并

---

## 23. 下一步最自然的问题

如果你已经接受这份代码，下一步最自然会问：

- 为什么 `online softmax` 对 attention 特别重要？
- 如何把 `row-wise online softmax` 推广到 `QK^T` 的 tiled 处理中？
- 为什么 FlashAttention 不只是 online softmax？
- prefill 和 decode 下，这套思路的价值为什么不一样？

这些问题正好就是你接下来往 `attention / FlashAttention / KV cache` 继续学的入口。
