# Attention: A Minimal CUDA Teaching Kernel

这个目录放一个最小可运行的 `scaled dot-product attention` CUDA 实验：

- `attention.cu`

这份说明会尽量把读者当成：

- 会写 Python
- 知道 attention 和 softmax 的基本公式
- 已经大概看过 `softmax/README.md`
- 但还没有认真把 attention kernel 从代码角度拆开

也就是说，这里不会默认你已经完全熟悉：

- 一个 attention kernel 为什么会同时包含 dot product、softmax、加权求和
- shared memory 为什么在这里很自然
- block 内 reduction 在 attention 里怎么出现

---

## 1. 这个实验到底在干什么

这份代码实现的是最基础的单头、无 mask 的 attention：

```text
scores = Q K^T / sqrt(head_dim)
probs  = softmax(scores)
out    = probs V
```

如果只看第 `i` 个 query row，它做的是：

```text
score[i, j] = dot(Q[i], K[j]) / sqrt(head_dim)
prob[i, :]  = softmax(score[i, :])
out[i]      = sum_j prob[i, j] * V[j]
```

也就是三步：

1. 当前 query 和所有 key 做点积
2. 这一行分数做 softmax
3. 用 softmax 权重对所有 value 做加权求和

这份教学版的核心价值，不是性能，而是把这三步完整放进一个最小 CUDA kernel 里。

---

## 2. 为什么 attention 比 softmax 更像一个“小流水线”

`softmax` 自己已经不是纯 elementwise，因为它需要 row-wise reduction。

attention 比 softmax 更进一步，因为它在同一行里串了三种不同工作：

1. `Q · K`
   - 一行 query 对所有 key 做点积
2. `softmax`
   - 需要 `max reduction` 和 `sum reduction`
3. `probs · V`
   - 再用权重做一次加权和

所以这类 kernel 很适合用来理解：

- 一个 block 不只是“算一堆独立元素”
- 它可以在 shared memory 里先放中间结果
- 然后同一个 block 再继续消费这些中间结果

这就是这份代码最值得观察的地方：

- **一个 block 把一整行 attention 从头做到尾**

---

## 3. 这份代码的整体分工

`attention.cu` 的组织方式非常直接：

- 一个 `block` 负责一个 `query_row`
- block 内线程先并行算这一行对所有 key 的分数
- 把分数放进 shared memory
- block 内做 softmax
- 再用 softmax 权重聚合所有 `V`

可以把它理解成：

```text
one block -> one query row
```

而 block 内的线程则按两种维度轮流分工：

- 在 “对所有 key_row 扫描” 这件事上分工
- 在 “输出向量的不同维度 d” 上分工

这也是为什么这份代码很适合教学：

- query 这一行的生命周期很完整
- 线程分工也不绕

---

## 4. 先看最重要的 kernel launch

`main` 里真正启动 kernel 的地方是：

```cpp
attention_kernel<<<seq_len, threads_per_block, shared_mem_bytes>>>(
    d_q, d_k, d_v, d_out, seq_len, head_dim, scale);
```

你可以先把这句理解成：

- 一共启动 `seq_len` 个 block
- 每个 block 有 `threads_per_block` 个线程
- 每个 block 还带一块动态 shared memory

代入当前参数：

- `seq_len = 64`
- `threads_per_block = 128`

也就是：

- 一共启动 64 个 block
- 每个 block 128 个线程

因为这里约定：

- 一个 block 处理一个 query row

所以：

- `block 0` 处理 `Q[0, :]`
- `block 1` 处理 `Q[1, :]`
- ...
- `block 63` 处理 `Q[63, :]`

---

## 5. kernel 的输入和输出是什么

kernel 定义是：

```cpp
__global__ void attention_kernel(const float* q, const float* k, const float* v,
                                 float* out, int seq_len, int head_dim,
                                 float scale)
```

这里的张量可以看成：

- `q[seq_len, head_dim]`
- `k[seq_len, head_dim]`
- `v[seq_len, head_dim]`
- `out[seq_len, head_dim]`

以及一个额外参数：

- `scale = 1 / sqrt(head_dim)`

这个 `scale` 就是 attention 公式里那项缩放：

```text
QK^T / sqrt(head_dim)
```

---

## 6. `int query_row = blockIdx.x;`

这一句表示：

- 当前 block 正在处理哪一行 query

因为 launch 是：

```cpp
<<<seq_len, threads_per_block, shared_mem_bytes>>>
```

所以：

- `blockIdx.x` 的范围就是 `0 ~ seq_len - 1`

这里把：

- block 号

直接映射成：

- query 行号

这是 attention 教学版里最自然的一种切分方式。

---

## 7. `int tid = threadIdx.x;`

这句表示：

- 当前线程在 block 内的编号

当前 block 配了 `128` 个线程，所以：

- `tid` 的范围是 `0 ~ 127`

这些线程不会分别处理不同的 query row，而是：

- 一起合作处理同一个 `query_row`

只是它们会在不同阶段分摊不同的工作。

---

## 8. 为什么这里要用动态 shared memory

代码里写了：

```cpp
extern __shared__ float shared[];
float* scores = shared;
float* reduce = shared + seq_len;
```

这表示：

- 当前 block 向运行时申请了一整块 shared memory
- 然后把它切成两段来用

第一段：

- `scores`
  - 长度是 `seq_len`
  - 用来存这一行 attention 的所有分数，后面也会被复用成 softmax 权重

第二段：

- `reduce`
  - 长度是 `threads_per_block`
  - 用来做 block 内 reduction

在 `main` 里，对应的 shared memory 大小是：

```cpp
const size_t shared_mem_bytes =
    (static_cast<size_t>(seq_len) + threads_per_block) * sizeof(float);
```

代入当前参数就是：

- `64 + 128 = 192` 个 `float`

也就是：

- `scores[64]`
- `reduce[128]`

这份代码很值得注意的一点是：

- 同一块 shared memory 先当 `scores`
- 再拿另一部分当 reduction workspace

这就是典型的“中间结果放在 block 内共享空间里反复使用”。

---

## 9. 第一阶段：并行计算 `scores`

代码先写了：

```cpp
for (int key_row = tid; key_row < seq_len; key_row += blockDim.x) {
  float dot = 0.0f;
  const float* q_row = q + static_cast<size_t>(query_row) * head_dim;
  const float* k_row = k + static_cast<size_t>(key_row) * head_dim;
  for (int d = 0; d < head_dim; ++d) {
    dot += q_row[d] * k_row[d];
  }
  scores[key_row] = dot * scale;
}
__syncthreads();
```

这段代码做的事非常直接：

- 对当前 `query_row`
- 遍历所有 `key_row`
- 计算 `dot(Q[query_row], K[key_row])`
- 再乘 `scale`
- 写入 `scores[key_row]`

### 9.1 为什么循环是 `key_row = tid; key_row += blockDim.x`

这表示：

- block 内线程分摊不同的 key 行

例如当前：

- `seq_len = 64`
- `blockDim.x = 128`

那只有前 64 个线程会各自算一个 `key_row`，后 64 个线程在这个阶段没有实际工作。

这不是最高效的配置，但很适合教学，因为：

- 逻辑简单
- `scores[key_row]` 的概念清楚

### 9.2 `q_row` 和 `k_row` 是什么

```cpp
const float* q_row = q + static_cast<size_t>(query_row) * head_dim;
const float* k_row = k + static_cast<size_t>(key_row) * head_dim;
```

这就是在一维连续内存里定位某一行的起始位置。

所以：

- `q_row[d]`
  - 就是 `Q[query_row, d]`
- `k_row[d]`
  - 就是 `K[key_row, d]`

### 9.3 这里的点积是怎么做的

```cpp
for (int d = 0; d < head_dim; ++d) {
  dot += q_row[d] * k_row[d];
}
```

这就是最朴素的向量点积：

```text
dot = sum_d Q[i, d] * K[j, d]
```

这份教学版没有做：

- tiled load
- 向量化
- shared memory 缓存 Q/K

因为这里的重点是先把数学链路写对。

---

## 10. 为什么这里需要 `__syncthreads()`

在算完所有 `scores[key_row]` 之后，代码立刻写了：

```cpp
__syncthreads();
```

原因很简单：

- 后面所有线程都要开始读 `scores`
- 所以必须保证前面负责写 `scores` 的线程都已经写完

这就是 block 内同步最典型的场景：

- 某些线程先生产 shared memory 数据
- 所有线程后续要共同消费

---

## 11. 第二阶段：先求 `row_max`

接下来代码开始做 softmax 的第一步：

```cpp
float local_max = -INFINITY;
for (int key_row = tid; key_row < seq_len; key_row += blockDim.x) {
  local_max = fmaxf(local_max, scores[key_row]);
}
reduce[tid] = local_max;
__syncthreads();
```

这段在做：

1. 每个线程先看自己负责的若干个 `scores`
2. 算一个局部最大值 `local_max`
3. 把局部最大值写到 `reduce[tid]`

这就是 block reduction 的标准起手式：

- 每个线程先算自己的局部结果
- 再把局部结果汇总

---

## 12. reduction 循环在干什么

代码随后写了：

```cpp
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
  if (tid < stride) {
    reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
  }
  __syncthreads();
}
```

这是最经典的 shared memory tree reduction 之一。

它的含义是：

- 第一轮，前一半线程和后一半线程两两合并
- 第二轮，再把剩下的一半继续两两合并
- 一直到只剩 `reduce[0]`

如果把 `reduce` 想象成一排数，这个循环就是在不断把这排数折半。

最后：

```cpp
float row_max = reduce[0];
```

也就是：

- 整个 query row 的最大分数

这一步和 `softmax/README.md` 里的逻辑是同一种 reduction 模式。

---

## 13. 第三阶段：计算稳定 softmax 的分子和分母

接下来代码写了：

```cpp
float local_sum = 0.0f;
for (int key_row = tid; key_row < seq_len; key_row += blockDim.x) {
  float exp_score = expf(scores[key_row] - row_max);
  scores[key_row] = exp_score;
  local_sum += exp_score;
}
reduce[tid] = local_sum;
__syncthreads();
```

这一步在做两件事：

1. 把原来的 `scores[key_row]` 改写成：

```text
exp(score - row_max)
```

2. 每个线程累加自己那部分的指数和

这里很关键的一点是：

- `scores` 这块 shared memory 被复用了

前面它保存的是：

```text
原始 attention score
```

现在它保存的是：

```text
exp(score - row_max)
```

这正是 shared memory 作为中间缓冲区的典型用法。

---

## 14. 第四阶段：求整行的指数和

接下来代码又做了一次和前面几乎一样的 reduction：

```cpp
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
  if (tid < stride) {
    reduce[tid] += reduce[tid + stride];
  }
  __syncthreads();
}
```

只不过这次不再是 `max`，而是：

- `sum`

最后：

```cpp
float row_sum = reduce[0];
```

所以现在我们已经得到了稳定 softmax 所需要的两样东西：

- `row_max`
- `row_sum`

---

## 15. 第五阶段：把 `scores` 归一化成真正的 softmax 权重

代码写了：

```cpp
for (int key_row = tid; key_row < seq_len; key_row += blockDim.x) {
  scores[key_row] /= row_sum;
}
__syncthreads();
```

这表示：

- 现在 `scores[key_row]` 不再是分数
- 也不再只是 `exp(score - row_max)`
- 而是变成了最终 softmax 权重

也就是：

```text
scores[key_row] = softmax(score_row)[key_row]
```

从这一步开始，你就可以把 `scores` 理解成：

- attention probability
- 或者 `probs`

---

## 16. 第六阶段：用 softmax 权重聚合 `V`

最后代码写了：

```cpp
for (int d = tid; d < head_dim; d += blockDim.x) {
  float acc = 0.0f;
  for (int key_row = 0; key_row < seq_len; ++key_row) {
    acc += scores[key_row] * v[static_cast<size_t>(key_row) * head_dim + d];
  }
  out[static_cast<size_t>(query_row) * head_dim + d] = acc;
}
```

这一步的含义是：

- block 内线程开始按输出维度 `d` 分工
- 对固定的输出维度 `d`
- 遍历所有 `key_row`
- 用 softmax 权重对 `V[key_row, d]` 做加权求和

数学上就是：

```text
out[query_row, d] = sum_j probs[query_row, j] * V[j, d]
```

### 16.1 为什么这里换成按 `d` 分工

因为现在 softmax 权重已经准备好了。

接下来的任务不再是“遍历所有 key_row 去算分数”，而是：

- 输出向量的每一维都要独立做一次加权和

所以最自然的分工就变成：

- 每个线程负责若干个 `d`

这就是 attention 这个 kernel 的第二次“分工切换”。

---

## 17. CPU reference 在做什么

CPU 版本是：

```cpp
void cpu_attention(...)
```

它基本上就是把 attention 公式按最直白的串行方式展开：

1. 对某个 `query_row`
2. 先算所有 `scores`
3. 求 `row_max`
4. 再算 `exp(scores - row_max)` 和 `row_sum`
5. 再对每个输出维度做加权和

它的价值非常直接：

- GPU kernel 再复杂
- 本质也必须和这份最简单的 reference 对齐

这就是教学代码里 CPU reference 的意义：

- 把并行写法锚定回最原始的数学定义

---

## 18. `main` 里的参数在表达什么

`main` 一开始写了：

```cpp
constexpr int seq_len = 64;
constexpr int head_dim = 32;
constexpr int threads_per_block = 128;
```

可以这样理解：

- `seq_len = 64`
  - 序列长度是 64
- `head_dim = 32`
  - 单个 attention head 维度是 32
- `threads_per_block = 128`
  - 一个 block 有 128 个线程

这组参数的特点是：

- 问题足够小，容易看懂
- 但又已经完整覆盖了 attention 的三段结构

这正适合“profile 之前”的教学版。

---

## 19. host 侧数据是怎么构造的

代码里构造了：

```cpp
std::vector<float> h_q(numel);
std::vector<float> h_k(numel);
std::vector<float> h_v(numel);
```

然后用 `sin / cos / 模运算偏移` 生成确定性数据。

这样做的目的不是模拟真实模型分布，而是：

1. 每次运行都可复现
2. 不同位置值不完全相同
3. 更容易暴露索引或 reduction 逻辑错误

这类教学样例里，确定性输入很重要，因为它让：

- “结果错了” 这件事更容易定位

---

## 20. 为什么 `scale` 单独算出来

代码里有：

```cpp
const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
```

它对应的就是 attention 公式里的：

```text
1 / sqrt(head_dim)
```

单独提前算出来有两个好处：

1. 数学含义清楚
2. kernel 里每次写 `scores[key_row] = dot * scale` 就很直接

---

## 21. 为什么误差阈值是 `1e-4`

最后程序用：

```cpp
bool ok = max_abs < 1e-4f;
```

这比纯加法 kernel 的阈值稍微放宽一点，很合理。

因为这份代码里涉及：

- 点积累加
- `exp`
- reduction
- 再次加权求和

也就是多次浮点运算叠加。

所以相比 `scatter` 这种纯加法样例，允许略大的浮点差异是正常的。

---

## 22. 编译和运行

编译：

```bash
cd experiments/cuda_kernels/attention
make
```

运行：

```bash
./attention
```

如果一切正常，你会看到类似输出：

- `attention passed`
- `seq_len`
- `head_dim`
- `threads_per_block`
- `max_abs_diff`
- 几个 sample 输出

---

## 23. 这份代码最值得记住的点

1. 这份教学版 attention 把三件事串在了一个 block 里：
   - `QK^T`
   - `softmax`
   - `probs V`
2. 一个 block 对应一个 query row，是这类最小样例里最自然的切分方式。
3. shared memory 在这里承担了两类角色：
   - 存整行 `scores`
   - 做 block 内 reduction workspace
4. attention 相比单独的 softmax，多出来的核心就是：
   - 前面有点积
   - 后面有对 `V` 的加权和
5. 这版代码的重点是把数据流和并行结构讲清楚，不是追求高性能 attention。
