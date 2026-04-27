# Attention: A Minimal CUDA Teaching Kernel

这个目录放一个最小可运行的 `scaled dot-product attention` CUDA 实验：

- `attention.cu`

这份说明会尽量把读者当成：

- 会写 Python
- 知道 attention 和 softmax 的基本公式
- 已经大概看过 `04_softmax/README.md`
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

## LLM 场景

如果你是从 LLM 推理进入这个话题，真正容易糊涂的地方通常不是：

- `QK^T -> softmax -> PV`

而是：

- 训练 / prefill 时在算什么
- decode 时“只输入 1 个 token”到底怎么算 attention
- `KV cache` 是什么时候写进去的

这几件事如果没理顺，后面的 kernel 看起来都会像“只是公式”。

### 先给一个完全数字化的例子

下面先固定一组参数，不然 `Q/K/V` 很容易一直停留在抽象符号层面。

假设当前模型的这一层有：

- `hidden_dim = 2048`
- `num_heads = 16`

那每个 head 的维度就是：

```text
head_dim = hidden_dim / num_heads = 2048 / 16 = 128
```

先假设这是最普通的 multi-head attention，不考虑 GQA / MQA，也不考虑 batch > 1。

那么：

- 输入到这一层 attention 之前，每个 token 的隐藏状态长度是 `2048`
- 做完 `W_q / W_k / W_v` 线性投影之后，`q / k / v` 的总宽度也还是 `2048`
- 然后再把这 `2048` 拆成 `16` 个 head，每个 head `128`

所以如果当前有 `T` 个 token，那么在“还没拆 head”的视角里：

```text
X[T, 2048]
Q[T, 2048]
K[T, 2048]
V[T, 2048]
```

在“已经拆成多头”的视角里，更常见的理解是：

```text
Q[T, 16, 128]
K[T, 16, 128]
V[T, 16, 128]
```

如果你只看某一个 head，比如第 `h=7` 个 head，那它其实就是：

```text
Q_h[T, 128]
K_h[T, 128]
V_h[T, 128]
```

而我们这个目录里的教学版 `attention.cu`，本质上就是在讲：

- **对某一个 head 的 `Q_h / K_h / V_h`，一行一行做 attention**

也就是说，这里可以直接把它理解成：

```text
Q[seq_len, head_dim]
K[seq_len, head_dim]
V[seq_len, head_dim]
```

其中在这个“单头视角”里：

- `seq_len`
  - 当前参与 attention 的 token 数量
- `head_dim`
  - 当前 head 的向量长度

### 先把 prefill 的 shape 写死

假设现在 prompt 长度是：

```text
T = 4
```

那进入这一层 attention 之前，这 4 个 token 的隐藏状态是：

```text
X[4, 2048]
```

经过三组投影矩阵后：

```text
Q[4, 2048]
K[4, 2048]
V[4, 2048]
```

拆成 16 个 head 之后：

```text
Q[4, 16, 128]
K[4, 16, 128]
V[4, 16, 128]
```

如果只看第 7 个 head，那么就是：

```text
Q_7[4, 128]
K_7[4, 128]
V_7[4, 128]
```

这时 attention 在这个 head 上算的就是：

```text
[4, 128] x [4, 128]^T -> [4, 4]
softmax([4, 4]) -> [4, 4]
[4, 4] x [4, 128] -> [4, 128]
```

所以 prefill 时，对单个 head 来说，本质上是在算：

```text
4 个 query rows
对 4 个 key rows
做完整 attention
```

当然真实 LLM 里还会加 causal mask，所以实际上：

- 第 0 行只能看第 0 列
- 第 1 行只能看第 0、1 列
- 第 2 行只能看第 0、1、2 列
- 第 3 行只能看第 0、1、2、3 列

但 shape 仍然是：

```text
Q_7[4, 128]
K_7[4, 128]
V_7[4, 128]
```

### 再把 decode 的 shape 写死

现在假设 prefill 已经结束，历史长度是：

```text
past_len = 4
```

也就是说，前面 4 个 token 的 `K/V` 已经缓存好了。

当前来了第 5 个 token，也就是只新输入：

```text
X_new[1, 2048]
```

经过投影之后：

```text
Q_new[1, 2048]
K_new[1, 2048]
V_new[1, 2048]
```

拆成多头后：

```text
Q_new[1, 16, 128]
K_new[1, 16, 128]
V_new[1, 16, 128]
```

如果仍然只看第 7 个 head，那么就是：

```text
q_new_7[1, 128]
k_new_7[1, 128]
v_new_7[1, 128]
```

而历史 cache 里，第 7 个 head 已经存着：

```text
K_cache_7[4, 128]
V_cache_7[4, 128]
```

append 当前 token 之后，会变成：

```text
K_cache_7[5, 128]
V_cache_7[5, 128]
```

所以 decode 时，这个 head 真正做 attention 的 shape 是：

```text
Q_decode_7[1, 128]
K_decode_7[5, 128]
V_decode_7[5, 128]
```

计算过程是：

```text
[1, 128] x [5, 128]^T -> [1, 5]
softmax([1, 5]) -> [1, 5]
[1, 5] x [5, 128] -> [1, 128]
```

这就是 decode 时最应该记住的一句话：

- **新 token 只有 1 行 query，但它会对“历史 + 当前”的全部 K/V 做 attention**

所以 decode 时不是：

```text
Q[1, 128], K[1, 128], V[1, 128]
```

然后做一个无聊的 `1x1 attention`。

而是：

```text
Q[1, 128]
K[past_len + 1, 128]
V[past_len + 1, 128]
```

这里如果 `past_len = 4`，那就是：

```text
Q[1, 128]
K[5, 128]
V[5, 128]
```

### 先分清 prefill 和 decode

在 LLM 里，通常会有两种阶段。

第一种是 `prefill`：

- 一次把一整段 prompt 喂进去
- 例如 `T = 128` 个 token 一起进模型

这时候对某一层、某一个 head，可以粗糙写成：

```text
Q[128, d]
K[128, d]
V[128, d]
```

然后做 causal attention：

- 第 0 个 token 只能看自己
- 第 1 个 token 看第 0、1 个
- ...
- 第 127 个 token 看前面全部 128 个

这时候更像“整段一起算 attention”。

第二种是 `decode`：

- prompt 已经处理完了
- 现在每次只生成 1 个新 token

这时最容易混淆，因为：

- 新输入只有 1 个 token
- 但 attention 不是只跟自己算
- 它还要看前面所有历史 token 的 K/V

所以 decode 的核心不是：

- “attention 只剩 1x1”

而是：

- “query 只有 1 行，但 key/value 来自整个历史 cache”

### decode 时到底在算什么

假设现在已经生成过 3 个历史 token：

```text
t0, t1, t2
```

它们的 K/V 已经在 cache 里：

```text
K_cache[0] = k0
K_cache[1] = k1
K_cache[2] = k2

V_cache[0] = v0
V_cache[1] = v1
V_cache[2] = v2
```

现在来了一个新 token `t3`。

经过当前层的线性投影之后，你会得到它自己的：

```text
q3, k3, v3
```

这时对当前 token 的 attention，本质上要算的是：

```text
score_3 = [ dot(q3, k0), dot(q3, k1), dot(q3, k2), dot(q3, k3) ]
prob_3  = softmax(score_3)
out_3   = prob_3[0] * v0 + prob_3[1] * v1 + prob_3[2] * v2 + prob_3[3] * v3
```

也就是说：

- query 只有一行
- 但 key/value 不是一行
- key/value 是“历史 token + 当前 token 自己”

如果写成形状，就是：

```text
Q_decode[1, d]
K_decode[4, d]
V_decode[4, d]
```

然后做的是：

```text
[1, d] x [4, d]^T -> [1, 4]
softmax([1, 4])
[1, 4] x [4, d] -> [1, d]
```

所以 decode 时 attention 没变，只是：

- query 维度退化成了 `1`
- key/value 维度仍然是“当前上下文长度”

如果你把前面那个数字化例子代进去，就更具体：

- `hidden_dim = 2048`
- `num_heads = 16`
- `head_dim = 128`
- `past_len = 4`

那在某一个 head 上，decode 时就是：

```text
q_t[1, 128]
k_cache[5, 128]
v_cache[5, 128]
```

然后算：

```text
score_t = q_t @ k_cache^T   -> [1, 5]
prob_t  = softmax(score_t)  -> [1, 5]
out_t   = prob_t @ v_cache  -> [1, 128]
```

### KV cache 到底是什么时候 append 的

继续用上面那个 `t3` 的例子。

当当前 token `t3` 来了以后，你会先算出：

```text
q3, k3, v3
```

然后需要把：

```text
k3, v3
```

写到 cache 的新位置，比如：

```text
K_cache[3] = k3
V_cache[3] = v3
```

这就是 append。

如果把它和 attention 串起来看，逻辑上可以理解成：

1. 先得到当前 token 的 `q_t, k_t, v_t`
2. 把 `k_t, v_t` append 到 cache 尾部
3. 用 `q_t` 对“整个 cache 里的 K/V”做 attention

也就是：

```text
cache before:  [k0, k1, k2]
append k3  ->  [k0, k1, k2, k3]

cache before:  [v0, v1, v2]
append v3  ->  [v0, v1, v2, v3]

then q3 attends to all 4 rows
```

你也可以把它理解成：

- append 并不是 attention 的替代品
- append 只是把“以后要被 attention 读到的 K/V”放到正确位置

### 为什么 decode 能省很多算力

如果没有 KV cache，那么生成第 `t` 个 token 时，你得把前面所有 token 又重新算一遍 K/V。

有了 KV cache 之后，你只需要：

1. 为当前新 token 算一次新的 `q_t, k_t, v_t`
2. 把 `k_t, v_t` append 进去
3. 用 `q_t` 读取整个历史 `K/V`

这样就不必重复计算旧 token 的 K/V。

所以 KV cache 省掉的是：

- **旧 token 的 K/V 重算**

而不是：

- 当前 token 对历史信息的注意力计算

### 这和当前这个教学版 `attention.cu` 是什么关系

当前这个目录里的 `attention.cu` 仍然是：

- 最基础的完整 attention
- 用 `Q[seq_len, d] / K[seq_len, d] / V[seq_len, d]`
- 一行一行算 `QK^T -> softmax -> PV`

它没有直接实现：

- causal mask
- prefill
- decode
- KV cache

但它仍然是理解 LLM attention 的基础，因为 decode 场景里真正算的那一行，本质上仍然是：

```text
一个 query row
对一批 key rows 做点积
再 softmax
再加权求和 value
```

如果你把它和 `10_kv_cache/README.md` 连起来看，逻辑就更清楚了：

- `09_attention/`
  - 负责回答“attention 这一行到底怎么算”
- `10_kv_cache/`
  - 负责回答“历史 K/V 放在哪里，当前 token 的 K/V 怎么 append 进去”

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

这一步和 `04_softmax/README.md` 里的逻辑是同一种 reduction 模式。

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
cd experiments/cuda_kernels/09_attention
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
