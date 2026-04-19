# KV Cache Append and Update

这个目录放一个最基础的 `KV cache append / update` CUDA kernel 实验：

- `kv_cache_append_update.cu`

这份说明会尽量把读者当成：

- 会写 Python
- 知道 decoder attention 里 KV cache 大概是什么
- 但还没有从 CUDA 索引映射的角度仔细看过“append / update 到底怎么写”

---

## 1. 这个实验到底在干什么

这份代码做的事情可以概括成：

```text
cache[head, slot, :] = src[token_id, head, :]
```

只不过这里对 `K` 和 `V` 各做一遍：

```text
K_cache[head, slot, :] = K_src[token_id, head, :]
V_cache[head, slot, :] = V_src[token_id, head, :]
```

如果这个 `slot` 是新的尾部位置，那就是 append。  
如果这个 `slot` 是已经存在的位置，那就是 update。

所以这个教学版的核心观点非常简单：

- append 和 update 在最小实现里，本质上就是同一种“按索引拷贝一行向量”

---

## 2. 为什么这个问题适合单独做一个样例

很多人第一次接触 KV cache，会先把注意力放在：

- paged cache
- 连续 batch
- decode 调度
- 多层、多头布局

这些都没错，但如果一开始就把所有复杂因素叠加起来，最基础的问题反而容易看不清：

- 当前 token 的 K/V 到底写到哪里
- head、slot、token_id 三个索引怎么对应到连续内存
- 为什么 update 本质上就是覆盖写

这份最小样例的价值就在于：

- 先把“写进去”这件事本身讲清楚

---

## 3. 先说清楚：这里的数据布局是什么

源码里你可以把数据理解成：

```text
K_src[num_tokens, num_heads, head_dim]
V_src[num_tokens, num_heads, head_dim]
K_cache[num_heads, max_seq_len, head_dim]
V_cache[num_heads, max_seq_len, head_dim]
```

这里：

- `num_tokens`
  - 可供读取的 token K/V 数量
- `num_heads`
  - attention head 数量
- `head_dim`
  - 每个 head 的向量长度
- `max_seq_len`
  - cache 里每个 head 最多保存多少个时间步

所以 source 和 cache 的主要差别是：

- `src`
  - 以 `token_id` 为第一维
- `cache`
  - 以 `slot` 为第二维

这个样例做的事，就是在这两种布局之间做定点拷贝。

---

## 4. 操作数组 `KvOp` 在表达什么

代码里定义了：

```cpp
struct KvOp {
  int head;
  int slot;
  int token_id;
};
```

这表示一次写操作只需要三个索引：

- `head`
  - 写哪个 attention head
- `slot`
  - 写到 cache 的哪个位置
- `token_id`
  - 数据从哪个 token 的 K/V 读取

所以一个操作可以读成：

```text
把 token_id 对应 head 的那一段向量，
写到 cache 的 (head, slot) 位置
```

这个设计很适合教学，因为它把问题压缩成：

- 一个操作 = 一次明确的 K/V 写入

---

## 5. 这份代码的整体分工

kernel 的组织方式非常直接：

- 一个 `block` 处理一个 `KvOp`
- block 先读 `{head, slot, token_id}`
- block 内线程沿 `head_dim` 方向分工
- 把对应的 K/V 向量写进 cache

可以把它理解成：

```text
one block -> one write op
```

这个样例没有：

- reduction
- shared memory tree reduction
- 原子操作

因为它的重点不是“多个线程一起算统计量”，而是：

- 根据索引把一段向量拷到正确位置

---

## 6. 先看最重要的 kernel launch

`main` 里 kernel 被调用了两次。

第一次：

```cpp
kv_cache_append_update_kernel<<<static_cast<int>(h_append_ops.size()),
                                threads_per_block>>>(
    ... d_append_ops ...);
```

第二次：

```cpp
kv_cache_append_update_kernel<<<static_cast<int>(h_update_ops.size()),
                                threads_per_block>>>(
    ... d_update_ops ...);
```

你可以先把它理解成：

- phase 1
  - 执行 append 操作
- phase 2
  - 执行 update 操作

当前参数里：

- `append_ops.size() = 6`
- `update_ops.size() = 2`
- `threads_per_block = 128`

也就是：

- append 阶段启动 6 个 block
- update 阶段启动 2 个 block
- 每个 block 128 个线程

---

## 7. 为什么 append 和 update 要分两次 launch

这是这份样例里非常关键的设计点。

程序里：

- append 会把 `token 0/1/2` 写入 `slot 0/1/2`
- update 会把 `token 5` 再写到 `slot 1`

如果把这些操作全部混在同一次 launch 里，问题在于：

- block 之间没有全局执行顺序保证

那结果就可能依赖调度顺序。

但如果分成两次 launch，语义就很稳定：

1. 先完成 append
2. 再完成 update

所以这份教学版是故意把“时序语义”讲清楚，而不是把所有操作塞成一个更花哨的 batch。

---

## 8. kernel 的输入和输出是什么

kernel 定义是：

```cpp
__global__ void kv_cache_append_update_kernel(const float* k_src,
                                              const float* v_src,
                                              const KvOp* ops,
                                              float* k_cache,
                                              float* v_cache,
                                              int num_ops,
                                              int num_heads,
                                              int max_seq_len,
                                              int head_dim)
```

这里的核心输入有三块：

- `k_src`
- `v_src`
- `ops`

核心输出有两块：

- `k_cache`
- `v_cache`

以及尺寸信息：

- `num_ops`
- `num_heads`
- `max_seq_len`
- `head_dim`

---

## 9. `int op_idx = blockIdx.x;`

这一句表示：

- 当前 block 正在处理第几个写操作

因为每个 block 对应一个 `KvOp`，所以：

- `blockIdx.x`
  - 就直接等于操作编号

这就是这份 kernel 最核心的映射关系：

```text
block -> op
```

---

## 10. `int tid = threadIdx.x;`

这一句表示：

- 当前线程在 block 内的编号

当前有 `128` 个线程，所以：

- `tid` 的范围是 `0 ~ 127`

这些线程不会负责不同的 `op_idx`，而是：

- 一起合作处理同一个写操作
- 只是在 `head_dim` 上分工

---

## 11. 为什么要检查 `op_idx` 和 `op` 的合法性

代码先写了：

```cpp
if (op_idx >= num_ops) {
  return;
}
```

然后又写：

```cpp
KvOp op = ops[op_idx];
if (op.head < 0 || op.head >= num_heads || op.slot < 0 ||
    op.slot >= max_seq_len) {
  return;
}
```

这两层检查的作用分别是：

1. 防止操作编号越界
2. 防止 `head` 或 `slot` 非法

这属于典型的防御式写法。

对教学来说，你可以把它理解成：

- 只有合法操作才允许写 cache

---

## 12. `src_offset` 和 `cache_offset` 是怎么来的

这段是整份 kernel 最值得真正看懂的地方：

```cpp
const size_t src_offset =
    static_cast<size_t>(op.token_id) * num_heads * head_dim +
    static_cast<size_t>(op.head) * head_dim;
const size_t cache_offset =
    static_cast<size_t>(op.head) * max_seq_len * head_dim +
    static_cast<size_t>(op.slot) * head_dim;
```

### 12.1 `src_offset`

source 张量可以看成：

```text
src[token_id, head, d]
```

按连续一维内存展开后：

- 先跨过前面所有 token
- 再跨过当前 token 里前面的所有 head
- 然后到达当前 head 的起始位置

所以：

```text
src_offset = token_id * num_heads * head_dim + head * head_dim
```

### 12.2 `cache_offset`

cache 张量可以看成：

```text
cache[head, slot, d]
```

按一维展开后：

- 先跨过前面所有 head
- 再跨过当前 head 里的前面所有 slot
- 然后到达当前 slot 的起始位置

所以：

```text
cache_offset = head * max_seq_len * head_dim + slot * head_dim
```

这就是这份样例最核心的布局映射。

---

## 13. 指针 `k_src_ptr / v_src_ptr / k_cache_ptr / v_cache_ptr` 是什么

代码接着写：

```cpp
const float* k_src_ptr = k_src + src_offset;
const float* v_src_ptr = v_src + src_offset;
float* k_cache_ptr = k_cache + cache_offset;
float* v_cache_ptr = v_cache + cache_offset;
```

这表示：

- 当前这次操作真正要读的 source 向量起点
- 当前这次操作真正要写的 cache 向量起点

所以：

- `k_src_ptr[d]`
  - 就是 `K_src[token_id, head, d]`
- `k_cache_ptr[d]`
  - 就是 `K_cache[head, slot, d]`

`V` 也是同理。

---

## 14. 最核心的循环在干什么

kernel 里最核心的几行是：

```cpp
for (int d = tid; d < head_dim; d += blockDim.x) {
  k_cache_ptr[d] = k_src_ptr[d];
  v_cache_ptr[d] = v_src_ptr[d];
}
```

这表示：

- block 内线程按 `head_dim` 分摊工作
- 每个线程负责若干个维度 `d`
- 把对应的 K 和 V 直接写到 cache

当前参数下：

- `head_dim = 16`
- `blockDim.x = 128`

所以实际上只有前 16 个线程各自负责一个维度。

这并不是为了效率最优，而是为了：

- 代码极其直白
- 你可以一眼看出“线程在拷贝哪个向量分量”

数学上它就是：

```text
K_cache[head, slot, d] = K_src[token_id, head, d]
V_cache[head, slot, d] = V_src[token_id, head, d]
```

---

## 15. 为什么这里不需要 `atomicAdd`

和 `scatter` 不同，这里不是：

```text
+=
```

而是：

```text
=
```

也就是说，这里做的是覆盖写，不是累加。

在当前这份教学样例里：

- 每个 `(head, slot)` 在同一次 launch 中不会被多个操作同时写

所以不需要原子操作。

真正需要注意的不是原子加，而是：

- 如果存在写同一位置的多个操作
- 那它们的执行顺序要通过多次 launch 或更高层逻辑来保证

这也是为什么 append / update 分成两个 phase。

---

## 16. CPU reference 在做什么

CPU 版本是：

```cpp
void cpu_kv_cache_append_update(...)
```

它就是按最朴素的串行方式：

1. 遍历每个 `KvOp`
2. 算出 source 起点和 cache 起点
3. 把长度为 `head_dim` 的向量复制过去

它本质上就是最原始的数学定义：

```text
cache[head, slot, :] = src[token_id, head, :]
```

所以 GPU kernel 再怎么并行，也必须和这份简单 reference 对齐。

---

## 17. `fill_source` 在做什么

代码里还有一个辅助函数：

```cpp
void fill_source(...)
```

它的作用是：

- 用确定性规则生成 `K_src` 和 `V_src`

用 `sin / cos` 和简单线性组合生成数据，目的还是一样：

1. 每次运行结果可复现
2. 数据不全是简单常数
3. 更容易发现索引映射错误

---

## 18. `print_sample` 在做什么

代码里还写了：

```cpp
void print_sample(...)
```

它只是一个帮助理解输出的辅助函数：

- 给定 `head` 和 `slot`
- 打印该位置上 K/V 的前几个分量

这对教学很有用，因为它让你能把：

- “逻辑上的写入”

和

- “实际 cache 某个 slice 的内容”

直接对上。

---

## 19. `main` 里的参数在表达什么

`main` 里有：

```cpp
constexpr int num_tokens = 8;
constexpr int num_heads = 2;
constexpr int max_seq_len = 6;
constexpr int head_dim = 16;
constexpr int threads_per_block = 128;
```

可以这样理解：

- source 里有 8 个 token 的 K/V
- 每个 token 有 2 个 head
- cache 每个 head 最多放 6 个时间步
- 每个 head 的向量长度是 16

这组参数故意选得很小，方便你直接在脑子里跟踪：

- token 0/1/2/5
- head 0/1
- slot 0/1/2

---

## 20. append 和 update 操作具体是什么

程序里构造了两组操作：

```cpp
std::vector<KvOp> h_append_ops = {
    {0, 0, 0}, {1, 0, 0}, {0, 1, 1}, {1, 1, 1}, {0, 2, 2}, {1, 2, 2},
};
std::vector<KvOp> h_update_ops = {
    {0, 1, 5},
    {1, 1, 5},
};
```

你可以读成：

- append 阶段：
  - token 0 写入 slot 0 的两个 head
  - token 1 写入 slot 1 的两个 head
  - token 2 写入 slot 2 的两个 head
- update 阶段：
  - token 5 覆盖 slot 1 的两个 head

这正好展示了：

- append
  - 往新位置写
- update
  - 往已有位置覆盖

---

## 21. GPU 执行闭环在做什么

后面程序按标准 CUDA 流程：

1. 分配 GPU 内存
2. 拷贝 source、ops 和初始 cache
3. launch append kernel
4. 同步
5. launch update kernel
6. 同步
7. 把 cache 拷回 host
8. 与 CPU reference 比较

这就是完整的 correctness 闭环。

---

## 22. 为什么误差阈值几乎是零

程序最后检查：

```cpp
bool ok = k_max_abs < 1e-6f && v_max_abs < 1e-6f;
```

这很严格，但这里是合理的，因为这份 kernel 做的只是：

- 读取
- 直接赋值写回

没有：

- `exp`
- reduction
- 原子冲突
- 复杂浮点路径

所以 GPU 和 CPU 结果应该几乎完全一致。

---

## 23. 编译和运行

编译：

```bash
cd experiments/cuda_kernels/kv_cache
make
```

运行：

```bash
./kv_cache_append_update
```

如果一切正常，你会看到类似输出：

- `kv_cache_append_update passed`
- `num_tokens`
- `num_heads`
- `max_seq_len`
- `head_dim`
- `append_ops`
- `update_ops`
- `k_max_abs_diff`
- `v_max_abs_diff`

并打印一个 cache slice 的 sample。

---

## 24. 这份代码最值得记住的点

1. append 和 update 在最小实现里，本质上都是：
   - `cache[head, slot, :] = src[token_id, head, :]`
2. 这份 kernel 的核心不是 reduction，而是：
   - 索引映射
   - 数据布局
   - 覆盖写语义
3. 一个 block 对应一个写操作，是这类教学样例里最直接的切分方式。
4. append / update 分两次 launch，是为了把时序语义固定住。
5. 这份代码非常适合作为理解 KV cache 基础写入逻辑、索引布局和 decode 状态更新的起点。
