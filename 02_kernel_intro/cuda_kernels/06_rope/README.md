# RoPE Forward

这个目录放一个最基础的 `RoPE (Rotary Position Embedding)` CUDA kernel 实验：

- `rope_forward.cu`

这份说明会尽量把读者当成：

- 会写 Python
- 知道 RoPE 大概是“按位置做旋转”
- 但还没有把这种二维旋转和 CUDA 索引映射一一对应起来看过

---

## 1. 这个实验到底在干什么

RoPE 的最基础形式是：

- 把向量按相邻两维组成一个 pair
- 对每个 pair 按 token 位置做二维旋转

如果 `head_dim = d` 且 `d` 为偶数，那么：

```text
(x_0, x_1), (x_2, x_3), (x_4, x_5), ...
```

对于第 `i` 个 pair，在位置 `m` 上会用到角度：

```text
theta_i = m / base^(2i / d)
```

然后做：

```text
y_{2i}   = x_{2i} * cos(theta_i) - x_{2i+1} * sin(theta_i)
y_{2i+1} = x_{2i} * sin(theta_i) + x_{2i+1} * cos(theta_i)
```

所以这份代码本质上就是：

- 读一个 `(token, head)` 向量
- 对里面每个二维 pair 做旋转
- 写回输出

---

## 2. 为什么 RoPE 很适合作为教学 kernel

RoPE 有几个非常适合入门 CUDA 的特点：

1. 数学上非常局部
   - 每个 pair 只依赖自己这两个分量
2. 没有 reduction
3. 没有原子操作
4. 没有跨 token / head 的依赖

这意味着它特别适合拿来讲：

- 数据布局怎么映射到 block / thread
- 为什么“一个 block 处理一整行”很自然
- pair 级别的索引怎么算

---

## 3. 这份代码的数据布局是什么

源码里的输入输出可以看成：

```text
x[seq_len, num_heads, head_dim]
y[seq_len, num_heads, head_dim]
```

这里：

- `seq_len`
  - 序列长度
- `num_heads`
  - attention head 数量
- `head_dim`
  - 每个 head 的维度

对固定的 `(token_idx, head_idx)`，会有一整行长度为 `head_dim` 的向量。

这份教学版就是按这整行来组织工作的。

---

## 4. 这份代码的整体分工

kernel 的组织方式很直接：

- 一个 `block` 处理一个 `(token, head)` 行
- block 内线程按 pair 维度分工
- 每个线程负责若干个二维 pair

可以把它理解成：

```text
one block -> one (token, head) row
```

这正好和 RoPE 的数学结构对齐：

- 旋转发生在一行内部
- 不同行之间相互独立

---

## 5. 先看最重要的 kernel launch

`main` 里启动 kernel 的地方是：

```cpp
rope_forward_kernel<<<total_rows, threads_per_block>>>(d_x, d_y, seq_len,
                                                       num_heads, head_dim,
                                                       kBase);
```

这里：

```cpp
const int total_rows = seq_len * num_heads;
```

代入当前参数：

- `seq_len = 128`
- `num_heads = 8`
- `total_rows = 1024`
- `threads_per_block = 128`

也就是：

- 启动 1024 个 block
- 每个 block 128 个线程

因为一个 block 对应一个 `(token, head)` 行，所以：

- `block 0`
  - 第 0 个 `(token, head)` 行
- `block 1`
  - 第 1 个 `(token, head)` 行
- ...

---

## 6. kernel 的输入和输出是什么

kernel 定义是：

```cpp
__global__ void rope_forward_kernel(const float* x, float* y, int seq_len,
                                    int num_heads, int head_dim, float base)
```

你可以把它读成：

- 输入 `x`
  - 原始向量
- 输出 `y`
  - 旋转后的向量
- `base`
  - RoPE 的频率基数，当前是 `10000`

---

## 7. `int row = blockIdx.x;`

这一句表示：

- 当前 block 正在处理第几个 `(token, head)` 行

因为所有 `(token, head)` 行都被线性展开了，所以：

- `row`
  - 只是展开后的一维行号

接下来代码还会把它再拆回：

- `token_idx`
- `head_idx`

---

## 8. `int total_rows = seq_len * num_heads;`

这句的作用是：

- 明确总共有多少个 `(token, head)` 行

因为对于每个 token，都有 `num_heads` 个 head。

所以：

```text
总行数 = seq_len * num_heads
```

这个量也正是 launch grid 的 `x` 维大小。

---

## 9. 为什么这里先判断 `row >= total_rows`

代码写了：

```cpp
if (row >= total_rows) {
  return;
}
```

当前这份代码里，由于 grid 大小正好等于 `total_rows`，这句实际上不会触发。

但它仍然是标准的防御式写法：

- 如果以后 launch 配置变化
- kernel 也不会越界访问

---

## 10. `token_idx = row / num_heads` 是怎么来的

代码写了：

```cpp
int token_idx = row / num_heads;
```

这是把线性展开后的 `row`，重新映射回：

- 第几个 token

因为每个 token 对应 `num_heads` 行，所以：

- 连续 `num_heads` 个 row
  - 属于同一个 token

比如：

- `row 0 ~ 7`
  - token 0
- `row 8 ~ 15`
  - token 1

当前代码里没有显式写出 `head_idx`，因为后续其实不需要它单独参与公式。

---

## 11. `pair_dim = head_dim / 2` 在表达什么

代码写了：

```cpp
int pair_dim = head_dim / 2;
```

这是因为：

- 每两个相邻维度组成一个 pair

如果：

- `head_dim = 64`

那就有：

- `32` 个 pair

也就是说，后面的线程分工不是按单个标量维度，而是按：

- pair 编号

---

## 12. `src` 和 `dst` 指针在做什么

代码写了：

```cpp
const float* src = x + static_cast<size_t>(row) * head_dim;
float* dst = y + static_cast<size_t>(row) * head_dim;
```

这表示：

- 当前 `(token, head)` 行在一维连续内存中的起点

所以：

- `src[even_col]`
  - 当前行某个偶数列分量
- `src[odd_col]`
  - 当前行对应的奇数列分量

这一步本质上是在说：

- 先把当前 block 要处理的那一整行定位出来

---

## 13. 最核心的循环在干什么

kernel 里最重要的几行是：

```cpp
for (int pair_idx = tid; pair_idx < pair_dim; pair_idx += blockDim.x) {
  int even_col = 2 * pair_idx;
  int odd_col = even_col + 1;

  float x0 = src[even_col];
  float x1 = src[odd_col];

  float exponent = (2.0f * static_cast<float>(pair_idx)) /
                   static_cast<float>(head_dim);
  float theta = static_cast<float>(token_idx) / powf(base, exponent);
  float cos_theta = cosf(theta);
  float sin_theta = sinf(theta);

  dst[even_col] = x0 * cos_theta - x1 * sin_theta;
  dst[odd_col] = x0 * sin_theta + x1 * cos_theta;
}
```

这段代码就是整个 RoPE kernel 的灵魂。

---

## 14. 为什么循环是按 `pair_idx` 走

这表示：

- 一个线程处理若干个 pair

而不是：

- 一个线程只处理一个标量维度

这样写非常自然，因为 RoPE 的基本操作单元本来就是：

- 一个二维 pair

当前参数下：

- `pair_dim = 32`
- `blockDim.x = 128`

所以只有前 32 个线程会各自处理一个 pair。

这不是极限效率配置，但它非常直白。

---

## 15. `even_col` 和 `odd_col` 是什么

代码里：

```cpp
int even_col = 2 * pair_idx;
int odd_col = even_col + 1;
```

这表示：

- 第 `pair_idx` 个 pair 对应的两个实际维度位置

例如：

- `pair_idx = 0`
  - `(0, 1)`
- `pair_idx = 1`
  - `(2, 3)`
- `pair_idx = 2`
  - `(4, 5)`

这正好把 RoPE 的数学写法和实际内存索引连上了。

---

## 16. `theta` 是怎么计算出来的

代码里：

```cpp
float exponent = (2.0f * static_cast<float>(pair_idx)) /
                 static_cast<float>(head_dim);
float theta = static_cast<float>(token_idx) / powf(base, exponent);
```

它直接对应公式：

```text
theta_i = m / base^(2i / d)
```

其中：

- `m`
  - 就是 `token_idx`
- `i`
  - 就是 `pair_idx`
- `d`
  - 就是 `head_dim`

这份教学版的好处就在于：

- 数学公式几乎是原封不动地落到了代码里

---

## 17. 为什么这里分别算 `cos_theta` 和 `sin_theta`

代码写了：

```cpp
float cos_theta = cosf(theta);
float sin_theta = sinf(theta);
```

这就是二维旋转矩阵里的那两个量。

后面对应的写回：

```cpp
dst[even_col] = x0 * cos_theta - x1 * sin_theta;
dst[odd_col] = x0 * sin_theta + x1 * cos_theta;
```

也正是标准二维旋转：

```text
[ cos  -sin ]
[ sin   cos ]
```

所以如果你想真正看懂 RoPE kernel，最应该盯住的就是这四行。

---

## 18. 为什么这里不需要 shared memory 或同步

这和 `softmax`、`attention`、`fused_rmsnorm` 很不一样。

这里每个 pair 的计算：

- 只读当前 pair 的两个输入
- 只写当前 pair 的两个输出

不同线程之间没有共享统计量，也没有写冲突。

所以这里不需要：

- shared memory reduction
- `__syncthreads()`
- 原子操作

这也是为什么 RoPE 是一个非常适合作为“索引映射型 kernel”入门样例的原因。

---

## 19. CPU reference 在做什么

CPU 版本是：

```cpp
void cpu_rope_forward(...)
```

它按最朴素的方式：

1. 遍历每个 `token_idx`
2. 遍历每个 `head_idx`
3. 遍历每个 `pair_idx`
4. 按同样公式做旋转

所以它和 GPU 版本的差别只在于：

- 一个是串行
- 一个是并行

数学上完全一致。

---

## 20. `main` 里的参数在表达什么

`main` 里有：

```cpp
constexpr int seq_len = 128;
constexpr int num_heads = 8;
constexpr int head_dim = 64;
constexpr int threads_per_block = 128;
```

可以这样理解：

- 一共有 128 个 token
- 每个 token 有 8 个 head
- 每个 head 维度是 64
- 每个 block 有 128 个线程

还写了：

```cpp
static_assert(head_dim % 2 == 0, "RoPE requires even head_dim.");
```

这非常重要，因为：

- RoPE 必须按两维一组旋转
- 所以 `head_dim` 必须是偶数

---

## 21. host 侧输入是怎么构造的

代码构造了 `h_x`，并用：

- `sin`
- `cos`
- 模运算偏移

生成确定性输入。

这样做的目的是：

1. 每次运行都可复现
2. 每个 token / head / dim 的值不完全相同
3. 更容易检查旋转是否真的发生了

---

## 22. 为什么误差阈值是 `2e-5`

程序最后用：

```cpp
bool ok = max_abs < 2e-5f;
```

这里的误差来源主要是：

- `pow`
- `sin`
- `cos`
- 浮点乘加

因为 CPU 和 GPU 的数学库实现细节不完全一样，所以允许一个很小但非零的阈值是合理的。

---

## 23. 编译和运行

编译：

```bash
cd 02_kernel_intro/cuda_kernels/06_rope
make
```

运行：

```bash
./rope_forward
```

如果一切正常，你会看到类似输出：

- `rope_forward passed`
- `seq_len`
- `num_heads`
- `head_dim`
- `threads_per_block`
- `max_abs_diff`
- 几个 sample 输出

---

## 24. 这份代码最值得记住的点

1. RoPE 的基本操作单元不是单个标量，而是：
   - 一个二维 pair
2. 这份 kernel 最自然的切分方式是：
   - 一个 block 处理一个 `(token, head)` 行
   - 线程按 pair 分工
3. 这类 kernel 的重点不是 reduction，而是：
   - 索引映射
   - pair 级别数学
   - 正确写回
4. 这份代码几乎把 RoPE 公式原样翻译成了 CUDA 写法。
5. 它非常适合作为理解 LLM 中高频小算子、位置编码和局部变换型 kernel 的入门样例。
