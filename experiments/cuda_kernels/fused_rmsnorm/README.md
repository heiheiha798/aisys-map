# Fused Residual + RMSNorm

这个目录放一个“profile 之前”的最小完整闭环实验：

- `fused_residual_rmsnorm.cu`

这份说明会尽量把读者当成：

- 会写 Python
- 大概知道 residual connection 和 norm 是什么
- 已经看过 `layernorm/README.md` 里 row-wise reduction 的基本味道
- 但还没有把 “residual add + rmsnorm” 放进同一个 CUDA kernel 里看过

---

## 1. 这个实验到底在干什么

这份代码实现的是：

```text
fused = x + residual
mean_sq = (1 / N) * sum_i fused_i^2
inv_rms = 1 / sqrt(mean_sq + eps)
y_i = gamma_i * fused_i * inv_rms
```

如果只看一行输入，它的流程就是：

1. 先逐元素做 residual add
2. 再对整行做 RMSNorm
3. 最后乘上 `gamma`

所以这类算子可以粗糙理解成：

- 前半段是 elementwise
- 中间有一次 row-wise reduction
- 后半段又回到 elementwise

---

## 2. 为什么这里叫 fused

如果把这个过程拆开，通常会是两步：

1. 先算：

```text
s = x + residual
```

2. 再对 `s` 做 RMSNorm：

```text
y = gamma * s / sqrt(mean(s^2) + eps)
```

这样做的问题是：

- 中间值 `s` 需要先写出去
- 后面又要再读回来
- 多了一次 kernel launch

所谓 fused，就是：

- 把这两个逻辑步骤放进同一个 kernel

这份教学版虽然没有做最激进的访存优化，但已经把“逻辑上的两步”融合成了一次 launch。

---

## 3. 这份代码的整体分工

这份 `fused_residual_rmsnorm.cu` 的结构很像 `layernorm` 里的 row-wise reduction：

- 一个 `block` 负责一整行
- block 内线程先扫描这一行，累加平方和
- 用 shared memory 做 reduction
- 再把这一行写成最终输出

可以把它理解成：

```text
one block -> one row
```

这份代码和 `attention` 的不同点在于：

- `attention` 里 shared memory 既保存中间分数，也做 reduction
- 这里 shared memory 只用来做平方和 reduction

---

## 4. 先看最重要的 kernel launch

`main` 里真正启动 kernel 的地方是：

```cpp
fused_residual_rmsnorm_kernel<<<rows, threads_per_block, shared_mem_bytes>>>(
    d_x, d_residual, d_gamma, d_y, rows, cols, kEps);
```

这句可以先理解成：

- 一共启动 `rows` 个 block
- 每个 block 有 `threads_per_block` 个线程
- 每个 block 带一块动态 shared memory

代入当前参数：

- `rows = 1024`
- `cols = 256`
- `threads_per_block = 256`

也就是：

- 一共 1024 个 block
- 每个 block 256 个线程
- 每个 block 处理一行长度为 256 的向量

---

## 5. kernel 的输入和输出是什么

kernel 定义是：

```cpp
__global__ void fused_residual_rmsnorm_kernel(const float* x,
                                              const float* residual,
                                              const float* gamma, float* y,
                                              int rows, int cols, float eps)
```

你可以把它们理解成：

- `x[rows, cols]`
- `residual[rows, cols]`
- `gamma[cols]`
- `y[rows, cols]`

这里 `gamma` 只有一维，是因为：

- 它对每一行都复用同一组列方向参数

这和常见的 RMSNorm / LayerNorm 参数布局一致。

---

## 6. `int row = blockIdx.x;`

这一句表示：

- 当前 block 在处理哪一行

因为 launch 是：

```cpp
<<<rows, threads_per_block, shared_mem_bytes>>>
```

所以：

- `blockIdx.x` 的范围是 `0 ~ rows - 1`

这意味着：

- 每个 block 只关心一整行

这种 row-wise 切分，正是 norm 类 kernel 最自然的教学起点。

---

## 7. `int tid = threadIdx.x;`

这一句表示：

- 当前线程在 block 内的编号

当前有 `256` 个线程，所以：

- `tid` 的范围是 `0 ~ 255`

这些线程会一起合作处理当前这一行，不同行之间互不干扰。

---

## 8. 为什么这里要用 shared memory

代码写了：

```cpp
extern __shared__ float s_sq_sum[];
```

这表示：

- 当前 block 申请了一块动态 shared memory
- 它专门用来存每个线程的局部平方和

在 `main` 里，对应大小是：

```cpp
const size_t shared_mem_bytes = threads_per_block * sizeof(float);
```

也就是：

- 每个线程一个 `float`

这正好对应：

- 每个线程先算自己的 `local_sq_sum`
- 再写进 shared memory
- 再做 block reduction

---

## 9. 第一阶段：每个线程先算局部平方和

kernel 一开始的核心逻辑是：

```cpp
float local_sq_sum = 0.0f;
for (int col = tid; col < cols; col += blockDim.x) {
  int idx = row * cols + col;
  float fused = x[idx] + residual[idx];
  local_sq_sum += fused * fused;
}
```

这段代码做了三件事：

1. 读 `x[row, col]`
2. 读 `residual[row, col]`
3. 计算 `fused = x + residual`
4. 把 `fused^2` 累加到本线程的局部平方和里

### 9.1 为什么循环是 `col = tid; col += blockDim.x`

这表示：

- block 内线程按列方向分工

当前参数下：

- `cols = 256`
- `blockDim.x = 256`

所以每个线程刚好负责一列。

但如果以后 `cols` 更大，这种写法仍然可以工作，因为线程会继续处理：

- `tid`
- `tid + 256`
- `tid + 512`
- ...

所以这是一个很常见、很通用的列方向遍历方式。

---

## 10. 为什么这里先不立刻写输出

这一点很关键。

虽然我们已经算出了：

```text
fused = x + residual
```

但此时还不能立刻写：

```text
y = gamma * fused * inv_rms
```

因为：

- `inv_rms` 还不知道

而 `inv_rms` 依赖于：

```text
mean_sq = mean(fused^2)
```

也就是说：

- 必须先把整行平方和 reduction 做完
- 才能知道这一行最终的缩放因子

这就是 norm 类 kernel 的基本结构：

- 先收集统计量
- 再回头写最终输出

---

## 11. 第二阶段：把局部平方和写进 shared memory

代码接着写：

```cpp
s_sq_sum[tid] = local_sq_sum;
__syncthreads();
```

这表示：

- 每个线程把自己的局部平方和放进 shared memory
- 然后所有线程同步

同步的原因很简单：

- 后面 reduction 要读所有线程写进去的值
- 所以必须保证前面的写入已经完成

---

## 12. reduction 循环在干什么

接下来代码写了：

```cpp
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
  if (tid < stride) {
    s_sq_sum[tid] += s_sq_sum[tid + stride];
  }
  __syncthreads();
}
```

这是标准的 block 内 sum reduction。

它的意思是：

- 第一轮把后一半线程的结果加到前一半
- 第二轮再继续折半
- 一直到 `s_sq_sum[0]` 里留下整行总平方和

所以最后：

```cpp
float mean_sq = s_sq_sum[0] / static_cast<float>(cols);
float inv_rms = rsqrtf(mean_sq + eps);
```

就得到了这一行所需的：

- 平方均值 `mean_sq`
- 归一化因子 `inv_rms`

---

## 13. 为什么这里用 `rsqrtf`

代码里写的是：

```cpp
float inv_rms = rsqrtf(mean_sq + eps);
```

而不是：

```cpp
1.0f / sqrtf(mean_sq + eps)
```

数学上两者等价，都是：

```text
1 / sqrt(...)
```

这里用 `rsqrtf` 的原因很简单：

- 它直接表达“倒数平方根”
- 在 CUDA 代码里也很常见

对教学来说，最重要的是你要知道：

- 这里需要的是一个整行共享的缩放系数
- 后面所有列都会复用它

---

## 14. 第三阶段：回头写最终输出

后面代码写了：

```cpp
for (int col = tid; col < cols; col += blockDim.x) {
  int idx = row * cols + col;
  float fused = x[idx] + residual[idx];
  y[idx] = gamma[col] * fused * inv_rms;
}
```

这一步就是最终输出阶段。

现在：

- 整行共享的 `inv_rms` 已经有了

所以每个线程只需要：

1. 再读一次 `x[idx]`
2. 再读一次 `residual[idx]`
3. 算 `fused = x[idx] + residual[idx]`
4. 写：

```text
y[idx] = gamma[col] * fused * inv_rms
```

这里很值得注意的一点是：

- 这份教学版没有把 `fused` 中间值保存在 shared memory 或寄存器数组里跨阶段复用

而是选择：

- 先把 reduction 逻辑讲清楚
- 输出阶段再重新计算一次 `fused`

这正是“教学性优先”的取舍。

---

## 15. CPU reference 在做什么

CPU 版本是：

```cpp
void cpu_fused_residual_rmsnorm(...)
```

它按最朴素的串行方式写：

1. 对每一行先遍历一次，算整行平方和
2. 求 `mean_sq` 和 `inv_rms`
3. 再遍历一次，写最终输出

它没有任何 CUDA 细节，只有最原始的数学定义。

这很重要，因为：

- GPU kernel 的 block / thread / shared memory 再复杂
- 都必须和这份 reference 对齐

---

## 16. `main` 里的参数在表达什么

`main` 一开始写了：

```cpp
constexpr int rows = 1024;
constexpr int cols = 256;
constexpr int threads_per_block = 256;
```

可以这样理解：

- 一共有 1024 行
- 每行 256 列
- 一个 block 有 256 个线程

这组参数有一个教学上的好处：

- 一行 256 列
- 一个 block 256 个线程

所以最直观的理解就是：

- 一线程一列

这让 reduction 和回写阶段都非常好理解。

---

## 17. host 侧数据是怎么构造的

代码构造了：

- `h_x`
- `h_residual`
- `h_gamma`

并用 `sin / cos / 模运算偏移` 生成确定性输入。

目的和其他教学样例一样：

1. 每次运行可复现
2. 不同位置不会全是简单常数
3. 更容易暴露索引或 reduction bug

`h_gamma` 单独按列生成，也是为了体现：

- 它是列方向共享的缩放参数

---

## 18. GPU 内存和执行闭环在做什么

后面的 CUDA 样板流程很标准：

1. `cudaMalloc`
   - 分配 `x / residual / gamma / y`
2. `cudaMemcpy`
   - 把 host 输入拷到 GPU
3. launch kernel
4. `cudaDeviceSynchronize`
   - 等 GPU 执行完成
5. 拷回 `h_y`
6. 跑 CPU reference
7. 比较最大绝对误差

这就是整份教学代码的完整闭环。

---

## 19. 为什么误差阈值是 `2e-4`

程序最后用：

```cpp
bool ok = max_abs < 2e-4f;
```

这比纯加法 kernel 略宽松，是合理的。

因为这里涉及：

- 行内 reduction
- `rsqrt`
- 多次浮点乘法

所以相比简单 elementwise 或 copy kernel，容忍略大的数值差异是正常的。

---

## 20. 编译和运行

编译：

```bash
cd experiments/cuda_kernels/fused_rmsnorm
make
```

运行：

```bash
./fused_residual_rmsnorm
```

如果一切正常，你会看到类似输出：

- `fused_residual_rmsnorm passed`
- `rows`
- `cols`
- `threads_per_block`
- `max_abs_diff`
- 几个 sample 输出

---

## 21. 这份代码最值得记住的点

1. 这个算子本质上是：
   - 先做 `x + residual`
   - 再做 row-wise RMSNorm
2. 这类 kernel 的核心结构是：
   - 先收集整行统计量
   - 再回头写每个元素的输出
3. 一个 block 对应一行，是 norm 类教学样例里最自然的切分方式。
4. fused 的第一层价值，不是极限性能，而是先把两个逻辑步骤放进一次 kernel launch。
5. 这份代码最适合作为理解 residual add、row-wise reduction、RMSNorm 和简单 fusion 的起点。
