# Scatter and Index-Add

这个目录放一个最基础的 `scatter / index_add` CUDA kernel 实验：

- `index_add_rows.cu`

这份说明会尽量把读者当成：

- 会写 Python
- 大概知道 `gather / scatter` 是索引相关操作
- 但还没有系统写过 CUDA kernel

也就是说，这里不会默认你已经熟悉：

- `threadIdx`
- `blockIdx`
- grid / block 的 launch 方式
- `atomicAdd`
- 为什么不规则写比不规则读更麻烦

---

## 1. 这个实验到底在干什么

这份代码实现的是一个最小行级 `index_add`：

```text
dst[ids[i], :] += src[i, :]
```

你可以先把它读成：

- `src`
  - 一张源表
- `ids`
  - 每一行源数据应该加到目标表的哪一行
- `dst`
  - 被累加的目标表

如果展开写，就是：

```text
dst_row = ids[i]
dst[dst_row, 0] += src[i, 0]
dst[dst_row, 1] += src[i, 1]
dst[dst_row, 2] += src[i, 2]
...
```

所以它和 `gather` 正好相反：

- `gather`
  - 按索引去读
- `scatter / index_add`
  - 按索引去写

---

## 2. 为什么 scatter 比 gather 更麻烦

`gather` 的主要麻烦是：

- 读的位置不规则
- 相邻线程不一定读相邻地址
- cache 命中和 coalescing 可能变差

但 `scatter` 除了这些，还会多一个真正关键的问题：

- **写冲突**

举个最小例子：

```text
ids[0] = 17
ids[1] = 17
ids[2] = 17
```

那就意味着：

```text
dst[17, :] += src[0, :]
dst[17, :] += src[1, :]
dst[17, :] += src[2, :]
```

也就是说，多行输入都要同时累加到同一行目标。

如果没有任何保护，多个线程可能会：

1. 同时读到旧值
2. 各自加上自己的增量
3. 又把结果写回去

这样就会发生经典的 race condition：

- 有些加法会丢失
- 最终结果不等于真正的总和

所以这类 kernel 的第一步几乎总是：

- 用原子操作保护写入

在 CUDA 里，对 `float` 做加法最直接的就是：

- `atomicAdd`

---

## 3. 这份代码的整体分工

这份 `index_add_rows.cu` 的设计非常直接：

- 一个 `block` 负责一行 `src`
- block 先找到这一行应该写到哪个 `dst_row`
- block 内线程沿着列维度分工
- 每个线程负责若干列
- 对目标位置做 `atomicAdd`

也就是：

```text
one block -> one src row
```

再展开一点：

```text
src_row = blockIdx.x
dst_row = ids[src_row]
for each handled col:
    atomicAdd(&dst[dst_row, col], src[src_row, col])
```

这版代码的价值不在于性能最强，而在于它把最核心的问题暴露得很清楚：

- 索引怎么映射
- 为什么是一个 block 对一行
- 为什么写回一定要用原子操作

---

## 4. 先看最重要的 kernel launch

`main` 里真正启动 kernel 的地方是：

```cpp
index_add_rows_kernel<<<src_rows, threads_per_block>>>(
    d_src, d_ids, d_dst, src_rows, dst_rows, dim);
```

你可以先把这句理解成：

- 一共启动 `src_rows` 个 block
- 每个 block 有 `threads_per_block` 个线程

代入当前参数：

- `src_rows = 4096`
- `threads_per_block = 256`

也就是：

- 一共启动 4096 个 block
- 每个 block 256 个线程

因为这份代码约定：

- 一个 block 处理一行 `src`

所以：

- `block 0` 处理 `src[0, :]`
- `block 1` 处理 `src[1, :]`
- ...
- `block 4095` 处理 `src[4095, :]`

这和 `softmax` 目录里“一行一个 block”的组织方式很像，但这里的重点从 row-wise reduction 变成了：

- 一行源数据要被写到哪一行目标
- 多个 block 可能会撞到同一个目标行

---

## 5. 从 kernel 第一行开始讲

kernel 定义是：

```cpp
__global__ void index_add_rows_kernel(const float* src, const int* ids, float* dst, int src_rows, int dst_rows, int dim)
```

它接收三块核心数据：

- `src`
  - 源矩阵，形状可以看成 `src[src_rows, dim]`
- `ids`
  - 索引数组，长度是 `src_rows`
- `dst`
  - 目标矩阵，形状可以看成 `dst[dst_rows, dim]`

以及三个尺寸参数：

- `src_rows`
- `dst_rows`
- `dim`

---

## 6. `int src_row = blockIdx.x;`

这一句的意思是：

- 当前 block 正在处理哪一行源数据

因为 launch 时就是：

```cpp
<<<src_rows, threads_per_block>>>
```

所以：

- `blockIdx.x` 的范围就是 `0 ~ src_rows - 1`

这也是为什么这里特别适合“一行一个 block”的教学写法：

- 行号可以直接用 `blockIdx.x`
- 很容易从 launch 配置对应到数据布局

---

## 7. `int tid = threadIdx.x;`

这一句表示：

- 当前线程在 block 内的编号

因为当前 block 配了 `256` 个线程，所以：

- `tid` 的范围是 `0 ~ 255`

这些线程不会去处理不同的 `src_row`，而是：

- 一起合作处理同一行 `src_row`
- 只是每个线程负责不同的列

---

## 8. 为什么这里还有越界判断

代码里先写了：

```cpp
if (src_row >= src_rows) {
  return;
}
```

在当前这份代码里，因为 block 数正好就是 `src_rows`，这句实际上不会触发。

但它仍然是一个很典型的 CUDA 防御式写法：

- 如果以后 launch 维度改了
- 或者想把 kernel 改成更通用形式
- 这句可以避免访问越界

对教学来说，你可以先把它理解成：

- 给 kernel 加一个安全护栏

---

## 9. `int dst_row = ids[src_row];` 到底在做什么

这句是整个 scatter 语义的核心：

```cpp
int dst_row = ids[src_row];
```

意思是：

- 当前这行源数据 `src[src_row, :]`
- 应该被加到目标矩阵的哪一行

也就是前面公式里的：

```text
dst[ids[i], :] += src[i, :]
```

这里把 `i` 换成了 `src_row`。

如果：

```text
ids[23] = 7
```

那这个 block 处理的就是：

```text
dst[7, :] += src[23, :]
```

---

## 10. 为什么还要检查 `dst_row`

紧接着代码写了：

```cpp
if (dst_row < 0 || dst_row >= dst_rows) {
  return;
}
```

意思是：

- 如果 `ids` 里给出的目标行号不合法
- 那这一行就不做任何写入

这也是一个非常常见的防御式处理。

因为在真实项目里，索引数组未必总是完全可信：

- 可能有负数
- 可能超过 `dst` 的行数

如果不检查，后面这行：

```cpp
dst + dst_row * dim
```

就可能指到非法地址。

所以这句本质上是在说：

- 只有合法索引才允许 scatter

---

## 11. `src_ptr` 和 `dst_ptr` 是什么

代码里接着写了：

```cpp
const float* src_ptr = src + static_cast<size_t>(src_row) * dim;
float* dst_ptr = dst + static_cast<size_t>(dst_row) * dim;
```

这是把二维数组的“某一行起始位置”算出来。

因为在内存里，`src` 和 `dst` 实际上都是连续的一维数组。

如果把：

```text
src[src_rows, dim]
```

按行展开，那第 `src_row` 行的起始地址就是：

```text
src + src_row * dim
```

同理，第 `dst_row` 行的起始地址就是：

```text
dst + dst_row * dim
```

所以：

- `src_ptr[col]`
  - 就是 `src[src_row, col]`
- `dst_ptr[col]`
  - 就是 `dst[dst_row, col]`

`static_cast<size_t>` 只是为了让大尺寸乘法更稳妥，避免中间整数乘法在类型上出问题。

---

## 12. 最核心的循环在干什么

kernel 里最重要的几行是：

```cpp
for (int col = tid; col < dim; col += blockDim.x) {
  atomicAdd(dst_ptr + col, src_ptr[col]);
}
```

这几乎就是整份 kernel 的灵魂。

先拆开看。

### 12.1 `for (int col = tid; col < dim; col += blockDim.x)`

这表示：

- 每个线程不是只负责一个列
- 而是负责：
  - `tid`
  - `tid + blockDim.x`
  - `tid + 2 * blockDim.x`
  - ...

如果当前：

- `dim = 256`
- `blockDim.x = 256`

那每个线程刚好负责一个列。

例如：

- `thread 0` 负责 `col = 0`
- `thread 1` 负责 `col = 1`
- ...
- `thread 255` 负责 `col = 255`

但如果以后 `dim` 更大，比如 `512`，那同一个线程还会继续处理：

- `tid`
- `tid + 256`

所以这种写法比“一个线程只写一个元素”更通用。

### 12.2 `atomicAdd(dst_ptr + col, src_ptr[col]);`

这句的意思是：

- 读取 `src[src_row, col]`
- 把它原子地加到 `dst[dst_row, col]`

写成更熟悉的数学形式就是：

```text
dst[dst_row, col] += src[src_row, col]
```

只不过这里的 `+=` 不是普通加法，而是：

- **原子加法**

---

## 13. 为什么这里必须是 `atomicAdd`

这是整份 scatter 教学样例最关键的地方。

假设有两个不同的 block：

- block A 在处理 `src[10, :]`
- block B 在处理 `src[37, :]`

并且它们恰好有：

```text
ids[10] = 5
ids[37] = 5
```

那它们都会往：

```text
dst[5, :]
```

上加东西。

如果不用原子操作，可能发生这种事：

1. block A 读出 `dst[5, 8] = 1.0`
2. block B 也读出 `dst[5, 8] = 1.0`
3. A 计算出 `1.0 + 0.3 = 1.3`
4. B 计算出 `1.0 + 0.7 = 1.7`
5. A 写回 `1.3`
6. B 再写回 `1.7`

最终结果变成：

```text
1.7
```

但正确结果其实应该是：

```text
2.0
```

也就是：

- A 的加法被 B 覆盖掉了

`atomicAdd` 的作用就是：

- 保证“读旧值 + 加增量 + 写回新值”这整个过程不可被别的线程打断

所以它首先解决的是：

- 正确性

但与此同时，它也会带来：

- 串行化
- contention
- 性能下降

这也是为什么 scatter 类 kernel 经常很难优化：

- 你不能先把原子操作拿掉
- 但保留原子操作又会让高冲突场景很慢

---

## 14. 这份代码里没有 shared memory reduction，为什么

这是它和 `softmax` 那份代码最不一样的地方。

`softmax` 的核心问题是：

- 同一个 block 内很多线程要一起算一行的统计量

所以它天然需要：

- shared memory
- block 内同步
- reduction pattern

而这份 `scatter / index_add` 样例的核心问题不是 reduction，而是：

- 索引决定写到哪里
- 多个 block 可能写同一个目标地址

所以它最核心的并发控制手段不是：

- `__syncthreads()`

而是：

- `atomicAdd`

这两个目录正好可以对照着看：

- `softmax`
  - 重点是“同一个 block 内怎么合作算统计量”
- `scatter`
  - 重点是“不同 block 之间怎么安全地往同一个地方写”

---

## 15. CPU reference 在做什么

GPU kernel 对应的 CPU 版本是：

```cpp
void cpu_index_add_rows(const std::vector<float>& src, const std::vector<int>& ids,
                        std::vector<float>& dst, int src_rows, int dst_rows,
                        int dim)
```

这段代码没有任何并行花样，就是最直接的三步：

1. 遍历每个 `src` 行
2. 读出 `dst_row = ids[r]`
3. 把这一行整行加到 `dst[dst_row, :]`

核心逻辑是：

```cpp
for (int r = 0; r < src_rows; ++r) {
  int dst_row = ids[r];
  ...
  for (int c = 0; c < dim; ++c) {
    dst_ptr[c] += src_ptr[c];
  }
}
```

这其实就是 scatter 数学定义最朴素的串行版本。

它的意义非常重要：

- GPU 写法再复杂
- 最终都必须和这份简单 reference 对齐

教学代码里有 CPU reference 的价值就在这里：

- 帮你把“并行写法”锚定回“最原始的正确语义”

---

## 16. `max_abs_diff` 在做什么

这段函数：

```cpp
float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b)
```

做的事情很直接：

- 遍历两个结果数组
- 计算每个位置的绝对误差
- 返回最大的那个

也就是：

```text
max_i |a[i] - b[i]|
```

最后程序就是靠它判断：

- GPU 输出和 CPU reference 是否一致

---

## 17. `main` 里的参数在表达什么

`main` 一开始写了：

```cpp
constexpr int src_rows = 4096;
constexpr int dst_rows = 512;
constexpr int dim = 256;
constexpr int threads_per_block = 256;
```

可以这样理解：

- `src_rows = 4096`
  - 一共有 4096 行输入要被 scatter
- `dst_rows = 512`
  - 目标表有 512 行
- `dim = 256`
  - 每一行向量长度是 256
- `threads_per_block = 256`
  - 一个 block 有 256 个线程

这个参数组合有一个很重要的含义：

- 源行数远大于目标行数

这意味着：

- 很多不同的 `src_row`
- 很可能会映射到同一个 `dst_row`

也就是说，这份样例天然就在制造：

- 写冲突

这正是 `atomicAdd` 发挥作用的场景。

---

## 18. host 侧数据是怎么构造的

代码里先分配了几块 host 数据：

```cpp
std::vector<float> h_src(src_numel);
std::vector<int> h_ids(src_rows);
std::vector<float> h_dst(dst_numel, 0.0f);
std::vector<float> h_ref(dst_numel, 0.0f);
```

它们分别是：

- `h_src`
  - host 侧源数据
- `h_ids`
  - host 侧索引
- `h_dst`
  - 从 GPU 拷回来的结果
- `h_ref`
  - CPU reference 结果

### 18.1 `h_src` 的构造

源码里用这段生成输入：

```cpp
float x = std::sin((r + 1) * 0.0031f) - std::cos((c + 9) * 0.015f);
float y = static_cast<float>(((r * 13 + c * 7) % 37) - 18) * 0.02f;
h_src[static_cast<size_t>(r) * dim + c] = 0.5f * x + y;
```

这不是在表达什么高深数学，而是在做两件很实际的事：

1. 生成确定性数据
2. 让不同位置上的值不要太单调

这样做的好处是：

- 每次运行都可复现
- 数据不至于全是 0 或简单常数
- 更容易发现索引或累加逻辑的 bug

### 18.2 `h_ids` 的构造

索引数组是这样生成的：

```cpp
h_ids[i] = (i * 17 + (i / 5) * 29) % dst_rows;
```

这句最关键的作用不是公式本身，而是：

- 把很多 `src_row` 映射到 `0 ~ dst_rows-1` 之间
- 而且会产生大量重复目标行

这就让程序自然覆盖了：

- 无冲突写入
- 有冲突写入

而不是只测一个“每行都写不同目标”的太简单场景。

---

## 19. GPU 内存分配和数据拷贝在做什么

后面这些语句是 CUDA 程序最常见的样板：

```cpp
cudaMalloc(&d_src, src_bytes);
cudaMalloc(&d_dst, dst_bytes);
cudaMalloc(&d_ids, ids_bytes);
```

意思是：

- 在 GPU 上分配三块内存：
  - 源矩阵
  - 目标矩阵
  - 索引数组

然后：

```cpp
cudaMemcpy(d_src, h_src.data(), src_bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_ids, h_ids.data(), ids_bytes, cudaMemcpyHostToDevice);
cudaMemset(d_dst, 0, dst_bytes);
```

这里分别是在做：

- 把 `src` 从 CPU 拷到 GPU
- 把 `ids` 从 CPU 拷到 GPU
- 把 GPU 上的 `dst` 清零

为什么 `d_dst` 要先清零？

因为这个 kernel 做的是：

```text
+=
```

如果目标数组一开始不是 0，那最后结果就会叠加在垃圾值上。

---

## 20. kernel 跑完之后做了什么

kernel launch 之后，代码写了：

```cpp
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

你可以先把它理解成：

- 检查 launch 有没有立刻报错
- 等 GPU 真正执行完成

然后再把结果拷回 host：

```cpp
cudaMemcpy(h_dst.data(), d_dst, dst_bytes, cudaMemcpyDeviceToHost);
```

接着跑 CPU reference：

```cpp
cpu_index_add_rows(h_src, h_ids, h_ref, src_rows, dst_rows, dim);
```

最后比较两者差异：

```cpp
float max_abs = max_abs_diff(h_dst, h_ref);
bool ok = max_abs < 1e-5f;
```

这就是整份教学代码的完整闭环：

1. 构造输入
2. 拷到 GPU
3. 跑 kernel
4. 拷回结果
5. 跑 CPU reference
6. 比对误差

---

## 21. 为什么这里能用很小的误差阈值

程序最后用的是：

```cpp
max_abs < 1e-5f
```

对这份代码来说，这个阈值已经相当严格。

原因是：

- 做的只是 `float32` 加法
- CPU 和 GPU 的数学路径非常接近
- 没有像 `exp`、`rsqrt` 这种更敏感的非线性

所以正常情况下，GPU 和 CPU 结果应该非常接近。

如果这里差很大，往往不是“浮点误差正常波动”，而更可能是：

- 索引错了
- 少加了某些行
- `atomicAdd` 使用不对
- 越界写了

---

## 22. 编译和运行

编译：

```bash
make
```

运行：

```bash
./index_add_rows
```

如果一切正常，你会看到类似输出：

- `index_add_rows passed`
- `src_rows`
- `dst_rows`
- `dim`
- `threads_per_block`
- `max_abs_diff`
- 几个 sample 输出

---

## 23. 这份代码最值得记住的点

1. `scatter / index_add` 的核心语义是：
   - `dst[ids[i], :] += src[i, :]`
2. 这类 kernel 的关键难点不是 reduction，而是：
   - 不规则写
   - 写冲突
   - 原子操作
3. 当前这份教学版采用：
   - 一个 block 处理一行源数据
   - block 内线程按列分工
   - 用 `atomicAdd` 保证正确性
4. `atomicAdd` 首先解决的是正确性问题，但也会带来 contention 和性能压力。
5. 这类算子非常适合作为理解推荐系统 embedding backward、图计算聚合、稀疏更新和很多不规则写模式的入口样例。
