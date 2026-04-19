# BF16 GEMM: CUDA Core vs Tensor Core

这份说明默认你已经看过前面的：

- `elementwise`
- `softmax`
- `online_softmax`

所以这里不再重复解释：

- `blockIdx / threadIdx`
- warp / block 的最基础概念
- shared memory 是什么
- 为什么 reduction 需要同步

涉及这些稳定术语时，这里尽量只给最短解释：

- GPU 存储层次与 `L1/TEX` 见 [../../../notes/gpu_components.md](../../../notes/gpu_components.md)
- `CUDA core / Tensor Core / FMA / MMA / WMMA / fragment` 见 [../../../notes/cuda_tensor_core_wmma.md](../../../notes/cuda_tensor_core_wmma.md)

这里直接进入重点：

- `bf16_gemm_cuda_core`
- `bf16_gemm_tensor_core`

到底分别在走什么计算路径，以及代码结构为什么会长得不一样。

---

## 1. 先说结论

这两个程序虽然都叫 `bf16 GEMM`，但它们比较的不是“同一种 kernel 的两个小变种”，而是两条本质不同的路线。

### `bf16_gemm_cuda_core`

- `bf16` 只用于输入存储
- 数据读出来后很快转成 `float`
- 主计算是线程自己手写的标量 `FMA`
- 核心优化手段是：
  - block tiling
  - shared memory reuse
  - register blocking

### `bf16_gemm_tensor_core`

- 输入仍然是 `bf16`
- 但乘加不再是线程自己写 `acc += a * b`
- 而是 warp 级别的：
  - `load_matrix_sync`
  - `mma_sync`
  - `store_matrix_sync`
- 也就是说，真正切到了 Tensor Core 路线
- 这里的 `mma_sync`、`MMA`、`WMMA` 都按 [../../../notes/cuda_tensor_core_wmma.md](../../../notes/cuda_tensor_core_wmma.md) 里的定义理解

最重要的一句就是：

- **`bf16` 这个数据类型本身，不自动等于 Tensor Core。**

决定你到底在跑哪条路的，是：

- 你是不是还在手写标量 `FMA`
- 还是已经在调用 `mma_sync`

---

## 2. 两个入口文件怎么分工

[bf16_gemm_cuda_core.cu](/data/home/tianjianyang/code/aisys-map/experiments/cuda_kernels/11_gemm/bf16_gemm_cuda_core.cu)

[bf16_gemm_tensor_core.cu](/data/home/tianjianyang/code/aisys-map/experiments/cuda_kernels/11_gemm/bf16_gemm_tensor_core.cu)

```cpp
int main() {
  return gemm_tc::run_tensor_core_experiment("bf16_tensor_core");
}
```

目录里现在的分工很明确：

- `bf16_gemm_cuda_core.cu`
  - 保留一版传统 tiled GEMM，用来对照 CUDA core 路线
- `bf16_gemm_tensor_core.cu`
  - 保留一版最小可读的 `bf16` WMMA kernel，用来对照 Tensor Core 路线

这里不再展开其他数据类型，只围绕这两条 `bf16` 路线看“计算主路径”。

---

## 3. CUDA core 版在做什么

### 3.1 block tile 结构

CUDA core 版的核心常量是：

```cpp
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 16;
constexpr int THREADS_X = 16;
constexpr int THREADS_Y = 16;
constexpr int TM = 4;
constexpr int TN = 4;
```

如果只看这 7 个名字，不够直观。  
更好的读法是：**逐个看它们到底在定义什么。**

### `BLOCK_M`

表示：

- 一个 block 在输出矩阵 `C` 的 **M 方向** 一次负责多少行

这里是：

```cpp
BLOCK_M = 64
```

所以当前 block 最终会负责输出里的 `64` 行。

它直接出现在 shared memory tile 里：

```cpp
__shared__ float a_tile[BLOCK_M][BLOCK_K];
```

也出现在 grid 划分里：

```cpp
dim3 grid((n + BLOCK_N - 1) / BLOCK_N, (m + BLOCK_M - 1) / BLOCK_M);
```

也就是说：

- grid 在 `y` 方向每前进一步
- 就是把 block 负责的输出区域往下挪 `64` 行

### `BLOCK_N`

表示：

- 一个 block 在输出矩阵 `C` 的 **N 方向** 一次负责多少列

这里是：

```cpp
BLOCK_N = 64
```

所以当前 block 最终会负责输出里的 `64` 列。

它直接出现在：

```cpp
__shared__ float b_tile[BLOCK_K][BLOCK_N];
dim3 grid((n + BLOCK_N - 1) / BLOCK_N, (m + BLOCK_M - 1) / BLOCK_M);
```

也就是说：

- grid 在 `x` 方向每前进一步
- 就是把 block 负责的输出区域往右挪 `64` 列

### `BLOCK_K`

表示：

- K 维一次推进多深

这里是：

```cpp
BLOCK_K = 16
```

矩阵乘法的 K 维不是一次全算完，而是分段：

```cpp
for (int k0 = 0; k0 < k; k0 += BLOCK_K) { ... }
```

所以每一轮 `k0`：

- 只取 `A` 的一块 `BLOCK_M x BLOCK_K`
- 只取 `B` 的一块 `BLOCK_K x BLOCK_N`

在当前参数下就是：

- `A` 取 `64 x 16`
- `B` 取 `16 x 64`

### `THREADS_X`

表示：

- 一个 block 在 `x` 方向放多少个线程

这里是：

```cpp
THREADS_X = 16
```

所以 launch 时：

```cpp
dim3 block(THREADS_X, THREADS_Y);
```

会得到：

- 每行线程数是 `16`

它后面直接决定：

- `threadIdx.x` 的范围
- 以及一个线程在输出 tile 里负责哪几列

### `THREADS_Y`

表示：

- 一个 block 在 `y` 方向放多少个线程

这里是：

```cpp
THREADS_Y = 16
```

所以一个 block 总线程数是：

```text
THREADS_X * THREADS_Y = 16 * 16 = 256
```

它后面直接决定：

- `threadIdx.y` 的范围
- 以及一个线程在输出 tile 里负责哪几行

### `TM`

表示：

- 每个线程在 **M 方向** 负责多少个输出元素

这里是：

```cpp
TM = 4
```

最直接的证据是：

```cpp
float acc[TM][TN];
```

以及：

```cpp
const int row_base = block_row + threadIdx.y * TM;
```

这说明：

- `threadIdx.y` 决定这个线程从哪一行开始
- 但这个线程不是只算 1 行
- 而是连续算 `TM = 4` 行

### `TN`

表示：

- 每个线程在 **N 方向** 负责多少个输出元素

这里是：

```cpp
TN = 4
```

对应代码是：

```cpp
float acc[TM][TN];
const int col_base = block_col + threadIdx.x * TN;
```

这说明：

- `threadIdx.x` 决定这个线程从哪一列开始
- 但这个线程不是只算 1 列
- 而是连续算 `TN = 4` 列

---

把这几个定义连起来，才会得到后面的整体结构：

- 一个 block 负责一个 `64 x 64` 的输出 tile
- 一个 block 有 `16 x 16 = 256` 个线程
- 每个线程负责一个 `4 x 4` 的输出子块

也就是：

- block 级别管大 tile
- thread 级别管小 tile
- K 维按 `16` 深度推进

这是很标准的 shared-memory tiled GEMM 写法。

如果你想检查这些数字是不是彼此匹配，可以看下面这组关系：

```text
THREADS_Y * TM = BLOCK_M
THREADS_X * TN = BLOCK_N
```

代进去就是：

```text
16 * 4 = 64
16 * 4 = 64
```

也就是说：

- block 在纵向有 `16` 个 thread-row
- 每个 thread-row 负责 `4` 行输出
- 所以总共覆盖 `64` 行

横向同理：

- block 在横向有 `16` 个 thread-col
- 每个 thread-col 负责 `4` 列输出
- 所以总共覆盖 `64` 列

因此这个 block 最终正好覆盖一个 `64 x 64` 的输出 tile。

`BLOCK_K = 16` 则表示 K 维不是一次性算完，而是每轮只推进 `16`。  
所以每一轮 `k0`，当前 block 只会取：

- `A` 的一个 `64 x 16` tile
- `B` 的一个 `16 x 64` tile

再把这两块累加到当前的 `64 x 64` 输出 tile 上。

如果写成一个很粗略的心智模型，就是：

```text
一个 block 负责:   64 x 64 的 C tile
每轮 K 方向取:     64 x 16 的 A tile
                  16 x 64 的 B tile
线程布局:          16 x 16
每线程输出:        4 x 4
```

后面你看到：

- shared memory tile
- `acc[TM][TN]`
- `row_base / col_base`

其实都是在落实这组分块关系。

### 3.2 数据路径

最关键的 shared memory 声明是：

```cpp
__shared__ float a_tile[BLOCK_M][BLOCK_K];
__shared__ float b_tile[BLOCK_K][BLOCK_N];
```

注意这里的类型是 `float`，不是 `__nv_bfloat16`。

而 global load 的位置是：

```cpp
val = storage_to_float(a[global_row * k + global_col]) * scale_a;
val = storage_to_float(b[global_row * n + global_col]) * scale_b;
```

对 `bf16` 来说：

- `scale_a = 1`
- `scale_b = 1`

所以实际含义就是：

- global memory 里存 `bf16`
- load 之后立刻转成 `float`
- shared memory 里存的是 `float`
- 寄存器里的累加器也是 `float`

因此这版的真实数据路径是：

```text
global(bf16) -> shared(float) -> registers(float) -> scalar FMA -> global(float)
```

这就是为什么它虽然名字是 `bf16_gemm_cuda_core`，但本质上并不是“bf16 Tensor Core compute”。

它更准确的定义是：

- `bf16` 输入存储
- `float` 累加
- CUDA core 标量乘加

### 3.3 计算核心

真正决定这版身份的是这一句：

```cpp
acc[i][j] += a_frag[i] * b_frag[j];
```

完整上下文是：

```cpp
for (int kk = 0; kk < BLOCK_K; ++kk) {
  ...
  acc[i][j] += a_frag[i] * b_frag[j];
}
```

这说明：

- 乘加是线程自己展开做的
- 本质还是标量 `FMA`
- shared memory 只是帮它把数据喂得更近一些

也就是说，这版优化的是：

- **传统 tiled GEMM**

而不是：

- Tensor Core MMA

### 3.4 同步为什么还在

这一版每轮 `k0` 都要：

```cpp
__syncthreads();
...
__syncthreads();
```

原因很直接：

- block 内线程共同填 shared memory tile
- 再共同消费 shared memory tile
- 下一轮还要覆盖写新的 tile

所以这个版本的典型气质就是：

- shared memory 很重
- block 内同步很重

---

## 4. Tensor Core 版在做什么

### 4.1 先看 fragment 类型

Tensor Core 版最关键的定义是：

```cpp
using FragA =
    wmma::fragment<wmma::matrix_a, 16, 16, 16, AType, wmma::row_major>;
using FragB =
    wmma::fragment<wmma::matrix_b, 16, 16, 16, BType, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
```

这几句基本就把这版 kernel 的身份说完了：

- tile 形状是 `16 x 16 x 16`
- `A` 按 row-major 读
- `B` 按 col-major 读
- 累加器是 `float`

所以对 `bf16` 来说，这一版是：

- `bf16` 输入
- `fp32` 累加
- WMMA/Tensor Core 计算

### 4.2 block / warp 分工

launch 方式是：

```cpp
dim3 block(32, WMMA_BLOCK_WARPS);
dim3 grid((n + WMMA_N - 1) / WMMA_N,
          (m + WMMA_BLOCK_WARPS_M * WMMA_M - 1) /
              (WMMA_BLOCK_WARPS_M * WMMA_M));
```

解释成结构就是：

- `block.x = 32`，刚好一个 warp
- `block.y = 8`
- 所以一个 block 有 `8` 个 warp
- 每个 warp 负责一个 `16 x 16` 输出 tile

代码里对应的是：

```cpp
int warp_slot = threadIdx.y;
int row = (blockIdx.y * WMMA_BLOCK_WARPS_M + warp_slot) * WMMA_M;
int col = blockIdx.x * WMMA_N;
```

也就是：

- `threadIdx.y` 在这版里本质上就是 warp slot

跟 CUDA core 版相比，这里最重要的变化不是 tile 数字，而是：

- **最小有效协作单位从 block 内线程拼 tile，变成了 warp 直接对应一个 MMA tile**

### 4.3 为什么 B 要转成 col-major

host 侧有一步：

```cpp
auto b_col_major = transpose_to_col_major_storage(b_storage, check_k, check_n);
```

原因不是“喜欢转置”，而是因为 kernel 里声明的是：

```cpp
wmma::fragment<wmma::matrix_b, ..., wmma::col_major>
```

所以这版多了一个 Tensor Core 特有的要求：

- 数据布局必须符合 WMMA 的读取方式

这也是 Tensor Core 路线和普通 tiled GEMM 很不一样的一点：

- 它对内存布局约束更强

### 4.4 计算核心

真正的核心循环是：

```cpp
for (int k0 = 0; k0 < k; k0 += 16) {
  FragA a_frag;
  FragB b_frag;

  for (int idx = linear_tid; idx < WMMA_N * WMMA_K;
       idx += blockDim.x * blockDim.y) {
    ...
    b_tile[tile_col * WMMA_B_LD + tile_row] = ...;
  }
  __syncthreads();

  if (warp_active) {
    wmma::load_matrix_sync(a_frag, a + row * k + k0, k);
    wmma::load_matrix_sync(b_frag, b_tile, WMMA_B_LD);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  __syncthreads();
}
```

和 CUDA core 版对照看，最关键的是：

- CUDA core 版核心动作是：
  `acc += a * b`
- Tensor Core 版核心动作是：
  `mma_sync(c_frag, a_frag, b_frag, c_frag)`

也就是说：

- 前者是 thread-level scalar FMA
- 后者是 warp-level matrix MMA

这就是两条路线最本质的分界线。

### 4.5 为什么这版仍然用了 shared memory

当前这版 Tensor Core 示例并不是纯粹的：

```text
global -> fragment -> mma_sync
```

它已经做了一个很小但很关键的 staging：

```cpp
__shared__ __nv_bfloat16 b_tile[WMMA_N * WMMA_B_LD];
```

也就是：

- `A` 仍然直接从 global memory 进 fragment
- `B` 先由整个 block 搬进 shared memory
- 再由每个 warp 从 `b_tile` 喂给 `wmma::load_matrix_sync`

所以这版更准确的数据路径是：

```text
A: global(bf16) -> fragment
B: global(bf16) -> shared(bf16) -> fragment
C: accumulator(float) -> global(float)
```

这正好对应当前代码的设计重点：

- 保持 WMMA API 结构足够直接
- 只给 `B` 加最小必要的 shared-memory staging
- 让 block 内多个 warp 复用同一个 `B` tile

---

## 5. 两边并排看，最重要的差别是什么

### 5.1 协作单位不同

CUDA core 版：

- 主要围绕 block 级 shared memory tile 来组织

Tensor Core 版：

- 主要围绕 warp 级 MMA tile 来组织

### 5.2 中间数据停留的位置不同

CUDA core 版：

- shared memory + registers

Tensor Core 版：

- fragment + accumulator fragment

### 5.3 核心计算指令不同

CUDA core 版：

- 标量 `FMA`

Tensor Core 版：

- `mma_sync`

### 5.4 布局约束不同

CUDA core 版：

- 自己读元素，布局自由度更大

Tensor Core 版：

- 要考虑 `row_major / col_major`
- 要考虑 tile shape
- 要考虑 WMMA 接口要求

---

## 6. 为什么 `bf16_gemm_cuda_core` 和真正低精度 compute 不是一回事

这部分其实就是你前面那个问题的核心。

`bf16_gemm_cuda_core` 的问题不在于：

- 输入精度不够低

而在于：

- 它真正的乘加仍然是普通 CUDA core 上的标量 `FMA`

也就是说，它优化的是：

- 存储带宽
- tile reuse
- shared memory 数据复用

但没有改变“谁在做乘加”这件事。

而 `bf16_gemm_tensor_core` 改变的正是这件事：

- 从线程手工乘加
- 切到 Tensor Core 做 warp-level matrix MMA

所以如果你只问：

- “是不是 bf16”

这是不够的。

更关键的问题一定是：

- **到底是 CUDA core FMA 路线，还是 Tensor Core MMA 路线？**

---

## 7. 结合当前实测怎么理解

当前这台机器上的结果是：

```text
bf16_gemm_cuda_core   avg_ms=0.0844  tflops=25.43
bf16_gemm_tensor_core avg_ms=0.0572  tflops=37.56
```

所以 first-order 结论很简单：

- Tensor Core 版明显更快

但更有意思的是 profile 画像。

### `bf16_gemm_cuda_core`

`ncu` 里大致是：

- `Memory Throughput ≈ 50%`
- `Compute (SM) Throughput ≈ 52%`

这说明：

- 它还是一版比较传统的 balanced tiled GEMM

### `bf16_gemm_tensor_core`

`ncu` 里大致是：

- `Memory Throughput ≈ 72.50%`
- `Compute (SM) Throughput ≈ 31.04%`
- `L1/TEX Cache Throughput ≈ 63.25%`

这里的 `L1/TEX` 按 [../../../notes/gpu_components.md](../../../notes/gpu_components.md) 里的定义理解。  
在当前这个 `bf16_gemm_tensor_core` 例子里，它主要是在提示：

- Tensor Core 不是没在工作
- 更大的问题仍然是数据 feeding path，尤其是靠近 `SM` 的 load/cache 路径压力还比较高

这说明：

- Tensor Core 路线已经切对了
- 但当前瓶颈已经主要不是乘加单元
- 而是 feeding path

也就是：

- global load
- layout
- coalescing
- L1/TEX

所以这个例子最值得记住的不是“Tensor Core 更快”这么简单，而是：

- **CUDA core 版的主要问题是你还在做标量 FMA**
- **Tensor Core 版的主要问题是你已经在做 MMA 了，但喂数还不够顺**

---

## 8. 你现在应该怎么读代码

如果你现在回去重新读这两份公共头文件，最值得抓的只有这几处。

### CUDA core 版只看这 4 个点

1. 常量：
   `BLOCK_M / BLOCK_N / BLOCK_K / TM / TN`
2. shared memory tile：
   `a_tile`、`b_tile`
3. 寄存器累加器：
   `float acc[TM][TN]`
4. 核心乘加：
   `acc[i][j] += a_frag[i] * b_frag[j]`

### Tensor Core 版只看这 5 个点

1. `wmma::fragment`
2. `block(32, 8)` 对应 8 个 warp
3. `transpose_to_col_major_storage`
4. `b_tile` 这块 shared memory staging
5. `mma_sync`

如果这 9 个点你都能在代码里重新定位出来，这个 `bf16` 对照就已经看懂了。

---

## 9. 最该记住的结论

1. `bf16` 只是数据格式，不是计算路径。
2. `bf16_gemm_cuda_core` 的本质是：
   - `bf16` 输入存储
   - `float` shared memory / register accumulation
   - CUDA core 标量 `FMA`
3. `bf16_gemm_tensor_core` 的本质是：
   - `bf16` 输入
   - `float` 累加
   - warp-level `WMMA`
   - Tensor Core `mma_sync`
4. 这两个 kernel 的真正区别，不是“都叫 bf16”，而是：
   - 一个在优化传统 tiled GEMM
   - 一个在优化 Tensor Core 路线下的 feeding path
5. 如果后面要继续优化 `bf16_gemm_tensor_core`，重点已经不是“再多写几层乘加循环”，而是：
   - 更好的 staging
   - 更好的 global load coalescing
   - 更好的 tile / layout 组织
