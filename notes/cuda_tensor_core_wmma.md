# CUDA Core, Tensor Core, WMMA

这份笔记只负责一条学习线：

- `CUDA core`
- `Tensor Core`
- `FMA`
- `MMA`
- `WMMA`
- `fragment`
- 低精度 GEMM

它主要回答：

- 普通 CUDA core 乘加和 Tensor Core 矩阵乘加到底差在哪
- `WMMA` 到底是硬件还是接口
- 为什么“低精度输入”不等于“低精度计算”

它**不负责**展开这些基础硬件/存储主题：

- `SM`
- `register`
- `shared memory`
- `L1/TEX`
- `L2`
- `VRAM`

这些放到 [gpu_components.md](./gpu_components.md)。

---

## 1. 先说最短结论

如果现在只记最重要的，先记这几句：

1. `FMA` 是标量 `a * b + c` 这种融合乘加。
2. `MMA` 是矩阵块级 `A x B + C` 的乘加。
3. `CUDA core` 是更通用的算术执行路径。
4. `Tensor Core` 是专门做小块矩阵乘加的硬件执行路径。
5. `WMMA` 是 CUDA 提供的一层 warp 级矩阵乘加接口，不是硬件本身。
6. 低精度数据只有真正走到 Tensor Core 或其他低精度专用执行路径时，才通常会带来很大的吞吐提升。
7. “低精度存储”只减少数据体积，不自动等于“低精度计算”。

---

## 2. `FMA` 是什么

`FMA` 的全称是：

- `fused multiply-add`

典型形式就是：

```text
d = a * b + c
```

或者在 GEMM 里更常见的写法：

```text
acc += a_val * b_val
```

它是：

- 标量级别的乘加

所以当你在普通 tiled GEMM 里看到：

```cpp
acc[i][j] += a_frag[i] * b_frag[j];
```

本质上就是在做很多次 `FMA`。

最重要的边界是：

- `FMA busy` 不等于 `Tensor Core busy`

很多时候它只是在说明：

- 普通 `CUDA core` 路径上的乘加很忙

---

## 3. `MMA` 是什么

`MMA` 是：

- `Matrix Multiply-Accumulate`

最直接的形式就是：

```text
C = A x B + C
```

如果把 `FMA` 看成：

- 标量乘加

那么 `MMA` 就是：

- 矩阵块级乘加

在 Tensor Core / WMMA 语境里，最典型的代码形式是：

```cpp
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```

它的含义就是：

```text
c_frag = a_frag x b_frag + c_frag
```

所以一个非常稳的记法是：

- `FMA`：标量乘加
- `MMA`：矩阵块乘加

---

## 4. `CUDA core` 到底是什么

在当前学习阶段，你可以先把 `CUDA core` 理解成：

- GPU 上更通用的算术执行路径

例如这些操作，大体都会主要落在这条路径上：

- `float` 加减乘除
- `int` 加减乘
- 普通标量 `FMA`
- 逻辑与控制相关指令

如果你写的是很普通的 CUDA kernel，例如：

```cpp
c[idx] = a[idx] + b[idx];
```

或者：

```cpp
acc += a_val * b_val;
```

那默认更应该先理解成它在走：

- `CUDA core` / 常规算术 pipeline

而不是 Tensor Core。

它最重要的边界是：

- 它不是 API
- 它不是某种软件抽象
- 它是更通用的执行资源/执行路径

---

## 5. `Tensor Core` 到底是什么

`Tensor Core` 是 GPU 上专门做：

- 小块矩阵乘加

的硬件执行资源。

它最核心的味道不是：

- “支持低精度”

而是：

- **它不是在做任意普通标量算术，而是在做特定 tile 形状、特定数据类型下的矩阵乘加。**

所以它天然最适合：

- GEMM
- attention 里的矩阵乘
- 可映射成 GEMM 的卷积

最关键的边界是：

- `Tensor Core` 是硬件
- `WMMA` 只是驱动这类硬件的一层接口

---

## 6. `WMMA` 是什么

`WMMA` 的全称是：

- `Warp Matrix Multiply Accumulate`

你可以把它理解成：

- CUDA 提供的一层 warp 级矩阵乘加接口

最常见的几个构件是：

- `fragment`
- `load_matrix_sync`
- `mma_sync`
- `store_matrix_sync`

它的目的就是：

- 让一个 warp 以矩阵 tile 的方式驱动 Tensor Core

最重要的边界是：

- `WMMA` 不是硬件
- `WMMA` 不是所有 Tensor Core 写法的全集
- `WMMA` 是一种比较高层、比较直观的编程接口

更底层还可以有：

- `mma` 相关 PTX / inline asm
- CUTLASS / cuBLAS 内部更复杂的调度方式

---

## 7. `fragment` 是什么

`fragment` 可以先理解成：

- `WMMA` 接口里的一个矩阵 tile 容器

例如：

```cpp
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
```

这里的含义不是：

- “我定义了一个普通二维数组”

而是：

- 我定义了一个 warp 要处理的小块矩阵对象

在 `WMMA` 流程里，通常是：

1. `load_matrix_sync` 把内存里的 tile 装到 `fragment`
2. `mma_sync` 用这些 `fragment` 做矩阵乘加
3. `store_matrix_sync` 再把结果写回内存

所以你可以把 `fragment` 理解成：

- `WMMA` 路径里的中间 tile 表达形式

---

## 8. `SM`、`CUDA core`、`Tensor Core` 三者关系怎么记

这里只强调边界，不展开 `SM` 的基础定义。  
`SM` 本身看 [gpu_components.md](./gpu_components.md)。

最容易混的，就是把这三个放在同一层。

其实它们不在同一层。

### `SM`

- 更大一级的执行容器

### `CUDA core`

- `SM` 里的通用算术执行资源

### `Tensor Core`

- `SM` 里的专用矩阵乘加执行资源

所以更准确的关系是：

- `SM` 是容器
- `CUDA core / Tensor Core` 是 `SM` 里的不同执行资源

---

## 9. 三层地图怎么记最稳

你可以把相关对象分成三层：

### 第 1 层：硬件执行资源

- `CUDA core`
- `Tensor Core`

### 第 2 层：编程接口 / 指令路径

- 普通 CUDA C++ 标量运算
- `WMMA`
- 更底层 `mma` 指令

### 第 3 层：库与系统实现

- cuBLAS
- CUTLASS
- TensorRT-LLM
- FlashInfer

最容易混的是把第 1 层和第 2 层混掉。

一定要记住：

- `Tensor Core` 是硬件
- `WMMA` 是接口

---

## 10. 为什么“低精度输入”不等于“自动更快”

这是低精度 GEMM 最容易踩的坑。

比如一个 kernel 可能做了这件事：

1. 从 global memory 读 `fp16 / bf16 / int8`
2. 一进 kernel 就转成 `float`
3. shared memory 里也放 `float`
4. 最后仍然做普通标量乘加

这意味着：

- 低精度只体现在存储
- 计算仍然是普通 `fp32` 路径

所以你能得到的主要收益是：

- 输入更省显存带宽

但你得不到这些收益：

- Tensor Core 的高吞吐矩阵乘加
- 真正低精度 compute path 的优势

所以一定要把这三件事分开：

- 低精度存储
- 低精度计算
- Tensor Core 计算

---

## 11. 真正会触发 Tensor Core 的通常是什么

粗略地说，要真正走 Tensor Core，通常要同时满足这些条件：

1. 数据类型在支持范围内
2. tile 形状符合硬件 / 接口要求
3. 使用了对应的矩阵乘加指令路径

在你当前的学习语境里，可以先把典型支持记成：

- `fp16`
- `bf16`
- `tf32`
- `int8`
- `int4`（更偏实验 / 更底层一些）

但最重要的不是死记支持表，而是记住：

- “数据类型改成半精度”不代表 GPU 会自动切到 Tensor Core

---

## 12. 为什么 GEMM 特别适合 Tensor Core

因为 GEMM 天然就是：

```text
C = A x B
```

而 Tensor Core 天然最擅长：

- 小矩阵 tile 的乘加

所以两者非常契合。

GEMM 里你经常会看到：

- block tile
- warp tile
- fragment tile

原因就是：

- 大矩阵会被不断切成小块
- 小块由 warp 驱动
- 最终映射到 Tensor Core 或普通 FMA 路径

---

## 13. 现在最该建立的最小地图

你现在最好把 GEMM 路线先记成下面这张地图：

### 路线 A：普通 CUDA core GEMM

- tiled
- shared memory reuse
- register blocking
- 普通 `FMA`

### 路线 B：Tensor Core GEMM

- warp-level tile
- `fragment`
- `load_matrix_sync`
- `mma_sync`
- `store_matrix_sync`

### 路线 C：库级高性能 GEMM

- cuBLAS
- CUTLASS
- TensorRT-LLM / TensorRT backend
