# CUDA Core, Tensor Core, WMMA

这份笔记只服务你当前这条学习线：

- CUDA kernel
- GEMM
- low precision
- Tensor Core
- `ncu`

目标不是做一份完整 GPU 教程，而是把几个最容易混的边界拆开：

- `FMA` 是什么
- `CUDA core` 是什么
- `Tensor Core` 是什么
- `WMMA` 是什么
- 为什么“低精度输入”不等于“自动更快”
- 为什么 `ncu` 里看到 `FMA` 忙，不代表 Tensor Core 在工作

---

## 1. 先说最短结论

如果现在只记最重要的，先记这几句：

1. `FMA` 是 `a * b + c` 这种融合乘加操作。
2. `CUDA core` 是普通算术执行路径。
3. `Tensor Core` 是专门做矩阵乘加的小块硬件单元。
4. `WMMA` 是 CUDA 提供的一层编程接口，不是硬件本身。
5. `SM` 是比 `CUDA core / Tensor Core` 更高一层的组织单位，不要把它们放在同一层比较。
6. 低精度数据只有在真正走到 Tensor Core 或其他低精度专用执行路径时，才通常会带来很大的吞吐提升。
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

### 为什么它重要

因为大量数值计算，特别是 GEMM，本质上都在反复做这个操作。

所以当你看到：

- FMA pipeline 利用率高

通常说明：

- kernel 在大量做普通乘加

### 为什么叫 fused

因为它不是简单地：

1. 先做一次乘法
2. 再单独做一次加法

而是：

- 作为一个融合操作来执行

这通常意味着：

- 更高的吞吐
- 更少的中间舍入

### 但一定要注意

`FMA busy` 不等于：

- `Tensor Core busy`

更常见的解释往往是：

- 普通 `CUDA core` 路径上的乘加在忙

---

## 3. `CUDA core` 到底是什么

在学习阶段，你可以先把 `CUDA core` 理解成：

- GPU 上负责执行普通标量/向量算术的一般计算单元

例如这些操作，大体都会主要落在普通算术路径上：

- `float` 加减乘除
- `int` 加减乘
- 普通 FMA
- 逻辑与控制相关指令

如果你写的是很普通的 CUDA kernel，例如：

```cpp
c[idx] = a[idx] + b[idx];
```

或者：

```cpp
acc += a_val * b_val;
```

那默认应该先假设它主要是在走：

- `CUDA core` / 常规算术 pipeline

而不是 Tensor Core。

### 它不是什么

`CUDA core` 不是：

- 一个 CUDA API
- 一个软件抽象类
- 一种“只支持 FP32”的窄概念

它更像是：

- 普通通用算术路径的总称

---

## 4. `Tensor Core` 到底是什么

`Tensor Core` 是 GPU 里专门做：

- 小块矩阵乘加

的硬件单元。

最核心的味道不是：

- “它支持低精度”

而是：

- **它不是在做任意普通标量算术，而是在做特定形状、特定数据类型、特定指令格式下的矩阵乘加。**

所以它天然最适合：

- GEMM
- attention 里的 `QK^T`
- attention 里的 `PV`
- convolution 里可映射成 GEMM 的部分

### 为什么它快

因为它做的不是：

- 一次只算一个 `a * b + c`

而是：

- 一次处理一个小 tile 的矩阵乘加

所以在合适的数据类型和 tile 形状下，硬件吞吐可以比普通算术路径高很多。

---

## 5. `WMMA` 是什么

`WMMA` 的全称是：

- `Warp Matrix Multiply Accumulate`

你可以把它理解成：

- CUDA 提供的一层 warp 级矩阵乘加接口

最常见的几个构件是：

- `fragment`
- `load_matrix_sync`
- `mma_sync`
- `store_matrix_sync`

这层接口的目的就是：

- 让一个 warp 以矩阵 tile 的方式驱动 Tensor Core

### 它不是什么

`WMMA` 不是：

- 硬件本身
- 所有 Tensor Core 功能的全部形式
- 一切高性能 GEMM 的唯一写法

更准确地说：

- `Tensor Core` 是硬件
- `WMMA` 是一种较高层的 CUDA 编程接口

在更底层，还可以有：

- `mma` 相关 PTX / inline asm
- CUTLASS / cuBLAS 内部更复杂的调度和流水线组织

---

## 6. `SM`、`CUDA core`、`Tensor Core` 三者关系怎么记最稳

`SM` 本身的基础定义，你已经在 [gpu_components.md](/data/home/tianjianyang/code/aisys-map/notes/gpu_components.md) 里写过了。这里不重复讲基础，只强调最容易混淆的边界：

最容易混的，就是把这三个放在同一层。

其实它们不在同一层。

### `SM`

是更大的那一层。

### `CUDA core`

是 `SM` 里面的一类普通计算单元。

### `Tensor Core`

也是 `SM` 里面的一类计算单元。

所以更准确的关系是：

- `SM` 是容器 / 计算簇
- `CUDA core` 和 `Tensor Core` 是 `SM` 里的不同执行资源

如果用最粗糙的类比：

- `SM`：一个车间
- `CUDA core`：车间里的通用工人
- `Tensor Core`：车间里的专用矩阵加工机床

### 最常见的误解

很多人会把：

- “我在 GPU 上算”

直接等同于：

- “我用了 Tensor Core”

这不对。

你在 GPU 上算，只能说明：

- 某个 `SM` 上的某些执行资源在工作

但到底是：

- 普通 `CUDA core`

还是：

- `Tensor Core`

还要看你的 kernel 走的是哪条指令路径。

---

## 7. 三者关系怎么记最稳

你可以把它们理解成三层：

### 第 1 层：硬件执行单元

- `CUDA core`
- `Tensor Core`

### 第 2 层：编程接口/指令路径

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

## 8. 为什么“低精度输入”不等于“自动更快”

这是你刚才 GEMM 实验里已经碰到的关键坑。

比如一个 kernel 可能做了这件事：

1. 从 global memory 读 `fp16` / `int8`
2. 一进 kernel 就转成 `float`
3. shared memory 里也放 `float`
4. 最后仍然做 `float` 标量乘加

这意味着：

- 低精度只体现在存储
- 计算仍然是普通 `fp32` 路径

所以这时你能得到的主要收益是：

- 输入更省显存带宽

但你得不到这些收益：

- shared memory 同样低位宽的读写收益
- Tensor Core 的高吞吐矩阵乘加收益

所以：

**低精度存储 != 低精度计算 != Tensor Core GEMM**

这是三件不同层次的事。

---

## 9. 真正会触发 Tensor Core 的通常是什么

粗略地说，要真正让硬件走 Tensor Core，通常要满足这些条件：

1. 数据类型在支持范围内
2. tile 形状符合硬件/接口要求
3. 使用了对应的矩阵乘加指令路径

在你的当前学习语境里，可以先把典型支持记成：

- `fp16`
- `bf16`
- `tf32`
- `int8`
- `int4`（更偏实验/更底层一些）

而不是理解成：

- “我只要把数组元素类型改成半精度，GPU 就自己切到 Tensor Core”

这通常不会自动发生。

---

## 10. 为什么 GEMM 特别适合 Tensor Core

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

这就是为什么在 AI 系统里：

- GEMM 常常是最值得专门优化的 kernel

---

## 11. 为什么 softmax 和 GEMM 的气质很不一样

你现在已经做过这两类实验，所以特别适合在这里统一一下。

### softmax

更像：

- reduction
- synchronization
- block 内协作
- 数值稳定

### GEMM

更像：

- data reuse
- tiling
- multiply-accumulate density
- Tensor Core / FMA pipeline 利用

所以 shared memory 在两类 kernel 里的角色都重要，但含义不一样：

### softmax 里

shared memory 常常是：

- 做 reduction 中转站
- 做线程间交换中间结果

### GEMM 里

shared memory 更像：

- 程序员显式管理的小缓存
- 用来把 tile 搬进片上后反复复用

这也是为什么你不能把：

- “用了 shared memory”

直接等同于：

- “这是同一种优化”

---

## 12. `ncu` 里看到什么才更像 Tensor Core 在工作

一个常见误区是：

- 看见 `FMA` 很忙
- 就以为 Tensor Core 在工作

这不对。

如果你看到的是：

- 普通 FMA pipeline 高利用

那往往更说明：

- kernel 在走常规 CUDA core 路径上的乘加

而不是 Tensor Core。

更像 Tensor Core 的情况通常需要：

- kernel 写法本身就是 `WMMA` / `mma`
- profile 里能看到更接近矩阵乘加专用路径的信号
- 或者至少你从实现就知道这条路径不再是普通标量 FMA

所以一定不要只看：

- “是不是在做矩阵乘”

还要看：

- **它是用什么指令路径做矩阵乘**

---

## 13. 你现在的 GEMM 实验该怎么重新分类

你前面那版 GEMM，更准确的名字其实不是：

- “低精度 Tensor Core GEMM”

而是：

- **低精度输入存储 + shared-memory tiled + float accumulate + 普通标量乘加 GEMM**

这就是为什么你看到：

- `fp16 / bf16 / int8` 的吞吐和 `fp32` 很接近

因为主计算路径根本没有换。

这个观察不是失败，反而很有价值，因为它帮你把下面三件事拆开了：

1. 输入位宽变小
2. 数据搬运量变化
3. 计算执行单元是否变化

很多人第一次学低精度 GEMM，最容易把这三件事混在一起。

---

## 14. 现在最应该建立的最小地图

你现在最好把 GEMM 路线先记成下面这张地图：

### 路线 A：普通 CUDA core GEMM

- tiled
- shared memory reuse
- register blocking
- 普通 FMA

### 路线 B：Tensor Core GEMM

- warp-level tile
- fragment
- `load_matrix_sync`
- `mma_sync`
- `store_matrix_sync`

### 路线 C：库级高性能 GEMM

- cuBLAS
- CUTLASS
- TensorRT-LLM / TensorRT backend

如果这三条线不分开，你后面看 profile 时就会一直混：

- 到底是 tile 提升了
- 还是位宽降低了
- 还是 Tensor Core 真正在工作了

---

## 15. 结合你当前进度，下一步最自然是什么

你现在已经：

- 看过 elementwise
- 看过 softmax / online softmax
- 写过一版 tiled GEMM
- 开始关心 low precision 和 Tensor Core

所以下一步最自然的不是继续补概念，而是：

1. 把 `fp16` GEMM 改成真正的 `WMMA` / Tensor Core 版本
2. 再把 `bf16 / int8 / int4` 按是否稳定支持逐个推进
3. 最后拿 `ncu` 对比：
   - 普通 tiled GEMM
   - Tensor Core GEMM

这样你会第一次真正看见：

- “低精度存储”
vs
- “低精度计算”
vs
- “Tensor Core 计算”

这三者的差别。
