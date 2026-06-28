# CUDA Core, Tensor Core, WMMA

这份笔记只保留理解 `02_kernel_intro/cuda_kernels/11_gemm` 和 `03_kernel_advanced/SGEMM_CUDA` 所需的计算路径边界。

## 最短结论

1. `CUDA core` 主要执行普通标量 / 向量算术，例如 FP32 FMA。
2. `Tensor Core` 是专门做矩阵乘加 tile 的硬件单元。
3. `FMA` 是标量层面的 fused multiply-add。
4. `MMA` 是矩阵 tile 层面的 multiply-accumulate。
5. `WMMA` 是 CUDA 暴露的 warp 级 Tensor Core 编程接口之一，不是硬件本身。
6. `fragment` 是 WMMA API 里描述 tile 数据的寄存器级容器抽象。

## 三层地图

| 层 | 代表对象 | 作用 |
|---|---|---|
| 硬件执行资源 | CUDA core、Tensor Core、SM | 真正执行指令 |
| 编程接口 / 指令路径 | FMA、MMA、WMMA、PTX | 描述走哪类计算路径 |
| 库和系统实现 | cuBLAS、CUTLASS、Triton、手写 CUDA | 选择 tile、layout、pipeline 和 dispatch |

## CUDA core 路线

普通 CUDA core GEMM 通常是：

```text
load A/B -> scalar FMA loop -> accumulator -> store C
```

特点：

- 更容易理解
- 适合教学和小规模实验
- 通常难以接近现代 GPU 的矩阵乘峰值

## Tensor Core 路线

Tensor Core GEMM 通常是：

```text
load matrix tile -> MMA on Tensor Core -> accumulator fragment -> store C
```

特点：

- 面向固定 tile 形状和支持的数据类型
- 需要满足 layout、alignment、dtype 等约束
- 对 GEMM / attention matmul 这类高复用计算非常重要

## WMMA / fragment 的边界

| 名词 | 含义 | 不是 |
|---|---|---|
| `WMMA` | CUDA 提供的 warp 级矩阵乘加 API | 不是硬件，也不是所有 Tensor Core 写法的全集 |
| `fragment` | WMMA 中存放 tile 的寄存器级抽象 | 不是普通全局内存数组 |
| `mma_sync` | 触发一次 warp 级矩阵乘加操作 | 不是普通标量 for-loop |

## 为什么低精度不自动更快

低精度输入只是必要条件之一。真正能否变快，还取决于：

- 是否走 Tensor Core / MMA 路径
- 数据 layout 和 alignment 是否合适
- tile 是否足够大，能摊薄 load/store 和 launch overhead
- accumulator、转换、cast、epilogue 是否引入额外成本
- 端到端是否被 memory、scheduler 或 runtime overhead 限制

## 读 GEMM 实验时先问

1. 当前实现走 CUDA core FMA，还是 Tensor Core MMA？
2. 输入 dtype、accumulator dtype、输出 dtype 分别是什么？
3. 数据是否 tile 化并被复用？
4. bottleneck 是计算、访存、layout 转换，还是 launch/runtime overhead？
5. cuBLAS / CUTLASS / Triton / 手写 CUDA 的差异是在 API 层、tile 策略层，还是执行路径层？
