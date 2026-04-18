# GEMM: CUDA Core vs Tensor Core

这个目录现在有两组 GEMM：

- `*_cuda_core`
  - shared-memory tiled GEMM
  - 标量 `FMA` 路径
  - 作为传统 CUDA core 对照组
- `*_tensor_core`
  - `WMMA` / Tensor Core 路径
  - 真正用低精度 Tensor Core 做矩阵乘加

另外现在补了一个第三方库基线：

- `bf16_gemm_cublas`
  - 直接调用 `cuBLAS`
  - 用来回答“成熟库做到什么水平”
  - 这也更接近 PyTorch eager 在 CUDA 上常见的 GEMM 底层路径

## 文件

### CUDA core 对照版

- `bf16_gemm_cuda_core.cu`

### Tensor Core 版

- `bf16_gemm_tensor_core.cu`
- `int8_gemm_tensor_core.cu`
- `int4_gemm_tensor_core.cu`
- `gemm_tensor_core_common.cuh`

### 第三方库基线

- `bf16_gemm_cublas.cu`

## 为什么分两条线

因为这两组 kernel 在回答两个不同问题：

- `cuda_core`
  - shared memory tiling、寄存器累加、普通 `FMA` 到底能做到什么程度
- `tensor_core`
  - 真正切到 `WMMA` / Tensor Core 之后，吞吐和瓶颈会怎么变化

最重要的一点是：

- 低精度数据类型本身，不等于低精度 Tensor Core 计算
- 真正决定计算路径的是 kernel 里到底走普通 `FMA`，还是走 `mma_sync`

## 为什么补 `cuBLAS`

这里单独补 `bf16_gemm_cublas`，不是为了把目录变成“库 benchmark 展示区”，而是因为它刚好回答了一个当前最需要面对的问题：

- 我们自己的 Tensor Core kernel 已经能跑、也能 profile
- 但它距离成熟实现到底差多少，不能只靠感觉判断

把 `cuBLAS` 放进来之后，这个目录里的三组角色就更清楚了：

- `*_cuda_core`
  - 传统 tiled GEMM 对照组
- `*_tensor_core`
  - 教学型 WMMA / Tensor Core 实现
- `bf16_gemm_cublas`
  - 当前机器上更接近工业实现上限的第三方基线

这里故意先选 `cuBLAS` 而不是直接选 `PyTorch`，因为对 CUDA GEMM 来说：

- `PyTorch` 更多是上层调用者
- 真正有代表性的底层强基线通常还是 `cuBLAS / cuBLASLt`

所以这个基线的意义不是“多测一个库”，而是：

- 明确我们现在这版 kernel 的位置
- 防止把教学型优化误判成已经接近库级实现

## 当前全部可运行的 kernel

```bash
make

./bf16_gemm_cuda_core
./bf16_gemm_tensor_core
./bf16_gemm_cublas
./int8_gemm_tensor_core
./int4_gemm_tensor_core
```

## 当前实测

下面是当前这台 `RTX 4090 / sm_89` 机器上的一组实测：

```text
bf16_gemm_cuda_core     avg_ms=0.0849  tflops=25.29
bf16_gemm_tensor_core   avg_ms=0.0571  tflops=37.63
bf16_gemm_cublas        avg_ms=0.0181  tflops=118.71

int8_gemm_tensor_core   avg_ms=0.0325  tflops=66.02

int4_gemm_tensor_core   avg_ms=0.0336  tflops=63.87
```

可以直接看出两件事：

- `bf16_gemm_cuda_core` 仍然代表传统 shared-memory tiled + 标量 `FMA` 路线
- 一旦切到 Tensor Core，`bf16` 有明显提升，`int8/int4` 提升更大

再补一个现在最重要的现实判断：

- 我们当前这个 `bf16_gemm_tensor_core` 已经比最初版本好，但它仍然只是教学型 WMMA kernel
- 和 `cuBLAS` 这种成熟库相比，`bf16` 还有大约 `3.2x` 的吞吐差距

这组差距也意味着：

- 现在继续在朴素 WMMA kernel 上做小修小补，收益空间已经开始变窄
- 如果后面要继续逼近 `cuBLAS`，方向应该转向更系统的 block tiling 和 multistage pipeline

## 精度怎么看

这里的误差要分两层看：

- `max_abs_vs_quant_ref`
  - 对比“同样量化后再做 CPU GEMM”的参考值
  - 主要用来判断 kernel 本身是不是算对了
- `max_abs_vs_fp32_ref`
  - 对比原始 `fp32` 参考
  - 这个差异里既包含 kernel 误差，也包含量化误差

所以：

- `int8/int4` 对 `fp32` 参考的误差明显更大，这是量化本身带来的，不是 kernel 算错
- `int4` 的 `max_rel_vs_quant_ref` 会比较大，主要是因为分母接近 0 的项把相对误差放大了，但它的 `max_abs_vs_quant_ref` 仍然很小

## NCU 怎么跑

为了避免 benchmark 循环把同一个 kernel 反复 profile，这里支持：

```bash
GEMM_PROFILE_ONCE=1
```

例如：

```bash
GEMM_PROFILE_ONCE=1 /usr/local/cuda-12.4/bin/ncu \
  --target-processes all \
  --set full \
  --launch-skip 1 \
  --launch-count 1 \
  --kernel-name wmma_gemm_int8_kernel \
  ./int8_gemm_tensor_core
```

更完整的结论看 [ncu_notes.md](/data/home/tianjianyang/code/aisys-map/experiments/cuda_kernels/gemm/ncu_notes.md)。

如果你想先只抓住一组最典型的代码对照，可以先看
[bf16_cuda_core_vs_tensor_core.md](/data/home/tianjianyang/code/aisys-map/experiments/cuda_kernels/gemm/bf16_cuda_core_vs_tensor_core.md)。

如果你想进一步看：

- 为什么 `cuBLAS` 会比我们当前的 WMMA kernel 快这么多
- 以及 [bf16_gemm_cublas.cu](/data/home/tianjianyang/code/aisys-map/experiments/cuda_kernels/gemm/bf16_gemm_cublas.cu) 这个包装程序到底做了什么

可以继续看
[bf16_cublas_vs_ours.md](/data/home/tianjianyang/code/aisys-map/experiments/cuda_kernels/gemm/bf16_cublas_vs_ours.md)。

## 现在最值得看的对照

如果你只想先看最关键的差别，建议按这个顺序：

1. `bf16_gemm_cuda_core` vs `bf16_gemm_tensor_core`
2. `bf16_gemm_tensor_core` vs `bf16_gemm_cublas`
3. `int8_gemm_tensor_core`
4. `int4_gemm_tensor_core`

这样最容易建立一个清晰直觉：

- `bf16_gemm_cuda_core` 代表传统 tiled GEMM
- `tensor_core` 版主要在优化 warp-level matrix multiply 的 feeding path
- `cuBLAS` 基线则代表成熟库如何把 Tensor Core 路径真正压满
