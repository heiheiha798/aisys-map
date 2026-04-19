# GEMM: BF16 CUDA Core, Tensor Core, and cuBLAS

这个目录现在只保留一条学习主线：`bf16 GEMM`。

目的很简单：

- 用一个 `CUDA core` 版本理解传统 tiled GEMM
- 用一个 `Tensor Core / WMMA` 版本理解 warp-level matrix multiply
- 用一个 `cuBLAS` 版本看成熟库在同一台机器上的基线

## 文件

- `bf16_gemm_cuda_core.cu`
  - shared-memory tiled GEMM
  - 标量 `FMA` 路径
- `bf16_gemm_tensor_core.cu`
  - `bf16` Tensor Core kernel 和公共实验代码
- `bf16_gemm_cublas.cu`
  - `cuBLAS` 封装
- `bf16_cuda_core_vs_tensor_core.md`
  - 重点代码对照说明
- `bf16_cublas_vs_ours.md`
  - 为什么 `cuBLAS` 更快
- `ncu_notes.md`
  - profile 结果

## 编译和运行

```bash
make

./bf16_gemm_cuda_core
./bf16_gemm_tensor_core
./bf16_gemm_cublas
```

## 当前实测

这台 `RTX 4090 / sm_89` 机器上的一组结果：

```text
bf16_cuda_core   avg_ms=0.0844  tflops=25.43
bf16_tensor_core avg_ms=0.0572  tflops=37.56
bf16_cublas      avg_ms=0.0181  tflops=118.93
```

直接看结论：

- `bf16_gemm_cuda_core` 是传统 tiled GEMM，对应标量 `FMA`
- `bf16_gemm_tensor_core` 已经明显更快，说明计算主路径切到了 Tensor Core
- `cuBLAS` 依然快很多，说明教学型 WMMA kernel 和成熟库还有明显差距

代码说明主要看 [bf16_cuda_core_vs_tensor_core.md](/data/home/tianjianyang/code/aisys-map/experiments/cuda_kernels/gemm/bf16_cuda_core_vs_tensor_core.md)，
库级对比主要看 [bf16_cublas_vs_ours.md](/data/home/tianjianyang/code/aisys-map/experiments/cuda_kernels/gemm/bf16_cublas_vs_ours.md)。

## NCU

为了只抓主样本 kernel，可以这样跑：

```bash
GEMM_PROFILE_ONCE=1 /usr/local/cuda-12.4/bin/ncu \
  --target-processes all \
  --set full \
  --launch-skip 1 \
  --launch-count 1 \
  --kernel-name wmma_gemm_16x16x16_kernel \
  ./bf16_gemm_tensor_core
```

更完整的结果见 [ncu_notes.md](/data/home/tianjianyang/code/aisys-map/experiments/cuda_kernels/gemm/ncu_notes.md)。
