# Elementwise Kernels

这个目录放最基础的 `elementwise` CUDA kernel 实验。

当前包含：

- `elementwise_add.cu`

## 这个实验在验证什么

这是最基础的 `elementwise` kernel。

重点不是算子本身，而是验证这些最基本的概念：

- `kernel` 是一段在 GPU 上并行执行的函数
- 每个 `thread` 通过
  - `blockIdx.x`
  - `blockDim.x`
  - `threadIdx.x`
  算出自己负责的数据位置
- 为什么通常需要 `if (idx < n)` 做边界保护
- host memory 和 device memory 怎么拷贝
- kernel launch、同步、结果校验的最小流程是什么

## 编译

```bash
make NVCC=/usr/local/cuda-12.4/bin/nvcc
```

## 运行

```bash
make run
```

## 最应该看的几行

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < n) {
  c[idx] = a[idx] + b[idx];
}
```

这就是最基础的“每个线程处理一个元素”的 CUDA 索引模式。
