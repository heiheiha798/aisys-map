# CUDA Kernels

这个目录放最基础的原生 CUDA C++ kernel 实验。

根目录只放通用信息。  
具体实验按类型拆到子目录里，例如：

- `elementwise/`
- `softmax/`
- `gemm/`

## 编译约定

每个子目录里可以有自己的：

- 源码
- Makefile
- README
- profiling 结果

如果 `nvcc` 不在默认 `PATH` 中，可以在各子目录显式指定：

```bash
make NVCC=/path/to/nvcc
```

当前本机可用的 `nvcc` 路径是：

```bash
/usr/local/cuda-12.4/bin/nvcc
```

例如：

```bash
make NVCC=/usr/local/cuda-12.4/bin/nvcc
```

## Profiling

本机的 `nsys` 放在：

```bash
/data/home/tianjianyang/download/nsys
```

如果后面要做 kernel profiling，可以从这里调用对应的 `nsys` 可执行文件。

本机可用的 `ncu` 路径是：

```bash
/usr/local/cuda-12.4/bin/ncu
```

---

## 当前目录建议

- `elementwise/`
  - 最基础的逐元素 kernel
- `softmax/`
  - 以 softmax 为入口理解 reduction、shared memory、同步和 block 内协作
- `gemm/`
  - matrix multiply / tiling / shared memory reuse
