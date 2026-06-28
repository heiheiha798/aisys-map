# Notes

`notes/` 放长期可复用的概念笔记：术语边界、执行模型、硬件与 runtime 背景。

不放具体实验、benchmark、profile 结论；这些放在根目录的实验分组里。这里的笔记应该能在不同实验之间反复复用。

## 文件索引

| 文件 | 负责内容 | 适合回答的问题 |
|---|---|---|
| [gpu_components.md](./gpu_components.md) | GPU 组织和存储层次 | `SM / register / shared memory / local memory / L1-TEX / L2 / VRAM` 分别是什么 |
| [cuda_programming_objects.md](./cuda_programming_objects.md) | CUDA 编程对象 | `kernel / grid / block / warp / thread / SM` 如何对应 |
| [cuda_kernel_advanced.md](./cuda_kernel_advanced.md) | 写 kernel 时的性能直觉 | index、block size、occupancy、register pressure、shared memory、divergence |
| [cuda_tensor_core_wmma.md](./cuda_tensor_core_wmma.md) | Tensor Core 路线 | `CUDA core / Tensor Core / FMA / MMA / WMMA / fragment` 的边界 |
| [basic_kernel_categories.md](./basic_kernel_categories.md) | kernel 类型分类 | elementwise、reduction、GEMM、indexed 等类型的瓶颈差异 |

## 写作边界

每篇笔记尽量只解决一个稳定主题：

- 先讲“是什么”和“边界是什么”
- 再讲它为什么影响性能或系统设计
- 少放一次性实验上下文
- 少重复 repo 级别的学习范围说明

当前最值得保留的主题可以压成两类：

- 物理资源和执行模型
- inference 主线里的基础桥接知识

如果内容已经依赖具体代码、性能数据或复现实验，优先放到对应的实验分组。
