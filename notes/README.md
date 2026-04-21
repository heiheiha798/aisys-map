# Notes

`notes/` 目录用于记录一些单元知识。

这里的内容主要包括：

- 目前还不理解的概念
- 容易混淆的边界
- 之后需要回顾的基础知识

使用建议：

- 每遇到一个不太懂但经常出现的概念，就单独记下来
- 先写“它是什么”，再写“它不是什么”
- 先求把边界说清，再追求实现细节
- 尽量用自己的话记录，而不是直接摘抄资料

建议把这里的文件理解成：

- 一个文件只负责一个稳定主题
- 尽量先把“边界”写清，再讲细节
- 尽量不要把实验上下文、benchmark 结果、profile 观察混进来

也就是说：

- `notes/` 负责术语、边界、长期可复用的背景知识
- `experiments/` 负责具体实验、代码、数据和结论

## 当前边界

这个目录现在也需要和整个 repo 的收束方向保持一致。

当前更适合放进 `notes/` 的，是这些长期有复用价值的背景知识：

- GPU 组织与 memory hierarchy
- CUDA 编程对象与执行模型
- Tensor Core / WMMA / GEMM 路线
- kernel 分类与 runtime 基本概念
- weight-only quantization 的基本边界与收益口径
- attention / KV cache / FlashAttention 这类 inference 主线的桥接知识

当前不再继续作为主线展开的，是：

- 训练系统大图
- 训练并行策略细节
- HTTP server / web service 细节
- 单个 kernel 的极限优化技巧堆叠

也就是说，`notes/` 现在应该更像：

- inference systems 背景知识的稳定笔记区

而不是：

- 所有 AI systems 主题的无限制收纳区

当前几个关键文件建议按下面分工理解：

- [gpu_components.md](./gpu_components.md)
  - GPU 硬件组织和存储层次的最小地图
  - 重点是 `SM / register / shared memory / local memory / L1-TEX / L2 / VRAM`
- [cuda_tensor_core_wmma.md](./cuda_tensor_core_wmma.md)
  - `CUDA core / Tensor Core / FMA / MMA / WMMA / fragment` 这一条线
  - 重点是“到底走哪条计算路径”
- [cuda_programming_objects.md](./cuda_programming_objects.md)
  - `kernel / grid / block / warp / thread / SM` 这些编程对象
- [cuda_kernel_advanced.md](./cuda_kernel_advanced.md)
  - occupancy、register pressure、index 设计这些更偏写 kernel 的主题
- [basic_kernel_categories.md](./basic_kernel_categories.md)
  - `elementwise / reduction / GEMM` 的最基础分类直觉

当前阶段最值得保留的 `notes/` 主题，也可以压成两类：

- 物理资源和执行模型
- inference 主线里的基础桥接知识

其中 quantization 目前更适合作为：

- inference systems 里的一个常见 execution path 主题

而不是：

- 独立展开成量化训练或量化算法研究主线
