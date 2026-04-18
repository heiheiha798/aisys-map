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

当前比较关键的桥接笔记：

- [attention_flash_bridge.md](./attention_flash_bridge.md)
  - 把 `safe softmax`、`online softmax`、`causal attention`、`KV cache`、`FlashAttention` 接到同一条执行路径里
- [cuda_tensor_core_wmma.md](./cuda_tensor_core_wmma.md)
  - 把 `CUDA core`、`Tensor Core`、`WMMA`、低精度 GEMM 放到同一张图里
