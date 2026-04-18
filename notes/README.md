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

`review_notes.md` 可以作为这类内容的总入口，后续也可以继续拆成更细的单文件。

当前比较关键的桥接笔记：

- [attention_flash_bridge.md](/data/home/tianjianyang/code/aisys-map/notes/attention_flash_bridge.md)
  - 把 `safe softmax`、`online softmax`、`causal attention`、`KV cache`、`FlashAttention` 接到同一条执行路径里
- [cuda_tensor_core_wmma.md](/data/home/tianjianyang/code/aisys-map/notes/cuda_tensor_core_wmma.md)
  - 把 `CUDA core`、`Tensor Core`、`WMMA`、低精度 GEMM 和 `ncu` 观察放到同一张图里
