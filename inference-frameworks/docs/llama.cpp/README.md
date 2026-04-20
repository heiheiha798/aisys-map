# llama.cpp / ggml Notes

这个目录用于记录 `llama.cpp` 中 `ggml` 路线的阅读笔记，重点关注：

- tensor / graph / operator 的执行模型
- backend abstraction
- allocator / memory planning
- CPU / CUDA 等不同 backend 的接入方式

后续会尽量只聚焦 `ggml backend`，不把注意力分散到太上层的推理逻辑。
