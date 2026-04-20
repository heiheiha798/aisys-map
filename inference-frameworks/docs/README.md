# Inference Framework Notes

这个目录用于记录推理框架与相关 runtime 的阅读笔记，目前主要跟踪两条路线：

- `vLLM`
- `llama.cpp / ggml`

它们的定位并不相同，因此阅读时也不应该混在一起。

## 1. 为什么同时看这两个项目

### vLLM

`vLLM` 更接近一个完整的现代 LLM inference / serving 框架，重点在：

- scheduler、continuous batching、prefix caching
- engine / worker / model executor 这一整套运行时结构
- paged attention、KV cache、attention backend 的接入方式
- 一个完整推理系统如何把调度、模型执行、kernel 与服务接口串起来

如果关注的是：

- 一个完整推理框架的主流程怎么组织
- 什么是现代 LLM serving 的主流工程结构
- kernel / backend 在框架内部到底位于什么层次

那么 `vLLM` 是更直接的材料。

### llama.cpp / ggml

`llama.cpp` 里真正值得重点看的不是“又一个 LLM 项目”，而是它背后的 `ggml` 路线。这里更像一套独立的 tensor runtime / backend abstraction，重点在：

- tensor / graph / operator 的执行模型
- backend 抽象层怎么设计
- allocator / memory planning
- CPU / CUDA / Metal / Vulkan 等不同 backend 如何挂到同一套 runtime 上

如果关注的是：

- 什么叫一个独立的 backend 体系
- 后端不只是 kernel，还包括 graph、memory、dispatch
- `ggml` 为什么能自成一派

那么 `llama.cpp` 很有代表性。

## 2. 两条路线的核心区别

可以先用一句话区分：

- `vLLM`：更像完整的 inference / serving 框架
- `ggml`：更像自带 graph / tensor / backend 抽象的轻量 runtime

前者更偏：

- LLM serving
- scheduler / worker / engine
- prefix caching / batching / KV 管理
- 完整框架如何接入底层 backend

后者更偏：

- tensor runtime
- graph execution
- backend abstraction
- 多硬件后端统一

## 3. 建议阅读顺序

建议先看 `vLLM`，再看 `llama.cpp / ggml`。

原因很简单：

- 前面已经看过 `nano-vllm`、`nanoPD`、paged KV、prefill / decode 这些概念
- 先看 `vLLM`，可以把这些 serving 概念接到一个更完整、更主流的 inference framework 上
- 再看 `ggml`，更容易把“后端”从 serving API 提升到 runtime / graph / backend abstraction 这一层

## 4. 当前目录说明

- `inference-frameworks/vllm`
  `vLLM` upstream submodule
- `inference-frameworks/llama.cpp`
  `llama.cpp` upstream submodule，后续重点看 `ggml` 相关代码
- `inference-frameworks/docs/llama.cpp`
  `llama.cpp / ggml` 的阅读笔记

## 5. 后续文档建议

后续可以分别补：

- `vllm/README.md`
  记录 `vLLM` 最重要的主流程、engine、scheduler 与 backend 接口
- `llama.cpp/README.md`
  记录 `ggml` 的 tensor / graph / backend 关键代码
