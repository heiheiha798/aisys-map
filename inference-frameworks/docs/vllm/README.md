# vLLM Notes

这个目录用于记录 `vLLM` 的阅读笔记，重点关注：

- engine、scheduler、worker、model executor 的主流程
- continuous batching、prefix caching、KV cache 管理
- attention backend 在框架中的接入位置
- `v1` 路径与旧路径的关系

后续会尽量优先梳理：

- 一个请求如何从入口走到模型执行
- scheduler 如何组织 prefill / decode
- `vLLM` 如何调用底层 kernel / backend
