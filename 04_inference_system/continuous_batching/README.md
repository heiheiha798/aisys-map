# Continuous Batching

这个目录专门学习 continuous batching。

这里故意只保留一条线：

1. `toy`
   完全手搓、只为理解调度逻辑的最小实验

原因是：

- 这一阶段最重要的是先把 continuous batching 的最小状态流转看清楚
- 先把“request 如何动态进入、batch 如何每轮重组”建立成稳定心智模型
- 不急着在这里混入真实框架源码或更复杂的工程策略

## 这个目录最想回答的问题

- 什么叫 continuous batching？
- 为什么它不是“提前把一批请求堆好再一起跑”？
- 新 request 中途进入系统时，为什么 batch 会在下一轮重组？
- 为什么 prefill 和 decode 很难用完全相同的调度逻辑处理？

## 计划文件

- `toy_scheduler.py`
  最小连续批处理调度器，只看 request 如何动态进出 batch。
- `toy_scheduler.md`
  解释 toy 的状态流转。
