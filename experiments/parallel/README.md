# Parallel Experiments

这个目录只关注**推理优化**里最值得优先理解的两类 parallel：

- `TP`，tensor parallel
- `EP`，expert parallel

这里不打算把所有训练 parallel 都做一遍。原因很简单：

- `DP`、`FSDP`、`ZeRO` 之类主要是训练问题
- 你当前更关心的是推理时模型怎么切、算子怎么分、通信为什么产生

所以这个目录的目标是：

1. 先用最小整数例子把 `TP` 和 `EP` 的计算逻辑讲清楚
2. 先建立“张量怎么切、结果怎么合”的心智模型
3. 不急着引入真实多卡通信和复杂框架封装

## 文件说明

- `tensor_parallel.py`
  最小 `TP` 实验，包含 column parallel 和 row parallel 两种最经典形态。
- `tensor_parallel.md`
  解释 `TP` 在推理里到底解决什么问题，以及两种切分方式分别在干什么。
- `expert_parallel.py`
  最小 `EP` 实验，用 toy MoE 展示 token routing、dispatch、expert 计算、聚合输出。
- `expert_parallel.md`
  解释 `EP` 为什么是 MoE 推理的核心 parallel，以及它和 `TP` 的根本区别。

## 建议阅读顺序

1. `tensor_parallel.md`
2. `tensor_parallel.py`
3. `expert_parallel.md`
4. `expert_parallel.py`

原因是：

- `TP` 更接近 dense model inference 的主线
- `EP` 要先理解 MoE 的 token routing，再看并行才更顺

## 这两个实验故意省略了什么

为了把主问题讲清楚，这里故意不做下面这些事情：

- 不做真实多 GPU 通信
- 不做 `torch.distributed`
- 不做复杂性能 benchmark
- 不做 all-reduce / all-to-all 的真实实现
- 不做 top-k MoE 的全部变体

所以这里的重点不是“把分布式环境搭起来”，而是：

- 先理解为什么需要切
- 先理解切完以后每一块各自算什么
- 先理解最后为什么还能恢复成和全量计算一致的结果
