# Expert Parallel

这份文档只解释推理里的 `EP`。

## `EP` 到底在解决什么问题

`EP` 主要服务 `MoE`，也就是 mixture-of-experts 模型。

MoE 的核心不是“所有 token 都走同一个 MLP”，而是：

- 不同 token 会被 gate 路由到不同 expert

于是问题就变成：

- expert 很多，单卡未必放得下
- 就算放得下，让每张卡都放所有 expert 也很浪费

所以 `EP` 的核心思想是：

- **把不同 expert 分散到不同 rank**
- **token 根据 routing 结果，去对应的 expert 计算**

这就是它和 `TP` 的根本区别。

`TP` 是：

- 同一个 layer 的张量被切开

`EP` 是：

- 同一层里，不同 expert 被分给不同 rank

## `EP` 的主流程

一个最小 `EP` 推理路径可以拆成四步：

### 1. Routing

gate 看每个 token，决定它应该去哪个 expert。

最简单的是 top-1 routing：

- 每个 token 只去 1 个 expert

更复杂的会是 top-k：

- 每个 token 同时去多个 expert

### 2. Dispatch

按 routing 结果，把 token 分发给各个 expert。

这一步的本质是：

- token 先按 expert 分桶

在真实多卡系统里，这一步通常会变成：

- all-to-all

因为 token 需要跨设备发往对应的 expert 所在 rank。

### 3. Expert Compute

每个 expert 只处理发到自己这里的 token。

注意：

- 不同 expert 收到的 token 数量通常不一样
- 所以 `EP` 天然面临 load imbalance 问题

### 4. Gather / Combine

expert 算完以后，再把输出按原 token 顺序拼回去。

如果是 top-k routing，还要进一步做：

- 加权求和
- combine 多个 expert 输出

## 这个实验脚本在做什么

[`expert_parallel.py`](/data/home/tianjianyang/code/aisys-map/experiments/parallel/expert_parallel.py) 做的是一个最小 top-1 `EP` 实验。

它包含：

- 4 个 token
- 2 个 expert
- 每个 token 由 `expert_ids` 指定去哪个 expert
- 每个 expert 用一个独立的小 linear 处理收到的 token
- 最后再把输出按原 token 顺序放回去

这个脚本刻意不做真实分布式通信。

但它把 `EP` 最核心的语义保留住了：

- token 不是大家都走同一个算子
- token 会先路由，再 dispatch，再按 expert 分开计算

## 为什么 `EP` 比 `TP` 更“MoE 专属”

因为 `TP` 的前提是：

- 大家都在算同一个 dense layer，只是张量被切开

而 `EP` 的前提是：

- 模型内部本来就有多个 expert 分支
- 不同 token 会走不同专家

所以如果你的模型不是 MoE，那么通常不会先想到 `EP`。

## 为什么 `EP` 常常受通信影响很大

因为 `EP` 里真正贵的地方往往不是 expert 自己的 matmul，而是：

- token routing 之后的数据搬运

也就是说：

- token 在逻辑上属于当前 batch
- 但 expert 在物理上可能分散在不同设备

所以 dispatch / gather 往往会带来：

- all-to-all
- 负载不均
- 小包通信很多

这也是为什么工业系统里，`EP` 的工程复杂度通常高于 toy 代码给人的第一印象。

## 你读代码时最该问的问题

1. 为什么 `EP` 的核心不是“切张量”，而是“分 token”？
2. 为什么 `routing` 和 `dispatch` 往往比 expert 自己的计算更麻烦？
3. 如果 expert 0 收到 1000 个 token，expert 1 只收到 10 个 token，会发生什么？
4. 为什么 dense model inference 里先学 `TP`，而 MoE inference 里必须再补 `EP`？
