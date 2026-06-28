# Toy Continuous Batching Scheduler

这份 toy 代码的目的不是模拟真实 vLLM，而是先把 continuous batching 的最小骨架讲清楚。

它保留了三件最核心的事情：

- request 会在不同时间到达
- batch 不是固定不变的，而是每一轮都会重组
- prefill 和 decode 不能直接当成一回事

## 这个 toy 的调度规则

### 1. request 可以在任意 step 到达

这里不是“先收齐一批再开跑”，而是：

- step 0 到一个 request
- step 1 又到一个 request
- step 3 再到一个 request

这就是 continuous batching 和静态 batching 的根本区别：

- 静态 batching 先组好再跑
- continuous batching 是系统已经在跑，新请求还能不断插进来

### 2. 每一轮只做一种 mode

这个 toy 故意简化成：

- 这一轮要么 prefill
- 要么 decode

原因不是说工业系统一定这样，而是这样最容易看懂状态流转。

### 3. prefill 走 token budget，decode 走 request budget

prefill 阶段更像：

- 看这一轮最多还能塞多少 token

decode 阶段更像：

- 每个活跃 request 给 1 个 token 的机会

## 你运行这个 toy 时应该关注什么

### 1. waiting 和 running 怎么变化

- 新请求先进入 waiting
- prompt prefill 完成后，它才进入 running
- 进入 running 后，才会开始参与 decode

### 2. 为什么 batch 每一轮都在变

因为：

- 有新 request 到达
- 有 request 刚完成 prefill
- 有 request 已经 finish

所以 batch 不可能像训练那样“开头固定好，后面一直不变”。

### 3. 为什么 continuous batching 的重点是 scheduler

这个 toy 故意没有任何模型计算，只打印状态。

你仍然会发现最难的问题已经出现了：

- 谁先 prefill
- 谁先 decode
- 新请求什么时候插进来
- 哪些请求应该继续留在 running

这就是 continuous batching 真正的核心。
