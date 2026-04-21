# llama.cpp 主流程：从入口到每个 Step 的执行

这份文档关注 `llama.cpp` 最典型的一条主线：

- 从 `llama-cli` 入口开始
- 模型与上下文如何初始化
- prompt 如何送进模型
- 每一轮 token 生成的 step 是怎么推进的

这里先只讲最基础的单模型、单轮生成主路径，不展开 speculative decoding、server、多模态等分支。

## 1. 先用一句话概括

和 `vLLM` 相比，`llama.cpp` 的核心运行形态更像：

- 一个单体推理程序
- 持有一个 `llama_model`
- 持有一个 `llama_context`
- 循环做 `decode -> sample -> accept -> 再 decode`

所以它更接近“执行引擎 + 推理循环”，而不是“请求调度系统”。

## 2. 最常见入口：`llama-cli`

最直观的入口在：

- `tools/cli/cli.cpp`

最重要的位置是：

- `tools/cli/cli.cpp`
  `main`

这条路径可以粗略理解为：

1. 解析命令行参数
2. 根据参数加载模型
3. 创建 context
4. 初始化 sampler
5. 把 prompt token 化并送入 decode
6. 进入生成循环
7. 每轮 sample 一个 token，再把它喂回去

这就是最基本的闭环。

## 3. 一开始先做什么

### 3.1 解析参数

首先会构造：

- `common_params`

然后解析 CLI 参数。

这一步决定了后面运行时的大部分配置，例如：

- 模型路径
- context length
- batch size
- ubatch size
- GPU offload 设置
- sampling 策略

这里的关键理解是：

- `llama.cpp` 很多运行时行为不是在 engine 里动态调整
- 而是在启动时通过参数决定

### 3.2 加载模型

接着会创建：

- `llama_model`

模型加载的本质是：

- 读取 GGUF 文件
- 解析模型结构与超参数
- 加载权重
- 初始化 backend 相关资源

这里的 `model` 更像“静态部分”：

- 权重
- 模型结构
- vocab

### 3.3 创建 context

然后会创建：

- `llama_context`

这是后面真正做推理时最关键的对象之一。

可以把 `context` 理解成：

- 一次运行会话的状态容器

它里面通常持有：

- runtime buffers
- KV cache
- 当前图执行需要的中间状态
- decode 过程中的上下文信息

如果说 `model` 更像静态模型定义，那么 `context` 更像动态执行现场。

## 4. Sampler 处于什么位置

在 `llama.cpp` 里，sampling 不是 engine 外围一层独立大系统，而是比较直接地挂在生成循环旁边。

常见相关位置：

- `common/sampling.h`
- `common/sampling.cpp`

它通常会把多种采样策略串成一个 sampler chain，例如：

- temperature
- top-k
- top-p
- repetition penalty
- grammar 约束

所以一个 token 生成之后，不是直接做 greedy，而是把 logits 交给 sampler chain 处理，再选出一个 token。

## 5. Prompt 阶段在做什么

在开始生成之前，程序要先把输入 prompt 处理掉。

逻辑上通常是：

1. 文本 tokenization
2. 形成一批输入 token
3. 构造 `llama_batch`
4. 调用 `llama_decode`

这里最重要的两个对象是：

- `llama_batch`
- `llama_decode`

### 5.1 `llama_batch`

`llama_batch` 可以理解成：

- 这一次送进模型执行的 token 批次描述

它不只是 token id，还会带上：

- position
- 属于哪个 sequence
- 哪些位置需要输出 logits

所以它是一次模型执行请求的最直接载体。

### 5.2 `llama_decode`

`llama_decode(ctx, batch)` 是 `llama.cpp` 推理主循环里最关键的动作之一。

它做的事情可以理解成：

- 把当前 batch 的 token 真正送进模型
- 完成这一轮 forward
- 更新 KV cache
- 产出后续 sampling 需要的 logits / hidden state

如果把 `vLLM` 想成“每个 step 是 scheduler 驱动的一轮系统动作”，那么在 `llama.cpp` 里，每个 step 更接近“一次明确的 decode 调用”。

## 6. Prefill 在 llama.cpp 里的样子

对 prompt 阶段来说，`llama.cpp` 的主路径通常就是：

- 把 prompt token 按 batch 或 ubatch 送入
- 多次调用 `llama_decode`
- 逐步把 prompt 对应的 KV cache 填满

这里的重点不在复杂调度，而在：

- 让模型把历史上下文编码进 KV cache

也就是说，prompt 阶段最重要的产物不是“立刻输出很多 token”，而是：

- 后续 decode 所需的历史状态已经进入 context / KV cache

## 7. 真正的生成循环怎么走

完成 prompt 之后，就进入最经典的 token-by-token 循环。

可以把它概括成四步：

1. 读取当前 logits
2. sampler 选出下一个 token
3. accept 这个 token，更新采样器内部状态
4. 把这个 token 作为下一轮输入，再调用 `llama_decode`

这四步会不断重复，直到：

- 命中 EOS
- 达到最大生成长度
- 命中 stop condition

## 8. 每个 Step 具体在做什么

这是最核心的部分。

### 8.1 先从上一次 decode 的结果里拿 logits

上一轮 `llama_decode` 执行完之后，当前 step 所需的 logits 已经在 context 相关缓冲区里准备好了。

这时程序会把这些 logits 交给 sampler。

### 8.2 sampler 选 token

sampler chain 会依次应用各种规则，例如：

- temperature
- top-k
- top-p
- grammar
- repetition penalty

最后产出一个 token。

### 8.3 accept token

选出 token 后，不只是“打印出来”。

还需要：

- 把 token 加到输出序列
- 通知 sampler 这个 token 已被接受
- 更新 repetition / grammar 等状态

这一步很重要，因为下一轮采样要依赖这些状态。

### 8.4 构造只含新 token 的 batch

接下来会构造一个新的 `llama_batch`，通常只包含刚刚新采出的 token。

因为在 decode 阶段：

- 历史 token 已经体现在 KV cache 中
- 不需要把整段 prompt 再跑一遍
- 只需要把最新 token 再送进模型

### 8.5 再次调用 `llama_decode`

这一步会：

- 读取历史 KV cache
- 用新 token 做当前轮前向
- 更新 KV cache
- 为下一轮准备 logits

然后循环回到下一轮 sampling。

## 9. 为什么说 llama.cpp 更像“执行循环”而不是“调度系统”

`llama.cpp` 当然也支持 batch、并行和 server，但它最直观、最核心的心智模型仍然是：

- 我有一个 context
- 我不断往这个 context 里喂 token
- 每次 decode 后再 sample 一个 token

它没有像 `vLLM` 那样把整套系统重心放在：

- request lifecycle
- continuous batching
- 大量请求混合调度
- 复杂 KV block 分配与换入换出

所以从学习角度讲：

- `llama.cpp` 更适合看清一次推理闭环到底是怎么跑的
- `vLLM` 更适合看清很多请求如何被统一调度

## 10. `batch` 和 `ubatch` 在这里是什么感觉

在 `llama.cpp` 里，你会经常看到：

- `n_batch`
- `n_ubatch`

可以先用一个朴素理解：

- `batch` 更像逻辑上希望一起处理多少 token
- `ubatch` 更像一次底层物理执行真正切多大一块

所以对于长 prompt：

- 不一定一次全部送入模型
- 可能会被切成多个较小执行块

这和 `vLLM` 的“按 request step 调度”不是同一种抽象层次。

## 11. KV cache 在这条主线里的角色

虽然 `llama.cpp` 不像 `vLLM` 那样把 paged KV 和调度系统绑定得那么紧，但 KV cache 仍然是 decode 能高效进行的核心。

最关键的逻辑是：

- prompt 阶段写入历史 token 的 K/V
- decode 阶段只新增一个 token 的 K/V
- attention 读取历史 KV，而不重算全部前缀

所以生成循环之所以成立，本质上依赖的也是 KV cache。

## 12. 真正靠近底层执行的函数在哪里

如果继续往下读，最重要的几个名字通常是：

- `llama_decode`
- `llama_batch`
- `llama_context`
- `llama_sampler`

再往更底层走，就会进入：

- graph 构建
- backend dispatch
- ggml tensor execution

也就是说，`llama.cpp` 的上层循环并不复杂，复杂性主要往下沉到：

- `ggml`
- graph execution
- backend implementation

## 13. 建议的最短阅读顺序

如果下一步继续读源码，建议按这个顺序：

1. `tools/cli/cli.cpp`
   先看 `main` 怎样把整条流程串起来
2. `common/sampling.cpp`
   看 token 是怎么从 logits 里采出来的
3. `src/llama-context.cpp`
   看 context 维护哪些运行时状态
4. `src/llama-kv-cache.cpp`
   看 KV cache 的核心管理逻辑
5. `src/llama.cpp`
   看更核心的 decode / execution 主干

如果再往下：

6. `src/llama-graph.cpp`
7. `ggml/src/...`

这时就从“推理循环”进入“后端执行系统”了。

## 14. 一句总结

如果用一句话概括 `llama.cpp` 的主流程：

程序先加载模型并创建 context，把 prompt token 通过 `llama_batch` 送入 `llama_decode` 完成 prefill，随后进入一个不断重复的循环：从当前 logits 中采样下一个 token，接受这个 token，再把它作为新的最小 batch 送回 `llama_decode`，靠 KV cache 持续推进生成，直到结束条件满足。
