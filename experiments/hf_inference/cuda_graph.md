# CUDA Graph

这份说明对应：

- [`cuda_graph_decode.py`](cuda_graph_decode.py)

它只解释一件事：

- 和普通 eager generate 相比，CUDA Graph 到底多做了哪些事

## 先说结论

如果只看生成逻辑，CUDA Graph 没有改变：

- 还是先 prefill
- 还是后面逐 token decode
- 还是复用 KV cache

它改变的不是：

- attention 数学
- KV cache 的语义
- token 是怎么生成出来的

它改变的是：

- 同一轮 decode forward 是怎样被发到 GPU 上执行的

## eager generate 在做什么

最普通的 eager generate 可以粗暴理解成：

1. Python 发起一次 forward
2. CUDA runtime 依次 launch 这一轮 forward 里需要的 kernels
3. 这轮 forward 跑完
4. Python 再发起下一轮 forward
5. 重复很多次

如果 decode 每一轮只生成：

```text
1 token
```

那每一轮真正的计算工作量并不大。

这时就容易出现一个现象：

- 真正算 attention / matmul 的时间没那么长
- 但 Python 调度和 kernel launch overhead 的占比开始变高

## CUDA Graph 多出来的操作是什么

和 eager 相比，CUDA Graph 主要多了下面几步。

### 1. 先把输入和相关 buffer 固定下来

在 eager 里，你每轮都可以比较自由地新建 tensor、改 shape、走新的执行路径。

但 CUDA Graph 不行。

想 capture graph，通常要先把这些东西固定住：

- `input_ids` 的 shape
- `attention_mask` 的 shape
- `cache_position` 的 shape
- `past_key_values` 的物理布局

在这个实验里，就是固定成：

```text
batch_size = 1
decode_input_ids shape = [1, 1]
decode_attention_mask shape = [1, prompt_len + decode_steps]
```

所以第一件额外要做的事就是：

- 把 decode 这一步变成固定 shape 的问题

这里最容易困惑的一点是：

- 真实 decode 过程里，`attention_mask` 明明会随着生成继续变长
- 那为什么这个实验还能满足 CUDA Graph 对固定 shape 的要求

答案是：

- 这个脚本确实 benchmark 了“连续 decode 多个 token”
- 但它没有让 tensor 的 shape 跟着每一步一起变
- 它用的是“固定最大 shape，逐步修改里面的有效内容”

具体来说，这个实验先做了两件事：

1. 先把 prompt 跑完 prefill
2. 然后连续 decode `decode_steps` 个 token

假设：

```text
prompt_len = 19
decode_steps = 10
```

那这个实验会一次性预留：

- 最长总长度 `19 + 10 = 29`

所以真正固定下来的是物理 shape：

```text
decode_attention_mask shape = [1, 29]
```

后面第 1 步 decode 时，逻辑上只需要前 `20` 列有效：

```text
[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0]
```

第 2 步 decode 时，逻辑上需要前 `21` 列有效：

```text
[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0]
```

一直到最后一步，前 `29` 列全部变成有效。

于是这里的：

```text
decode_attention_mask shape = [1, prompt_len + decode_steps]
```

并不是在说：

- 每一步的逻辑有效长度不变

而是在说：

- 每一步都复用同一块固定大小的 buffer
- 每一步只是在这块 buffer 里原地改内容
- 所以 shape 不变，但有效区域在变

所以要区分两个层次：

1. 真实生成过程
   - 逻辑上长度会继续增长
   - mask 也会继续变长

2. 这个 CUDA Graph benchmark
   - 确实连续跑了多步 decode
   - 但物理 tensor shape 固定成了最大长度
   - 每一步只是修改同一块 tensor 里的值

也就是说，这个实验其实是在回答：

- “如果连续 decode 10 步，但每一步都维持固定 shape、只原地更新内容，graph replay 能不能减掉 eager overhead？”

而不是在回答：

- “一个完全动态变长的整段生成过程，能不能只用一张 graph 从头跑到尾？”

后者通常没这么直接，因为：

- 如果你真的让 tensor shape 每一步都重新分配和变化
- 那 `attention_mask` shape 会变
- `cache_position` 的相关布局也会跟着变

这些都会破坏“同一张 graph 反复 replay”的前提。

这里还要再区分三个不同层次的问题：

1. tensor 的内容能不能变
2. tensor 的 shape 能不能变
3. tensor 底层对应的那块内存地址能不能变

最准确的结论是：

- 内容可以变
- shape 不能随便变
- 底层 buffer 的地址也应该保持稳定

### 内容为什么可以变

因为如果内容不能变，CUDA Graph 就没有实际价值了。

这个实验里每一步都会改：

- `input_ids` 里的 token id
- `attention_mask` 里哪些列是 `1`
- `cache_position` 当前对应哪个逻辑位置

这些内容都在变，但 graph 仍然可以 replay。

所以：

- CUDA Graph 不是要求“数值不能变”
- 它允许你在同一块已经 capture 过的 tensor 里原地写入新内容

### shape 为什么不能随便变

因为 graph capture 记录的是一条固定的执行图。

例如 capture 时如果是：

```text
input_ids shape      = [1, 1]
attention_mask shape = [1, 29]
```

那 replay 时如果你突然改成：

```text
input_ids shape      = [2, 1]
attention_mask shape = [2, 30]
```

那整条 graph 通常就不匹配了。

因为它录下来的不只是“做一次 attention”，而是：

- 这批 kernel 怎么调
- 每个 kernel 看到多大的 tensor
- 中间张量怎样布局

所以在这个实验里才会选择：

- 一开始就预留固定最大长度
- 后面每一步只改前多少列有效

而不是：

- 每一步都重新创建一个更长的 `attention_mask`

### 地址为什么也要稳定

这是比 shape 更底层的一层要求。

CUDA Graph capture 录下来的不是抽象公式，而是一次具体的 GPU 工作提交流程。

如果 capture 时某个输入 tensor 底层地址是 A，replay 时你换成了一个新 tensor，底层地址变成 B，那么 graph 往往就不能直接安全复用。

所以常见做法就是：

- 先分配好固定输入 buffer
- capture 用这块 buffer
- replay 继续用同一块 buffer
- 只是原地改里面的内容

这也是为什么这个脚本里会显式维护：

- `static_input_ids`
- `static_attention_mask`
- `static_cache_position`

它们在整个 graph benchmark 过程中：

- 是同一批 tensor
- shape 不变
- 但内容每一步都在变

所以可以把这一点压成一句话：

- CUDA Graph 允许你改“这块内存里的值”，但不希望你改“这块内存本身的身份和结构”。

### 2. 先做 warmup

在 eager generate 里，直接跑 forward 就行。

但 CUDA Graph 往往会先做若干次 warmup forward。

原因是：

- 让相关 kernel / 内存分配 / runtime 状态先稳定下来
- 避免把一些初始化阶段的行为混进 capture

所以和 eager 相比，graph 路径通常会多一个：

- warmup 阶段

### 3. 显式创建 `torch.cuda.CUDAGraph()`

这一步是 eager 没有的。

代码里会显式写：

```python
graph = torch.cuda.CUDAGraph()
```

这表示：

- 后面要把一段 GPU 工作录下来

### 4. 显式 capture 一次固定 forward

这是和 eager 最大的区别。

eager 是：

- 每一轮都重新从 Python 发起 forward

graph 是：

- 先把“固定 shape 的 decode forward 模板”录制一次

代码里对应的是：

```python
with torch.cuda.graph(graph):
    graph_outputs = model(...)
```

这一步的含义不是：

- 把整个生成过程一次性录完

而是：

- 把“固定 shape 的单步 decode 模板”录成图
- 后面每一步都复用这张图，只修改输入 buffer 的内容

## 5. 后面不是重复 forward，而是重复 replay

这是最关键的执行差异。

eager 路径里，benchmark 时做的是：

```python
for _ in range(N):
    outputs = model(...)
```

而 CUDA Graph 路径里做的是：

```python
for _ in range(N):
    graph.replay()
```

也就是说：

- eager: 每次都重新走一轮 Python -> runtime -> kernel launch
- graph: 直接重放已经录好的 GPU 工作图

这就是它能减少 overhead 的根本原因。

## 6. 通常还要为 graph 单独准备一份固定 cache / buffer

在 eager 里，你往往只关心逻辑正确：

- 这一步把 KV 传进去
- 下一步继续复用

但在 graph 里，还要额外关心：

- capture 时访问到的那些 tensor 地址和布局要稳定

所以这个实验里专门给 eager 和 graph 各自准备了一份 `StaticCache`：

- `eager_cache`
- `graph_cache`

这样做不是因为 attention 数学不同，而是因为：

- graph replay 对底层 buffer 稳定性更敏感

## 为什么这些额外操作主要适合 decode

因为 decode 更符合 CUDA Graph 喜欢的条件：

1. 每轮 shape 更稳定
   - 常见就是 `[batch, 1]`

2. 每轮工作量更小
   - 更容易被 launch overhead 吃掉

3. 同一类 forward 会重复很多次
   - 很适合“录一次，重放很多次”

而 prefill 往往不是这样：

- shape 更大
- 计算更重
- launch overhead 占比往往没 decode 那么夸张

所以 CUDA Graph 在 LLM inference 里最自然的落点通常就是：

- decode

## 这一页最该记住什么

1. CUDA Graph 没有改变生成逻辑，变的是 GPU 工作的提交方式。
2. 和 eager 相比，它多了固定 shape、warmup、capture、replay 这些步骤。
3. `graph.replay()` 替代的是“每轮重新从 Python 发起一次 forward”。
4. 它最适合 `tokens per iteration` 很小、shape 很稳定的 decode 场景。
5. 它不是为了让 attention 数学变强，而是为了减少 eager 路径里的 launch overhead。
