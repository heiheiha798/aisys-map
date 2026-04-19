# HF Inference

这个目录放的是：

- 基于 Hugging Face `transformers`
- 基于真实模型
- 面向推理路径的最小实验

它的定位不是：

- 解释最基础的 attention 数学
- 解释所有 attention 变体
- 作为 vanilla transformer 入门

这些更基础的内容应该看：

- [`experiments/vanilla_transformer/`](/data/home/tianjianyang/code/aisys-map/experiments/vanilla_transformer)
- [`experiments/attention_variants/`](/data/home/tianjianyang/code/aisys-map/experiments/attention_variants)

这个目录更像“真实 backend 对照物”。

也就是：

- 前面已经把概念讲清楚了
- 现在回到真实模型，看这些概念在 HF 路径里怎样出现

## 这个目录会放什么

后续这里会统一放这些真实推理实验：

1. `single_request_decode.py`
   - 最小真实模型实验
   - 看 prefill 怎么建立首份 KV cache
   - 看 decode 怎么复用 KV cache 并逐步 append

2. `batching.py`
   - 看多个 request 一起进入 HF backend 时，shape 和 padding 怎样变化

3. `chunked_prefill.py`
   - 看长 prompt 被切块后，prefill 路径怎样变化

4. `cuda_graph_decode.py`
   - 看固定 `bsz=1` 的 decode 单步 forward 怎样做 CUDA Graph capture
   - 看它和普通 eager decode 的吞吐差异
   - 具体解释看 [`cuda_graph.md`](/data/home/tianjianyang/code/aisys-map/experiments/hf_inference/cuda_graph.md)

5. 其他和真实推理路径直接相关的小实验
   - 比如 attention mask
   - 比如 batch 内长度差异
   - 比如 prefix 部分和 decode 部分的 shape 演化

## 当前目录的原则

这里统一坚持三条原则：

1. 用真实模型
   - 当前默认模型是 `/data/pretrained_models/Qwen3-0.6B`

2. 用真实 backend
   - 当前用 Hugging Face `transformers`

3. 重路径，不重优化
   - 目标是把真实推理路径跑实
   - 不是在这个目录里做 kernel 级性能优化

## 当前已有实验

### `single_request_decode.py`

这个脚本会做一件事：

- 用真实模型显式跑一次 prefill
- 再显式跑若干步逐 token decode

它会打印：

- prompt 文本
- prompt token 数
- Qwen3-0.6B 的 attention 真实参数
- prefill 的输入 shape
- prefill 时的 `attention_mask`
- prefill 后第一层 KV cache 的 shape
- 每一步 decode 的输入 shape
- 每一步 decode 时 K/V cache 怎样增长
- 每一步 decode 时 `attention_mask` 怎样增长

### 什么叫 eager generate

这里的 `eager generate` 可以先理解成：

- 不做 graph capture
- 不用专门的 serving engine scheduler
- 直接在 Python 里按步调用模型 forward
- 一步一步把生成过程跑完

所以它强调的不是：

- 某种新的 attention 公式

而是：

- 一种执行方式

最常见的 Hugging Face 写法是：

```python
model.generate(...)
```

但如果把这层高层接口拆开，本质上还是：

1. 先做一次 prefill
2. 建立第一份 `past_key_values`
3. 然后进入逐 token decode loop
4. 每一步都复用旧 KV，只输入当前新 token
5. 一直跑到达到停止条件

所以：

- `model.generate(...)`
  - 是封装好的 eager generate
- [`single_request_decode.py`](/data/home/tianjianyang/code/aisys-map/experiments/hf_inference/single_request_decode.py)
  - 是手动展开后的 eager generate

这个区分很重要，因为后面如果去看推理引擎，你会看到另一类东西：

- runtime 会接管 request 管理、batching、KV 管理、调度

那时关注点就不再只是：

- “Python 里怎么一轮一轮 forward”

而会变成：

- “系统怎么组织很多 request 一起生成”

### `batching.py`

这个脚本会做两件事：

- 先把两条不同长度的 prompt padding 成一个 batch，一起做 prefill
- 再对这个 batch 一起做一步 decode

它主要用来看：

- batched `input_ids` 和 `attention_mask` 长什么样
- padding 是怎样进入 HF backend 的
- batched prefill 后第一层 KV cache shape 怎样变化
- batched decode 时，输入 shape 怎样从 `[batch, seq_len]` 变成 `[batch, 1]`
- left padding 下，新 decode token 会怎样统一接到 batch 的最右侧

### `cuda_graph_decode.py`

这个脚本只看一个很具体的问题：

- 当 decode 每一轮只生成很少 token 时
- 普通 eager forward 的 Python + launch overhead 会不会显得更重

这里固定：

- `batch_size = 1`
- 每轮 decode 只输入 `1` 个新 token
- 也就是最典型的 `tokens per iteration = 1`

然后对比两条路径：

1. 普通 eager decode
   - 每一轮都从 Python 发起一次 forward

2. CUDA Graph replay
   - 先把固定 shape 的 decode forward capture 成 graph
   - 后面每一步原地更新输入内容，再重复 replay

所以这个实验的重点不是：

- 更复杂的 KV 管理
- 更复杂的调度
- 完整 serving 场景

而是很简单的一点：

- decode 每轮工作量很小的时候，overhead 占比会更高
- CUDA Graph 的价值主要就出现在这种“每步很短、shape 很稳定”的路径上

这也是为什么它更常用于：

- decode

而不是：

- 大块 prefill

因为 prefill 往往本身计算更重，launch overhead 占比没那么高。

## 为什么 batching 很关键

`batching` 不是一个可有可无的小技巧，而是 LLM inference 里最核心的吞吐手段之一。

原因很直接：

- 单条请求单独跑，GPU 很容易吃不满
- 很多 launch / runtime 开销会被重复支付
- 同样一块 GPU，如果能把多条请求合并成一个更大的规则 batch，通常吞吐会更高

所以服务系统几乎都会想办法把多个 request 拼在一起跑。

如果没有 batching，系统会更像：

- 请求 A 来了，单独跑一次
- 请求 B 来了，再单独跑一次
- 请求 C 来了，再单独跑一次

这样虽然简单，但代价是：

- GPU 上的小计算很多
- launch 次数更多
- 很难形成高吞吐

而 batching 的目标就是：

- 把多个 request 合成一个更大的规则张量
- 尽量一次 forward 处理更多样本

## 不同 seq_len 为什么不能直接 batch

问题的关键在于：

- 不同请求的 `seq_len` 往往不同

比如两条请求：

- request A 长度是 `8`
- request B 长度是 `14`

那它们原始输入其实是：

```text
A: input_ids shape = [1, 8]
B: input_ids shape = [1, 14]
```

这两个 tensor 不能直接 stack 成一个标准二维 batch，因为第二维长度不一样。

而 GPU backend 通常最适合处理的是规则矩阵，比如：

```text
[batch, seq_len]
```

或者再往下进入 attention 后的规则 tile。

所以如果想 batch，就必须先把不同长度补成一样长。

## 为什么要 padding

padding 的作用只有一个：

- 把不同长度的请求补成相同长度，变成一个规则 batch

还是上面的例子。

如果 batch 里最长的是 `14`，那长度为 `8` 的 request A 必须先补到 `14`。

在 decoder-only inference 里，更适合解释 decode 的方式通常是：

- `left padding`

也就是把短请求补在左边，而不是右边。

于是 batch 更像：

```text
A: [PAD PAD PAD PAD PAD PAD t0 t1 t2 t3 t4 t5 t6 t7]
B: [u0  u1  u2  u3  u4  u5  u6 u7 u8 u9 u10 u11 u12 u13]
```

这样它们才能变成一个真正可以送进模型的 batch：

```text
input_ids shape = [2, 14]
```

所以：

- `padding` 不是语义内容
- 它只是为了把不规则请求变成规则张量

## attention_mask 到底在做什么

padding 只是把 shape 补齐，但补出来的位置不能真的当作有效 token 参与模型计算。

这时就必须引入：

- `attention_mask`

对应上面的例子，mask 会像：

```text
A: [0 0 0 0 0 0 1 1 1 1 1 1 1 1]
B: [1 1 1 1 1 1 1 1 1 1 1 1 1 1]
```

含义是：

- `1` 表示真实 token
- `0` 表示 padding

所以：

- `padding` 负责把 shape 补齐
- `attention_mask` 负责告诉模型哪些位置是假的，占位用，不该当作真实上下文

这两个角色不能混。

## batched prefill 里 attention mask 到底怎么生效

继续用上面的 batch。

prefill 时模型看到的是：

```text
input_ids shape      = [2, 14]
attention_mask shape = [2, 14]
```

但注意，这里的 `attention_mask` 还不是最终 attention 内部真正使用的那张四维 mask。

真正生效的约束可以理解成两层叠加：

1. `padding mask`
   - 屏蔽左边那些 PAD 列

2. `causal mask`
   - 每个位置仍然不能看未来列

所以对样本 A 来说，虽然它被 pad 到了长度 `14`，但真实有效上下文只在最后 `8` 列。

也就是说，样本 A 在 prefill 后，第一层 KV cache 物理 shape 会是：

```text
[batch=2, num_kv_heads, 14, head_dim]
```

这里对 batch 第 0 条样本来说：

- cache 的时间维一共有 `14` 列
- 但前 `6` 列只是左 padding 对应的位置
- 真正有效的 token 是最后 `8` 列

所以 prefill batching 的关键不是：

- “每条样本都天然一样长”

而是：

- “先把它们补成一样长，再用 mask 告诉模型哪些列是真的”

## KV cache 在 batched prefill 后到底长什么样

这是最容易被忽略的点。

假设：

- batch size = `2`
- 最大长度 = `14`
- `num_kv_heads = 8`
- `head_dim = 128`

那第一层 KV cache 的物理 shape 会是：

```text
K cache: [2, 8, 14, 128]
V cache: [2, 8, 14, 128]
```

这里每一维分别表示：

- `2`
  - batch 里两条请求
- `8`
  - 8 个 KV heads
- `14`
  - 这一批次对齐后的时间长度
- `128`
  - 每个 head 的维度

但这里必须区分：

- `物理 shape`
- `逻辑有效长度`

对上面的样本 A：

- 物理上 cache 有 `14` 列
- 逻辑上只有最后 `8` 列是有效 token

对样本 B：

- 14 列都有效

所以 batching 之后，KV cache 不是“每条请求都真的有一样多的有效历史”，而是：

- 每条请求都被放进一个统一大小的物理槽位里
- 哪些列有效，要靠 `attention_mask` 来区分

## batched decode 时，新 token 到底放到哪里

如果继续用 `left padding`，那 decode 时最重要的直觉就是：

- 新 token 统一接到 batch 的最右边新开出来的那一列

还是刚才那个 batch。

prefill 后：

```text
A: [PAD PAD PAD PAD PAD PAD t0 t1 t2 t3 t4 t5 t6 t7]
B: [u0  u1  u2  u3  u4  u5  u6 u7 u8 u9 u10 u11 u12 u13]
```

这时如果两条样本一起做一步 decode，当前步输入会是：

```text
input_ids shape = [2, 1]
```

假设这一步新生成的 token 分别是：

- A 生成 `a8`
- B 生成 `b14`

那么 decode 之后，可以理解成逻辑上都往右再扩一列：

```text
A: [PAD PAD PAD PAD PAD PAD t0 t1 t2 t3 t4 t5 t6 t7 a8]
B: [u0  u1  u2  u3  u4  u5  u6 u7 u8 u9 u10 u11 u12 u13 b14]
```

于是：

- `attention_mask` 也从 `[2, 14]` 变成 `[2, 15]`
- KV cache 也从：

```text
[2, 8, 14, 128]
```

增长为：

```text
[2, 8, 15, 128]
```

这就是 batched decode 时“新 token 放到哪里”的最直接答案：

- 放到统一扩出来的新最右列

## decode 时 Q / K / V / 输出的 shape 到底是什么

这也是初学者最容易突然卡住的点。

假设现在已经有：

- 历史 `4` 个 token 的 KV cache

然后又来了一个新 token。

这时总上下文长度会变成：

```text
5
```

但注意：

- 当前这一步真正输入模型的，只有这个新 token
- 前面那 `4` 个 token 不会重新做一遍 `Q / K / V`
- 它们只通过 `past_key_values` 作为历史存在

所以 decode 这一步最重要的一句话是：

- `Q` 的长度是 `1`
- `K / V` 的长度是 `5`

### 不分 head 的最朴素理解

如果先不看 multi-head，只看最基础版本：

当前新 token 进入这一层 attention 前的 hidden state 是：

```text
[1, 1, hidden_size]
```

这里三维分别表示：

- batch = `1`
- 当前步 seq_len = `1`
- hidden_size = 模型宽度

然后当前步只会新算出：

```text
Q_new: [1, 1, hidden_size]
K_new: [1, 1, hidden_size]
V_new: [1, 1, hidden_size]
```

与此同时，历史 cache 原本长度是 `4`。

把新的 `K_new / V_new` append 进去后，当前步真正参与 attention 的 K/V 会变成：

```text
K_all: [1, 5, hidden_size]
V_all: [1, 5, hidden_size]
```

于是这一步 attention 本质上在做：

```text
Q_new: [1, 1, hidden_size]
去看
K_all: [1, 5, hidden_size]
```

对应的 score shape 就是：

```text
scores: [1, 1, 5]
```

softmax 后：

```text
probs: [1, 1, 5]
```

再对 `V_all` 做加权求和后：

```text
attn_out: [1, 1, hidden_size]
```

### 分 head 后的常见写法

如果把 head 维写出来，更常见的 attention 内部 shape 会是：

```text
Q_new: [batch, num_q_heads, 1, head_dim]
K_all: [batch, num_kv_heads, 5, head_dim]
V_all: [batch, num_kv_heads, 5, head_dim]
```

如果是标准 `MHA`，通常 `num_q_heads = num_kv_heads`。

如果是 `GQA`，像 Qwen3-0.6B 这种：

- `num_attention_heads = 16`
- `num_key_value_heads = 8`
- `head_dim = 128`

那么 decode 某一步更像：

```text
Q_new: [1, 16, 1, 128]
K_all: [1,  8, 5, 128]
V_all: [1,  8, 5, 128]
```

也就是说：

- 当前 query 只有 `1` 个 token
- 但它要去看长度为 `5` 的全部历史 K/V

### 最终生成结果的 shape

这一步 attention 输出继续经过：

- output projection
- residual
- norm
- FFN
- 后续层
- 最后的词表投影

于是模型最终给出的 logits shape 会是：

```text
logits: [1, 1, vocab_size]
```

再取最后一个 token 的 logits：

```text
next_token_logits: [1, vocab_size]
```

最后选出一个 token：

```text
next_token_id: [1]
```

所以把 decode 这一步最该记住的 shape 压成一句话，就是：

```text
Q 的长度是 1
K/V 的长度是历史长度 + 1
最终 logits 的 shape 是 [batch, 1, vocab_size]
```

## 为什么这个实验必须用 left padding

因为 decode 的当前步输入总是：

```text
[batch, 1]
```

而且每做一步，历史长度都统一 `+1`。

如果用 left padding，你就可以把整个 batch 理解成：

- 左边是旧的对齐补位
- 右边是已经形成的真实前缀
- 每一步 decode 都往最右边追加一个新 token

这样：

- `attention_mask` 的增长方向清楚
- KV cache 的增长方向清楚
- 新 token 的物理落点也清楚

反过来说，如果这里用 `right padding`，就会把最关键的几件事讲乱。

还是前面的两条样本：

```text
A: [t0 t1 t2 t3 t4 t5 t6 t7 PAD PAD PAD PAD PAD PAD]
B: [u0 u1 u2 u3 u4 u5 u6 u7 u8  u9  u10 u11 u12 u13]
```

这时如果你再说“batched decode 新 token 统一接到最右边”，就会马上出现概念冲突：

- 对样本 B，最右边确实是它的新 token 位置
- 但对样本 A，右边原本还是 padding 区域

于是你会很难把下面这三件事说清楚：

1. 新 token 到底接在哪一列
2. 哪些列属于旧 padding，哪些列属于真实历史
3. KV cache 的时间维增长时，哪一列才算真正的 append 位置

所以：

- `right padding` 当然可以作为 tokenizer 默认行为存在
- 但它不适合拿来做这个 batched decoder-only decode 的教学解释

这个实验选择 `left padding`，不是随便选的风格问题，而是因为它能把下面这条物理直觉固定住：

- 左边是旧的补位
- 右边是真实历史
- decode 每一步统一往最右边 append 一个新 token

也就是说，这里用 `left padding` 是为了把：

- `attention_mask`
- `KV cache`
- `new token append`

这三件事的空间关系讲直。

## decode batching 为什么仍然麻烦

即使理解了“新 token 统一接到最右边”，decode batching 也仍然麻烦。

因为真正困难的不是：

- 当前步 input_ids 能不能写成 `[batch, 1]`

而是：

- 每条请求的逻辑有效长度不同
- 某些样本可能提前结束
- 某些样本下一步还要继续生成
- batch 会动态变化

所以在线 serving 里真正难的是：

- 谁应该继续留在 batch 里
- 谁应该退出
- 新请求什么时候插进来
- batch 内长度差异怎么控制

这就是为什么真正工程系统里会继续往下走到：

- continuous batching
- request-level scheduler
- prefill/decode mixing

## batching 带来的收益和代价

可以把 batching 理解成一个典型 tradeoff。

收益：

- 更少的 launch overhead
- 更高的并行度
- 更高的 GPU 吞吐
- 更好的硬件利用率

代价：

- 需要 padding
- 长短请求混在一起时会有 padding 浪费
- 调度会更复杂
- decode 时不同请求进度不同，batch 很难一直规则

所以真实系统里，batching 从来不是一句“拼起来一起跑”就结束了。

系统真正一直在平衡的是：

- batch 大一点，吞吐更高
- 但 batch 内长度差异太大，又会浪费很多计算

这也是后面会进一步引出：

- continuous batching
- chunked prefill
- prefill/decode mixing
- 更复杂的调度器

## 用这份目录里的实验怎么理解 batching

这个目录里的三个脚本可以按下面顺序看：

1. `single_request_decode.py`
   - 先把单条请求的真实路径看清楚

2. `batching.py`
   - 再看两条不同长度请求怎样被 left pad 成同一个 batch
   - 看 `attention_mask` 怎么标出哪些位置是真 token，哪些位置是 padding
   - 看 batched decode 时新 token 怎样统一接到最右边

3. `chunked_prefill.py`
   - 再进一步看长 prompt 为什么不一定一次性整段送进模型

这样顺序更自然，因为：

- `single request` 先解决“单条路径是什么”
- `batching` 再解决“多条路径怎么拼成规则张量”
- `chunked prefill` 再解决“单条超长路径怎样分块”

### `chunked_prefill.py`

这个脚本会做一件事：

- 把一条较长 prompt 分成多个 chunk，逐段送进模型

它主要用来看：

- 每个 chunk 进入模型时 `input_ids` 和 `attention_mask` 的 shape
- KV cache 怎样随 chunk 数逐步增长
- chunked prefill 最终是否和 full prefill 得到同样的 next-token 预测

## 运行

```bash
conda run -n aisys python experiments/hf_inference/single_request_decode.py
```

如果你已经激活了 `aisys` 环境，也可以直接：

```bash
python experiments/hf_inference/single_request_decode.py
```

## 现在最该记住什么

1. 这个目录是“真实模型推理路径实验”，不是“基础 attention 教程”。
2. `prefill` 和 `decode` 的最大区别，不是公式变了，而是：
   - prefill 一次送整段 prompt
   - decode 每步只送一个新 token，并复用已有 KV cache
3. `batching` 的本质是：把多条不同长度请求通过 `padding + attention_mask` 变成一个规则 batch。
4. `batching` 很关键，因为它直接决定吞吐；同时也很难，因为长度差异、padding 浪费和 decode 进度差异都会带来复杂度。
5. 后续 `batching`、`chunked prefill` 这类实验，应该都继续放在这个目录下，因为它们属于同一层：真实 inference path。
