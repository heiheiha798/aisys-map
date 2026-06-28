# Attention Patterns

这个目录只保留一条独立主线：

- `dense attention`
- `window attention`
- `sparse attention`
- `linear attention`

这里不再讨论别的目录，也不再依赖别的 attention 目录来解释自己。  
这四类 attention 都在这个目录里单独落成脚本。

## 这个目录在做什么

这个目录关注的是：

- token 和 token 之间怎么连接
- attention 到底看哪些位置
- attention 的计算是怎样组织的

也就是：

- 有些方法在改 `connectivity pattern`
- 有些方法在改 `execution pattern`

这里统一只用教学版 toy 输入，不追真实模型兼容，也不追 kernel 最优。

## 当前文件

- `dense_attention.py`
- `window_attention.py`
- `sparse_attention.py`
- `linear_attention.py`

## 每种 attention 在讲什么

### 1. Dense Attention

文件：

- `dense_attention.py`

这是最标准的 causal dense attention 基线。

它的特点是：

- 每个 query token 都能看见所有历史 token
- causal mask 只负责屏蔽未来位置
- attention matrix 在下三角区域是稠密的

它最适合回答：

- 标准 causal attention 到底长什么样
- 最后一个 token 为什么能看全部历史
- 后面的各种改法到底是在偏离哪条基线

一句话记忆：

- `dense attention` = 看全部可见历史

### 2. Window Attention

文件：

- `window_attention.py`

这里实现的是最直观的 sliding-window causal attention。

它的特点是：

- 每个 query 只能看最近一个窗口内的历史 token
- 更早的 token 即使是历史位置，也被直接屏蔽掉
- attention 的可见范围从“全部历史”收缩成“最近局部”

它最适合回答：

- `window` 到底裁掉了哪些连接
- 为什么这种方法天然更偏 local context
- 为什么它会直接影响 attention 读历史 token 的范围

一句话记忆：

- `window attention` = 只看最近一段历史

### 3. Sparse Attention

文件：

- `sparse_attention.py`

这里实现的是教学版 `local + global sparse attention`。

它的特点是：

- 不再要求每个 query 都看完整历史
- 一部分连接来自局部窗口
- 另一部分连接来自少量全局 token
- 整张 attention 连接图因此变稀疏

这个实现里用的是：

- local window
- global indices

所以最适合回答：

- `sparse` 到底稀疏在哪
- 为什么 sparse 不是简单“少看一点”那么随意
- 为什么局部连接和全局连接通常要配合出现

一句话记忆：

- `sparse attention` = 只保留部分有选择的连接

### 4. Linear Attention

文件：

- `linear_attention.py`

这里实现的是教学版 linear attention，而且它比前面三类更值得单独理解。

前面三类更像：

- 还是 softmax attention 主线
- query 仍然要和一批 key 显式两两交互
- 只是“能看哪些 key”不一样

也就是：

- `dense attention` 改的是“全看”
- `window attention` 改的是“只看最近窗口”
- `sparse attention` 改的是“只看部分挑选后的连接”

但 `linear attention` 不是在回答“看谁”，而是在回答：

- 还要不要显式构造一整张 `QK^T`
- 能不能把历史先压成一个递推状态
- 当前 query 能不能直接查询这个状态

所以它更像：

- 改 attention 的计算组织方式
- 改 attention 的执行范式
- 尝试摆脱显式 attention matrix

#### 为什么它更重要

`linear attention` 比 `dense / window / sparse` 更重要，不是因为它一定更快，而是因为它改动的层级更深。

`window` 和 `sparse` 本质上仍然是在说：

- attention 还是 pairwise interaction
- `q_i` 还是要和一批 `k_j` 做匹配
- 只是这批 `k_j` 的范围缩小了

但 `linear attention` 在挑战另一件事：

- attention 是否必须先形成一整行或一整块 score
- 历史信息是否必须以“所有过去 token 显式摆在那里”的方式保存

它的思路更接近：

- 把历史 token 的贡献累计成 prefix state
- 当前 query 不再显式扫完整历史
- 而是直接查询这个 prefix state

所以它不是“稀疏版 softmax attention”，而是“试图把 attention 改写成一种可递推的状态查询过程”。

#### 这份脚本里到底做了什么

`linear_attention.py` 里的核心步骤可以直接按代码变量来理解。

第一步，先得到普通的 `Q / K / V`：

- `q`
- `k`
- `v`

这部分和别的 attention 没区别。

第二步，对 `Q` 和 `K` 做 feature map：

- `q_phi = elu(q) + 1`
- `k_phi = elu(k) + 1`

这里的重点不是“用了 ELU”本身，而是：

- 把原来的 `q`、`k` 映射到另一个空间
- 让后面的 attention 计算可以写成更适合 prefix 累积的形式
- 同时尽量保持数值为正，减少归一化时的不稳定

这一步可以理解成：

- 不再直接用原始 `q^T k`
- 而是改成 `phi(q)^T phi(k)` 这类更适合重组的交互形式

第三步，维护两个 prefix state：

- `kv_prefix`
- `k_prefix`

它们分别对应：

- `kv_prefix`
  - 到当前位置为止，所有历史 token 的 `phi(k_j) outer v_j` 的累计和
- `k_prefix`
  - 到当前位置为止，所有历史 token 的 `phi(k_j)` 的累计和

这就是整份 linear attention 最关键的地方。

标准 softmax attention 更像：

- 当前 token 去显式看所有历史 token

而这里更像：

- 历史已经被压进一个可递推更新的状态

第四步，按 token 顺序在线更新：

- 先把当前 token 的 `k_phi[i]` 和 `v[i]` 写入 prefix state
- 再用当前 token 的 `q_phi[i]` 去查询这个 prefix state
- 得到当前位置输出 `out_i`

这个过程可以粗暴理解成：

- 不是“先构造一整行 attention score，再乘 V”
- 而是“先维护历史状态，再让 query 去读这个状态”

#### 为什么这里不需要显式 causal mask

这是 linear attention 最容易被低估的一点。

`dense/window/sparse` 这些 softmax attention 变体通常都需要：

- 先算 score
- 再用 mask 把未来位置屏蔽掉

但这份 linear attention 实现不一样。

它的因果性不是来自显式 mask，而是来自：

- prefix state 只按时间顺序累计
- 第 `i` 个 token 只能读到 `0..i` 的历史状态
- 未来 token 还没有被写进 `kv_prefix` 和 `k_prefix`

也就是说：

- softmax causal attention：因果性来自 mask
- linear attention：因果性来自递推顺序

这是一个本质区别。

#### 它和 sparse/window 到底差在哪

最短的区分方式：

- `window attention`
  - 还是显式 attention matrix
  - 只是每行只保留局部窗口

- `sparse attention`
  - 还是显式 attention matrix
  - 只是只保留部分挑选过的连接

- `linear attention`
  - 不以完整 `QK^T` 为主心智模型
  - 更像 prefix state + online query

所以：

- `window / sparse` 更偏“连接裁剪”
- `linear` 更偏“计算重写”

#### 应该记住的系统含义

从系统角度看，`linear attention` 重要在于它触碰的是 attention 的执行范式，而不只是连接范围。

它试图带来的变化包括：

- 不再依赖显式 attention matrix
- 更接近流式、在线的状态更新
- 让历史信息不必总以完整 token-by-token 形式参与当前计算

当然，这不等于它天然就更好。

它也会带来新的问题：

- feature map 怎么选
- 数值稳定性怎么保证
- prefix state 会不会损失表达能力
- 理论复杂度变好之后，真实实现是否真的更快

所以理解 `linear attention` 的重点不是背“线性复杂度”这几个字，而是先记住：

- 它改的不是“看谁”
- 而是“怎么算”

#### 一句话记忆

- `linear attention` = 改计算路径，不是改可见连接图

## 这四类 attention 怎么区分

最短的区分方式可以记成：

- `dense attention`：看所有可见历史
- `window attention`：只看最近窗口
- `sparse attention`：只看部分挑选后的连接
- `linear attention`：不按完整 `QK^T` 的方式算

如果再压成两类：

- `dense / window / sparse`
  - 更偏“连接图怎么定义”

- `linear`
  - 更偏“计算怎么组织”

## 当前目录的原则

- 只做教学版 dummy weight
- 先求逻辑正确
- 每种 attention 独立一个脚本
- 不用 compare 主入口
- 不引入真实模型复杂度
- 不先做 kernel 最优

所以这里暂时不做：

- block-sparse Triton kernel
- 真实 long-context benchmark
- 工程级 runtime 对接
- 真实模型权重适配

## 运行

```bash
python experiments/01_model_basics/attention_patterns/dense_attention.py
python experiments/01_model_basics/attention_patterns/window_attention.py
python experiments/01_model_basics/attention_patterns/sparse_attention.py
python experiments/01_model_basics/attention_patterns/linear_attention.py
```

如果用当前机器上的 `aisys` 环境：

```bash
conda run -n aisys python experiments/01_model_basics/attention_patterns/dense_attention.py
conda run -n aisys python experiments/01_model_basics/attention_patterns/window_attention.py
conda run -n aisys python experiments/01_model_basics/attention_patterns/sparse_attention.py
conda run -n aisys python experiments/01_model_basics/attention_patterns/linear_attention.py
```

## 现在最该记住什么

1. 这个目录里的每个脚本只负责一种 attention。
2. `dense / window / sparse` 主要差在“看谁”。
3. `linear` 主要差在“怎么算”。
4. 这四种 attention 已经在这个目录里自洽，不需要再拆成两个 md 去解释。
