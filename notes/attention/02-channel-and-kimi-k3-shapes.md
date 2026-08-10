# 02：KDA 与 Kimi K3 的真实 Shape

前置阅读：[`00-from-vanilla-attention-to-kda.md`](./00-from-vanilla-attention-to-kda.md) 和 [`01-linear-attention-background.md`](./01-linear-attention-background.md)。本文直接接续第 01 篇的 Gated DeltaNet。

本文使用 Kimi K3 官方公开的模型配置和实现。查询日期：2026-08-10。

官方资料：

- [Kimi K3 官方仓库](https://github.com/MoonshotAI/Kimi-K3)
- [Kimi K3 官方 config.json](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)
- [Kimi K3 官方 KDA 实现](https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_linear.py)
- [Kimi K3 Technical Report](https://arxiv.org/abs/2607.24653)

## 1. 阅读 KDA Shape 时，Channel 指什么

第 00 篇已经用 `[B,T,D]` 解释过 channel：向量最后一维中的每个坐标就是一个 channel。进入 KDA 后，只需区分三个层级：

- hidden-state channel：Kimi K3 有 7168 个；
- 某个 KDA head 内的 q/k/v channel：有 128 个；
- attention head：Kimi K3 有 96 个，每个 head 包含一组独立的 128 维表示。Head 和 channel 分属两个层级。

后面的状态矩阵 $S\in\mathbb R^{128\times128}$ 中，行对应 key channel，列对应 value channel。

## 2. 从 Gated DeltaNet 到 KDA

第 01 篇最后得到 Gated DeltaNet：一个 head 使用标量 $\alpha_t$ 衰减整张旧状态。KDA 将它扩展成长度为 $d_k$ 的向量：

$$
\boldsymbol\alpha_t=
[\alpha_{t,0},\alpha_{t,1},\ldots,\alpha_{t,d_k-1}]
$$

形成对角矩阵：

$$
D_t=\operatorname{Diag}(\boldsymbol\alpha_t)
$$

一个 token、一个 head 的 KDA 更新分为四步：

$$
\bar S_t=D_tS_{t-1}
$$

$$
\hat v_t=\bar S_t^\top k_t
$$

$$
S_t=\bar S_t+
\beta_tk_t(v_t-\hat v_t)^\top
$$

$$
o_t=S_t^\top q_t
$$

依次表示逐 key-channel 衰减、读取当前预测、写入 prediction error、使用 query 读取输出。合并后的递推式为：

$$
\boxed{
S_t=
(I-\beta_tk_tk_t^\top)
\operatorname{Diag}(\boldsymbol\alpha_t)S_{t-1}
+\beta_tk_tv_t^\top
}
$$

第 01 篇已经证明了 delta correction 对当前重建误差的局部性质。本篇只研究新增的向量 decay 以及它在 Kimi K3 中的真实 shape。

## 3. Kimi K3 的官方 Attention 配置

官方配置给出的关键数值是：

| 配置 | Kimi K3 数值 |
| --- | ---: |
| Transformer 层数 | 93 |
| KDA 层数 | 69 |
| Gated MLA 层数 | 24 |
| hidden size | 7168 |
| KDA head 数 $H$ | 96 |
| 每个 KDA head 的维度 $D$ | 128 |
| 最大 context length | 1,048,576 |

需要注意：

$$
96\times128=12288
$$

这比 hidden size 7168 大。没有矛盾，因为 q/k/v 是通过学习到的线性层从 7168 维投影到 12288 维的；attention 完成后，再通过 output projection 从 12288 维投影回 7168 维。

## 4. K3 一个 KDA 层的真实 Tensor Shape

先定义符号：

- $B$：batch size；
- $T$：当前一起处理的 token 数；
- $H=96$：head 数；
- $D=128$：每个 head 的 q/k/v 维度；
- $d_{model}=7168$：hidden size。

KDA 层的输入是：

$$
X\in\mathbb R^{B\times T\times7168}
$$

官方实现中的 q/k/v projection 都把最后一维从 7168 投影到 12288：

$$
Q_{flat},K_{flat},V_{flat}
\in\mathbb R^{B\times T\times12288}
$$

然后拆成 96 个 head：

$$
Q,K,V\in\mathbb R^{B\times T\times96\times128}
$$

Forget gate 的原始输入以及论文中的逐 channel decay 也对应每个 head 的 128 个 channel：

$$
\alpha\in\mathbb R^{B\times T\times96\times128}
$$

Delta update 的强度 $\beta$ 是每个 token、每个 head 一个标量：

$$
\beta\in\mathbb R^{B\times T\times96}
$$

数学上，每个 head 独立维护一张状态矩阵：

$$
S_t^{(h)}\in\mathbb R^{128\times128}
$$

把 batch 和所有 head 放在一起：

$$
S_t\in\mathbb R^{B\times96\times128\times128}
$$

这里两个 128 含义不同：

- 第一个 128 是 key channel，对应矩阵的行；
- 第二个 128 是 value channel，对应矩阵的列。

K3 恰好令 $d_k=d_v=128$，所以状态是方阵。如果 $d_k$ 和 $d_v$ 不相等，状态会是矩形 $d_k\times d_v$。

## 5. 给出一个具体 Batch 的 Shape

假设：

```text
batch size B = 2
sequence length T = 1024
```

那么一个 KDA 层中主要 tensor 的 shape 是：

| Tensor | Shape |
| --- | --- |
| 输入 hidden states | `[2, 1024, 7168]` |
| q projection 后、拆 head 前 | `[2, 1024, 12288]` |
| q | `[2, 1024, 96, 128]` |
| k | `[2, 1024, 96, 128]` |
| v | `[2, 1024, 96, 128]` |
| channel-wise decay $\alpha$ | `[2, 1024, 96, 128]` |
| update gate $\beta$ | `[2, 1024, 96]` |
| recurrent state | `[2, 96, 128, 128]` |
| 每个 head 的输出 | `[2, 1024, 96, 128]` |
| 拼接所有 head 后 | `[2, 1024, 12288]` |
| output projection 后 | `[2, 1024, 7168]` |

自回归解码一个新 token 时，$T=1$：

```text
hidden: [B, 1, 7168]
q/k/v:  [B, 1, 96, 128]
state:  [B, 96, 128, 128]
output: [B, 1, 7168]
```

序列从 1K 增长到 1M token 时，KDA state 仍然是 `[B, 96, 128, 128]`。矩阵内容持续变化，shape 保持不变。

单个 batch item、单层共有：

$$
96\times128\times128=1,572,864
$$

个 state 元素。只按 BF16 的 2 bytes 做量级估算约为 3 MiB；实际 kernel 可能采用更高精度保存或计算 state。

## 6. Channel-wise Decay 到底衰减什么

只看某一个 head，K3 的 decay 向量是：

$$
\alpha_t^{(h)}=
[\alpha_0,\alpha_1,\ldots,\alpha_{127}]
$$

对应状态矩阵：

$$
S_{t-1}^{(h)}=
\begin{bmatrix}
\text{第 0 个 key channel 对应的一整行}\\
\text{第 1 个 key channel 对应的一整行}\\
\vdots\\
\text{第 127 个 key channel 对应的一整行}
\end{bmatrix}
$$

执行：

$$
\bar S_t^{(h)}=
\operatorname{Diag}(\alpha_t^{(h)})S_{t-1}^{(h)}
$$

等价于：

```text
state 的第 0 行   *= alpha[0]
state 的第 1 行   *= alpha[1]
...
state 的第 127 行 *= alpha[127]
```

写到单个矩阵元素上，就是：

$$
\bar S[i,j]=\alpha[i]S[i,j]
$$

因此，$128\times128$ 状态矩阵中的每个数都会乘到一个 decay；同一行的 128 个数共享同一个 $\alpha[i]$。一个 head 每个 token 产生 128 个独立 decay，数量没有扩展到 16384 个。K3 的 96 个 heads 合计产生：

$$
96\times128=12288
$$

个 decay 标量，每个 batch item、每个 token 都会重新生成一组。

所以 channel-wise decay 的准确含义是：

> 一个 head 内的 128 个 key-space 方向，可以使用 128 个不同的遗忘比例。

历史 token 已经被压缩进 $S$，所以这里的 gate 作用于 state 的 key-space 方向。KDA 每个 head 有 128 个 decay 值；Gated DeltaNet 每个 head 使用一个 scalar decay。

## 7. 用缩小到 2 个 Channel 的数值例子手算一次

K3 单个 head 实际是 $128\times128$，无法在页面上完整手算。下面保持相同运算，但临时缩小为 $2\times2$。

假设旧状态是：

$$
S_{t-1}=
\begin{bmatrix}
1&0\\
0&2
\end{bmatrix}
$$

当前 token 产生：

$$
\alpha=
\begin{bmatrix}
0.5\\
1.0
\end{bmatrix},\qquad
k=
\begin{bmatrix}
1\\
0
\end{bmatrix},\qquad
v=
\begin{bmatrix}
3\\
4
\end{bmatrix},\qquad
\beta=0.5
$$

### 第一步：逐 channel 衰减

$$
\bar S=
\operatorname{Diag}(\alpha)S_{t-1}
=
\begin{bmatrix}
0.5&0\\
0&2
\end{bmatrix}
$$

第 0 个 key channel 对应的整行乘了 0.5；第 1 行乘了 1，因此保持不变。

### 第二步：用当前 key 读取旧记忆

$$
\hat v=\bar S^\top k
=
\begin{bmatrix}
0.5\\
0
\end{bmatrix}
$$

目标 value 是 $[3,4]^\top$，所以 reconstruction error 是：

$$
e=v-\hat v
=
\begin{bmatrix}
2.5\\
4
\end{bmatrix}
$$

### 第三步：写入一半误差

$$
\beta ke^\top
=0.5
\begin{bmatrix}
1\\0
\end{bmatrix}
\begin{bmatrix}
2.5&4
\end{bmatrix}
=
\begin{bmatrix}
1.25&2\\
0&0
\end{bmatrix}
$$

新状态为：

$$
S_t=\bar S+\beta ke^\top
=
\begin{bmatrix}
1.75&2\\
0&2
\end{bmatrix}
$$

### 第四步：用 query 读取输出

如果当前 query 是：

$$
q=
\begin{bmatrix}
1\\0
\end{bmatrix}
$$

输出为：

$$
o=S_t^\top q
=
\begin{bmatrix}
1.75\\2
\end{bmatrix}
$$

如果 query 是与当前 key 正交的 $[0,1]^\top$，则输出为 $[0,2]^\top$。刚才沿第 0 个 key channel 写入的 correction 不会影响这个正交方向。

K3 的每个 head 做的是完全相同的操作，只是把 2 个 channel 换成 128 个，并同时运行 96 个 head。

## 8. KDA 有没有标准 Attention 的 $T\times T$ 权重矩阵

没有显式构造。

标准 Attention 通常产生：

$$
A\in\mathbb R^{B\times H\times T\times T}
$$

其中 $A_{t,i}$ 表示第 $t$ 个 query 对第 $i$ 个历史 token 的权重。

KDA 则维护：

$$
S_t\in\mathbb R^{B\times H\times128\times128}
$$

历史 token 已经被更新规则压缩进 state，所以不能直接查看“当前 token 对第 372 个历史 token 的 attention weight”。KDA 的读取是：

$$
o_t^{(h)}=S_t^{(h)\top}q_t^{(h)}
$$

这也是 KDA 比 full attention 节省长上下文存储，但精确定位历史 token 更困难的根本原因。

## 9. Kimi K3 为什么混合 KDA 与 MLA

第 00 篇介绍的 MLA 仍会为每个历史 token 保存 compressed latent，并允许当前 query 区分各个历史位置。KDA 将全部历史压进固定状态 $S$，存储量不随 context length 增长，逐 token 的位置边界也随压缩消失。

| 机制 | 持久状态 | 随 context length 增长 | 精确选择历史位置 |
| --- | --- | --- | --- |
| MLA | 每个 token 的 KV latent 与 RoPE key | 是 | 可以 |
| KDA | 每个 head 的固定矩阵 $S$ | 否 | 较困难 |

Kimi K3 共 93 层，其中 69 层使用 KDA，24 层使用 Gated MLA。其重复模式可以写成：

```text
KDA -> KDA -> KDA -> Gated MLA
```

KDA 层承担低成本的持续状态更新，Gated MLA 层保留逐位置全局检索路径。两类层组合后，同时利用固定状态的长上下文效率和 full attention 的定位能力。

## 10. 哪些是 K3 的真实数值，哪些只是讲解示例

本文中的以下数值来自 Kimi K3 官方发布：

- hidden size 7168；
- 96 个 KDA head；
- head dimension 128；
- 69 个 KDA 层和 24 个 Gated MLA 层；
- q/k/v 拆 head 后的 shape；
- 每个 head 的 state shape 为 $128\times128$。

第 7 节里的矩阵元素 0.5、1、2、3、4 仅供手算示例使用，未取自 K3 的真实推理 activation。

K3 的真实 q/k/v 数值会随以下条件变化：

- 具体输入和 tokenizer 结果；
- 第几层；
- 第几个 head；
- 前面所有 token 形成的 hidden state；
- 模型权重和运行精度。

要取得真实 activation，需要指定 prompt，并实际加载 K3 权重运行 forward hook。K3 是 2.8T 参数模型，本地是否能够运行取决于可用硬件；仅靠 config 无法推导某个 token 的真实 q/k/v 元素值。
