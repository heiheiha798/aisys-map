# 01：从 Linear Attention 到 Gated DeltaNet

前置阅读：[`00-from-vanilla-attention-to-kda.md`](./00-from-vanilla-attention-to-kda.md)。如果 MHA、MQA、GQA、MLA 的 head 和 tensor shape 还不熟悉，请先读 00。

这份笔记只讲理解 KDA 递推式所需的背景，暂时忽略 chunkwise、WY representation、UT transform 和 GPU kernel。第 00 篇已经讲过 causal self-attention、softmax、MHA 和 MLA，这里直接从“如何压缩历史”开始。

主线是：

> Linear Attention -> 关联记忆 -> Reconstruction Error -> Delta Rule -> Gate / Decay -> Gated DeltaNet

## 0. 先解释 $t$ 和“固定大小”

$t$ 表示当前正在处理的 token 位置，也可以理解为时间步。

例如句子有四个 token：

```text
token:  我    喜欢    线性    注意力
t:      1      2       3       4
```

- $S_1$：处理完第 1 个 token 后的记忆。
- $S_2$：处理完第 2 个 token 后的记忆。
- $S_t$：处理完第 $t$ 个 token 后的记忆。

下标 $t$ 只是在说明“这是哪个时刻的状态”，并不表示 $S_t$ 有 $t$ 行或 $t$ 列。

假设 key 维度是 $d_k$，value 维度是 $d_v$：

$$
k_t \in \mathbb R^{d_k}, \qquad
v_t \in \mathbb R^{d_v}
$$

它们的外积始终是：

$$
k_t v_t^\top \in \mathbb R^{d_k \times d_v}
$$

状态更新为：

$$
S_t = S_{t-1} + k_t v_t^\top
$$

由于两个相加的矩阵形状必须相同，所以每一步都有：

$$
S_t \in \mathbb R^{d_k \times d_v}
$$

因此，$S_1,S_2,\ldots,S_t$ 的**内容不同，形状始终相同**。这些符号表示同一个记忆状态在不同时刻的连续版本，实际计算只需保留当前版本：

```text
S_0 --处理 token 1--> S_1 --处理 token 2--> S_2 --...--> S_t
```

标准 Attention 会额外保存每个历史位置的 $k_i,v_i$，所以存储量随 $t$ 增长。Linear Attention 则不断把新的 $k_tv_t^\top$ 合并进同一个 $d_k \times d_v$ 状态矩阵，因此状态大小不随序列长度增长。

“固定大小”只表示存储形状固定，不表示记忆容量无限。越来越多的信息被压进同一个矩阵，正是 Linear Attention 可能发生信息干扰和遗忘的原因。

### 0.1 固定形状不等于数值不会变大

这是 vanilla linear attention 的一个核心问题。

它不断执行：

$$
S_t=S_{t-1}+k_tv_t^\top
$$

即使 $S_t$ 始终是同样大小的矩阵，里面的元素和矩阵范数仍可能随着 $t$ 持续增长。正负数都可能有问题：

- 同方向的写入不断叠加，数值可能越来越大。
- 不同方向的写入互相抵消，可能丢失原有信息。
- 很久以前、已经过时的信息仍留在状态中，干扰新信息。

Decay 的确是解决这个问题的重要机制。最简单的形式是：

$$
S_t=\alpha S_{t-1}+u_t, \qquad 0\leq\alpha<1
$$

其中 $u_t$ 表示本次新写入的信息。如果每次写入的大小不超过 $U$，那么旧信息会按 $\alpha^n$ 衰减，状态大小大致受下面的几何级数控制：

$$
\|S_t\| \lesssim U(1+\alpha+\alpha^2+\cdots)
=\frac{U}{1-\alpha}
$$

所以 decay 能够阻止旧写入永久、等强度地累积，并为记忆提供有限寿命。

完整机制还会使用 delta rule 控制写入，用 L2Norm 控制 q/k 长度，用有界 gate 控制更新强度，并用 RMSNorm 控制输出尺度。后续章节会按这个顺序展开。

另外，$\alpha_t$ 是模型根据输入动态产生的，有时可能非常接近 1，所以 decay 无法为所有情况提供“绝对有界”的数学保证。但它给了模型主动控制记忆大小和寿命的能力。

## 1. Linear Attention 的基本想法

第 00 篇已经给出了 causal self-attention 的完整计算。这里沿用一个结论：full attention 保存逐 token 的 K/V，并让当前 query 与各个历史 key 分别匹配。

先暂时去掉 softmax，考虑：

$$
o_t = \sum_{i \le t}(q_t^\top k_i)v_i
$$

利用矩阵乘法结合律，可以改写成：

$$
o_t = \left(\sum_{i \le t}k_i v_i^\top\right)^\top q_t
$$

定义固定大小的状态矩阵：

$$
S_t = \sum_{i \le t}k_i v_i^\top
$$

就得到递推形式：

$$
S_t = S_{t-1} + k_t v_t^\top
$$

$$
o_t = S_t^\top q_t
$$

两种机制的直觉区别是：

- 标准 Attention 保存每条历史记录，查询时重新搜索。
- Linear Attention 把历史不断压进固定大小的矩阵 $S$，查询时只读取 $S$。

这里的 "linear" 主要指计算量相对于序列长度 $T$ 线性增长。整个模型依然包含大量非线性运算。

### 1.1 Feature map 是什么

Feature map（特征映射）就是一个函数 $\phi$，它把原向量转换成另一个向量：

$$
\phi:\mathbb R^d\rightarrow\mathbb R^m
$$

例如，最简单的 feature map 是恒等映射：

$$
\phi(x)=x
$$

这时相似度仍是普通点积：

$$
\phi(q)^\top\phi(k)=q^\top k
$$

也可以使用逐元素的非线性映射，例如：

$$
\phi(x)=\operatorname{ELU}(x)+1
$$

它会使映射后的每个分量为正，从而使点积相似度非负。

在这里，feature map 负责把所需相似度写成下面这种可分解形式；语义特征提取并非此处讨论的重点：

$$
\operatorname{sim}(q,k)=\phi(q)^\top\phi(k)
$$

这种能写成特征空间内积的相似度也叫 kernel。不同的 $\phi$ 就定义了不同的 query-key 相似度。

### 1.2 Feature map 为什么能得到 Linear Attention

使用上述相似度，先考虑没有归一化的输出：

$$
o_t=\sum_{i\leq t}
\left(\phi(q_t)^\top\phi(k_i)\right)v_i
$$

由于 $\phi(q_t)$ 与历史位置 $i$ 无关，可以利用结合律将它移到求和外面：

$$
o_t=\phi(q_t)^\top
\left(\sum_{i\leq t}\phi(k_i)v_i^\top\right)
$$

于是可以提前维护状态：

$$
S_t=\sum_{i\leq t}\phi(k_i)v_i^\top
$$

查询时只需计算：

$$
o_t=\phi(q_t)^\top S_t
$$

关键在于相似度被拆成只依赖 query 的部分和只依赖 key 的部分，因此可以预先汇总所有历史 key-value。非线性变换本身无法直接带来这种计算重排。

### 1.3 Linear Attention 里的归一化

如果直接使用未归一化相似度，历史 token 越多，输出通常也越容易变大。因此经典 kernel linear attention 常将权重除以总相似度：

$$
o_t=
\frac{
\sum_{i\leq t}
\left(\phi(q_t)^\top\phi(k_i)\right)v_i
}{
\sum_{i\leq t}\phi(q_t)^\top\phi(k_i)
}
$$

分子可以用矩阵状态 $S_t$ 计算。分母也可以维护一个向量状态：

$$
z_t=\sum_{i\leq t}\phi(k_i)
$$

因此最终形式是：

$$
o_t=
\frac{\phi(q_t)^\top S_t}
{\phi(q_t)^\top z_t}
$$

这个分母的作用是让当前 query 对历史位置的权重总和为 1。例如原始相似度是 2 和 1，归一化后就是 $2/3$ 和 $1/3$。

它与 softmax 并不相同：

$$
\text{普通归一化：}\quad
\frac{2}{2+1},\frac{1}{2+1}
$$

$$
\text{softmax：}\quad
\frac{e^2}{e^2+e^1},\frac{e^1}{e^2+e^1}
$$

二者都使权重总和为 1，但 softmax 还通过指数产生更强的选择性。

### 1.4 三种归一化分别作用在哪里

KDA 相关材料中会遇到三种不同操作：

| 名称 | 操作对象 | 作用 |
| --- | --- | --- |
| Attention weight normalization | 所有历史位置的权重 | 让权重总和为 1 |
| L2Norm | 单个 $q$ 或 $k$ 向量 | 让向量长度为 1，点积变成 cosine similarity |
| RMSNorm | KDA 输出或 hidden state | 控制输出向量的整体数值尺度 |

L2Norm 的定义是：

$$
\operatorname{L2Norm}(x)=\frac{x}{\|x\|_2}
$$

它只保证每个 $q/k$ 向量自身长度为 1，并不保证不同历史位置的 attention 权重总和为 1。

KDA 使用 L2-normalized $q/k$ 来改善状态更新的稳定性，但它的核心递推式没有经典 kernel linear attention 的 $z_t$ 分母。因此 KDA 不应简单理解成“用 feature map 近似 softmax”；它更接近一个带有学习、纠错和遗忘机制的 recurrent associative memory。

## 2. 为什么 $S$ 是关联记忆

假设：

$$
S \in \mathbb{R}^{d_k \times d_v}
$$

那么：

$$
S^\top k \in \mathbb{R}^{d_v}
$$

因此可以把 $S$ 看成一个从 key 到 value 的临时映射：

$$
k \longmapsto v
$$

写入一个 key-value 对时：

$$
S \leftarrow S + kv^\top
$$

读取时：

$$
\hat v = S^\top q
$$

如果 $q$ 与某个历史 key 相似，就会读出与它相关的 value。这就是 associative memory，也称 fast-weight memory。

这里的 fast weight 是处理序列时不断变化的 $S_t$。通过正常反向传播学习的 $W_q$、$W_k$、$W_v$ 等模型参数则是 slow weights。

## 3. Vanilla Linear Attention 的问题

简单相加没有覆盖或删除机制：

$$
S_t = S_{t-1} + k_t v_t^\top
$$

假设同一个归一化 key $k$ 先对应标量 value 3，后来变为 5。简单相加后，读取结果会接近：

$$
S^\top k = 3 + 5 = 8
$$

合理的更新结果应该是 5；直接累加却得到了 8。

当大量 key 方向相似时，它们还会写入 $S$ 的相同区域，造成 memory interference。因为 $S$ 大小固定，序列越长，压缩冲突通常越明显。

### 3.1 什么叫“重建”

这里的“重建”只描述一条 key-value 映射的还原过程，不涉及整个 token、句子或原始输入：

> 我们曾想让记忆保存映射 $k\mapsto v$；现在再次用 $k$ 查询记忆，看它能否还原出原来的 $v$。

状态矩阵 $S$ 被当作一个简单的 key-to-value 函数。给它 key $k$，它返回：

$$
\hat v=S^\top k
$$

$\hat v$ 读作 “v hat”，表示记忆当前给出的预测 value。

我们原本希望这个 key 对应的 value 是 $v$。因此：

- 目标值是 $v$。
- 记忆读出的预测值是 $\hat v=S^\top k$。
- 如果 $\hat v=v$，说明这条 key-value 映射记对了。
- 如果二者不同，说明记忆需要被修正。

把“从 key 还原出 value”的过程称为 reconstruction，把预测值与目标值的差称为 reconstruction error。

### 3.2 最简单的标量例子

先假设 value 只是一个数字。我们希望记忆保存：

```text
这个 key -> 5
```

但当前从 $S$ 中读出的结果是：

```text
这个 key -> 3
```

那么 reconstruction error 是：

$$
e=v-\hat v=5-3=2
$$

这个 2 表示记忆的输出还缺少 2。Delta rule 接下来会沿着这个 key 的方向，把缺少的部分写入 $S$。

如果当前已经读出 5，那么：

$$
e=5-5=0
$$

此时不需要再次写入。这正是 delta rule 与简单累加的关键区别。

### 3.3 当 value 是向量时

实际的 $v$ 是一个向量。例如：

$$
v=
\begin{bmatrix}
2\\
-1
\end{bmatrix},
\qquad
\hat v=
\begin{bmatrix}
1.5\\
-0.2
\end{bmatrix}
$$

误差向量就是逐元素相减：

$$
e=v-\hat v=
\begin{bmatrix}
0.5\\
-0.8
\end{bmatrix}
$$

它不仅表示“错了多少”，也表示每个 value 维度应该向哪个方向修正。

### 3.4 为什么还要定义一个 reconstruction loss

误差 $e$ 是向量，不方便用一个数衡量整体错误大小。因此通常使用 squared L2 norm：

$$
\mathcal L(S)=\frac12\|\hat v-v\|_2^2
=\frac12\|S^\top k-v\|_2^2
$$

对上面的二维例子：

$$
\mathcal L
=\frac12\left(0.5^2+(-0.8)^2\right)
=0.445
$$

这个 loss 有几个简单性质：

- 预测完全正确时，loss 为 0。
- 错得越多，loss 越大。
- 正负误差平方后都计入，不会互相抵消。
- 前面的 $1/2$ 只是为了求导时抵消平方产生的 2，不改变最优点。

### 3.5 它与语言模型训练 Loss 属于两类目标

这里容易混淆两层优化：

1. **内部在线记忆目标**：在处理当前序列时，$S_t$ 用 reconstruction error 更新，使它更好地保存 $k_t\mapsto v_t$。
2. **外部语言模型训练目标**：整个模型通过 next-token cross-entropy 和反向传播学习 $W_q,W_k,W_v$、gate 等长期参数。

$v_t$ 来自当前 token hidden state 的模型投影，随后写入临时记忆，无需人工标签。Reconstruction loss 主要用于推导 delta update；实现中通常可以直接执行更新公式，无需单独计算该 loss 并调用一次反向传播。

到这里，delta rule 可以先只读成一句话：

> 用 key 读取当前记忆；比较读出的 $\hat v$ 和想存的 $v$；只把二者之间缺少的部分写进去。

## 4. Delta Rule：只写入预测误差

Delta rule 在写入之前，先询问旧记忆对当前 key 的预测：

$$
\hat v_t = S_{t-1}^\top k_t
$$

然后计算 prediction error：

$$
e_t = v_t - \hat v_t
$$

最后只写入误差：

$$
S_t = S_{t-1} + \beta_t k_t e_t^\top
$$

也就是：

$$
\boxed{
S_t = S_{t-1} + \beta_t k_t
\left(v_t - S_{t-1}^\top k_t\right)^\top
}
$$

$\beta_t \in [0,1]$ 控制这次修正的强度。

展开后得到论文中的形式：

$$
S_t =
(I - \beta_t k_t k_t^\top)S_{t-1}
+ \beta_t k_t v_t^\top
$$

它来自重建误差：

$$
\mathcal L_t(S) =
\frac{1}{2}\left\|S^\top k_t-v_t\right\|^2
$$

对 $S$ 做一步梯度下降，就会得到上面的更新。因此 delta 指的是目标值与当前预测值之差。

如果 $k_t$ 已归一化，并且 $\beta_t=1$，那么更新后：

$$
S_t^\top k_t = v_t
$$

这意味着同一个 key 再次出现时，delta rule 会纠正旧映射，避免新旧 value 无限制累加。

### 4.1 数学性质与经验效果的适用范围

最准确的结论是：

> Delta rule 对“用线性记忆 $S$ 重建当前 key-value 映射”具有可证明的局部最优性质；但它并不保证整个语言模型或所有历史记忆达到全局最优。

下面分别说明它能证明什么。

### 4.2 性质一：当前重建误差一定缩小

记更新前的预测误差为：

$$
e=v-S^\top k
$$

做一次 delta update：

$$
S'=S+\beta ke^\top
$$

更新后的预测为：

$$
S'^\top k=S^\top k+\beta e\|k\|_2^2
$$

所以新误差是：

$$
e'=v-S'^\top k
=\left(1-\beta\|k\|_2^2\right)e
$$

当 $k$ 经过 L2Norm 后，$\|k\|_2=1$，于是：

$$
e'=(1-\beta)e
$$

相应的重建 loss 变为：

$$
\mathcal L(S')=(1-\beta)^2\mathcal L(S)
$$

因此，只要 $0<\beta\leq1$，当前样本的重建误差不会变大；$\beta=1$ 时，当前 key 的误差直接变成 0。

例如旧记忆对某个 key 的预测是 3，新目标是 5：

- 误差是 $5-3=2$。
- $\beta=0.5$ 时，新预测是 $3+0.5\times2=4$。
- $\beta=1$ 时，新预测正好是 5。

这是一种负反馈：已经预测正确的部分不会再次写入，只有错误部分会被修正。

### 4.3 性质二：它是满足当前映射的最小改动

假设 $\|k\|_2=1$、$\beta=1$，希望找到一个改动 $\Delta S$，使更新后恰好满足：

$$
(S+\Delta S)^\top k=v
$$

等价于约束：

$$
\Delta S^\top k=e
$$

在所有满足该约束的改动中，下面这个解的 Frobenius norm 最小：

$$
\boxed{\Delta S=ke^\top}
$$

它正是 delta rule。

简单证明如下。把 $\Delta S$ 的第 $j$ 列记作 $\Delta s_j$，约束就是：

$$
k^\top\Delta s_j=e_j
$$

由 Cauchy--Schwarz 不等式：

$$
|e_j|=|k^\top\Delta s_j|
\leq\|k\|_2\|\Delta s_j\|_2
$$

当 $\|k\|_2=1$ 时，任何可行改动都必须满足 $\|\Delta s_j\|_2\geq|e_j|$。只有当 $\Delta s_j=e_jk$，即每一列都平行于 $k$ 时取到等号。把所有列合起来正好得到 $\Delta S=ke^\top$。

直觉上，若只需要修正 key $k$ 对应的预测，就没必要修改与 $k$ 垂直的矩阵方向。Delta rule 只沿 $k$ 的方向改动 $S$，因此是完成当前纠错所需的最小矩阵变化。

若 $k$ 没有归一化，最小改动是：

$$
\Delta S=\frac{ke^\top}{\|k\|_2^2}
$$

这也是 KDA 对 key 做 L2Norm 后公式会特别简洁的原因之一。

### 4.4 性质三：对其他记忆的影响由 key 相似度决定

考虑另一个查询 $q$。Delta update 对它的读取结果造成的变化为：

$$
\begin{aligned}
\Delta o(q)
&=(\Delta S)^\top q\\
&=\beta e(k^\top q)
\end{aligned}
$$

因此：

- 如果 $q$ 与当前 $k$ 正交，即 $k^\top q=0$，该查询的记忆完全不受影响。
- 如果二者相似，修改会按相似度传播过去。
- 如果二者非常相似却应该对应不同 value，它们仍会互相干扰。

所以 delta rule 能局部化写入，但无法突破固定维度记忆的容量限制。Key 是否能在学习后形成足够可区分的方向非常重要。

### 4.5 从优化角度看，它是什么

Delta rule 是当前重建 loss 的一次梯度下降：

$$
\mathcal L_t(S)=\frac12\|S^\top k_t-v_t\|^2
$$

在经典在线学习中，这也称为 Least Mean Squares（LMS）或 Widrow-Hoff update。若目标映射固定、数据满足一定统计条件、学习率合适，它具有相应的收敛结果。

但语言序列中的目标会随上下文变化，token 通常不满足独立同分布假设；KDA 的 $k_t,v_t,\alpha_t,\beta_t$ 也都由神经网络学习得到。因此这些经典结果不能直接证明 KDA 对语言建模全局最优。

### 4.6 最终该怎样评价 Delta Rule

它同时具有两种身份：

1. **有数学依据的在线更新规则**：对当前平方重建误差做梯度下降，并且在归一化 key 下具有精确纠错、最小改动和局部影响等性质。
2. **一种 inductive bias**：模型被限制为用“预测误差修正关联记忆”的方式处理历史信息。这种 bias 是否优于其他架构，最终仍要通过端到端训练和实验验证。

因此，不能说“Delta rule 在线性代数上全面优于所有更新规则”。更合适的说法是：

> 如果我们接受“$S$ 是一个在线学习的线性 key-to-value 映射”这个建模前提，那么 delta rule 是一个非常自然且有局部最优性质的更新；这个建模前提对语言模型是否有效，则是经验问题。

## 5. Gate 和 Decay

Gate 是模型根据当前输入生成的控制量，通常位于 $[0,1]$：

- 接近 1：保留信息或允许信息通过。
- 接近 0：遗忘信息或阻断信息。
- 位于中间：部分保留。

Decay 是一种用于旧状态的 gate。例如：

$$
S_t = 0.99S_{t-1} + \text{new information}
$$

一条旧信息经过 100 步后，剩余比例约为：

$$
0.99^{100} \approx 0.366
$$

如果每一步的 decay 不同，从第 $i$ 步到第 $t$ 步的累计保留比例就是：

$$
\prod_{j=i+1}^{t}\alpha_j
$$

所以 decay 也隐式提供距离或新旧信息：一般来说，越早的信息被连续衰减的次数越多。

## 6. Gated DeltaNet

Gated DeltaNet 先用标量 $\alpha_t$ 衰减旧记忆：

$$
\bar S_t = \alpha_t S_{t-1}
$$

然后执行 delta correction：

$$
S_t = \bar S_t +
\beta_t k_t\left(v_t-\bar S_t^\top k_t\right)^\top
$$

展开后是：

$$
S_t =
\alpha_t(I-\beta_t k_tk_t^\top)S_{t-1}
+\beta_t k_tv_t^\top
$$

这里有两个不同控制量：

- $\alpha_t$ 是 forget/decay gate，控制旧记忆保留多少。
- $\beta_t$ 是 update/write gate，控制当前误差修正多少。

GDN 的 $\alpha_t$ 是一个标量，因此一个 head 内所有记忆方向只能一起衰减。它无法让一部分信息长期保留，同时让另一部分信息迅速遗忘。

下一篇将标量 decay 扩展为逐 channel decay，并代入 Kimi K3 的真实 shape：[`02-channel-and-kimi-k3-shapes.md`](./02-channel-and-kimi-k3-shapes.md)。
