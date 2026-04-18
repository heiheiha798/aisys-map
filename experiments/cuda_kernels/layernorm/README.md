# LayerNorm and RMSNorm

这个目录放两个最基础的归一化 kernel 实验：

- `row_layernorm.cu`
- `row_rmsnorm.cu`

这两个实验的重点不是追极致性能，而是把下面这些概念讲清楚：

- 为什么 `layernorm` 本质上是 row-wise reduction + elementwise normalize
- 为什么 `rmsnorm` 比 `layernorm` 少了一步减均值
- shared memory reduction 怎么服务于高频 memory-bound kernel

## 先说清楚：这里的 norm 是什么意思

如果之前没系统看过各种 `norm`，可以先把它们粗糙理解成：

- **按照某种统计量，把一组数重新缩放到更稳定的范围**

这里的“稳定”不是说所有数都一样，而是说：

- 不希望某一行特别大、某一行特别小
- 不希望每一层输入的尺度波动太剧烈
- 希望后面的计算更容易训练，也更不容易数值爆炸

在这个目录里，`norm` 讨论的不是整个矩阵一起做一个全局归一化，而是：

- **对每一行单独做归一化**

也就是说，如果输入是：

```text
X[rows, cols]
```

那么：

- 第 0 行自己算自己的统计量
- 第 1 行自己算自己的统计量
- 不同行之间互不干扰

## 这两个 kernel 在算什么

### LayerNorm

先假设一行输入是：

```text
x = [x_1, x_2, ..., x_N]
```

`layernorm` 分三步看最清楚。

### 第一步：算均值

```text
mean = (1 / N) * sum_i x_i
```

意思是：

- 先看这一行的平均值是多少

### 第二步：算方差

```text
var = (1 / N) * sum_i (x_i - mean)^2
```

意思是：

- 看这一行里的数，围绕平均值波动得有多大

从实现角度，它也可以写成：

```text
var = (1 / N) * sum_i x_i^2 - mean^2
```

这两种写法是等价的。

这也是当前这版 kernel 采用的形式，因为它只需要两次 row-wise reduction：

- 一次 `sum(x)`
- 一次 `sum(x^2)`

### 第三步：归一化

```text
y_i = (x_i - mean) / sqrt(var + eps)
```

这里：

- `x_i - mean`
  - 先把这一行整体平移，让平均值变成 0
- `sqrt(var + eps)`
  - 再按标准差缩放
- `eps`
  - 一个很小的正数，用来避免分母太小甚至变成 0

所以 `layernorm` 的核心动作是：

1. 减均值
2. 除标准差

### RMSNorm

`rmsnorm` 更简单。

它不去减均值，而是只看这一行的均方根大小：

```text
rms = sqrt((1 / N) * sum_i x_i^2 + eps)
y_i = x_i / rms
```

这里最关键的一点是：

- **没有 `x_i - mean` 这一步**

和 `layernorm` 相比，`rmsnorm` 没有减去均值，所以它只需要：

- 一次 `sum(x^2)` reduction

这就是它在实现上更轻一点的根本原因。

你也可以把两者粗糙对比成：

- `layernorm`
  - 既管中心位置，也管尺度
- `rmsnorm`
  - 主要只管尺度，不主动把中心移到 0

## 把两个公式并排看

### LayerNorm

```text
mean = (1 / N) * sum_i x_i
var = (1 / N) * sum_i (x_i - mean)^2
y_i = (x_i - mean) / sqrt(var + eps)
```

### RMSNorm

```text
rms = sqrt((1 / N) * sum_i x_i^2 + eps)
y_i = x_i / rms
```

只看公式，最关键的差别就是：

- `layernorm` 需要 `mean`
- `rmsnorm` 不需要 `mean`

这会直接反映到 kernel 结构里：

- `layernorm` 要处理 `sum(x)` 和 `sum(x^2)`
- `rmsnorm` 只需要处理 `sum(x^2)`

## 当前 kernel 结构

这两份代码都采用最朴素的教学结构：

- 一个 `block` 负责一整行
- 一个 `thread` 负责这一行里的一个或多个元素
- 先做 block 内 reduction
- 再回写归一化结果

所以它们的气质和 `softmax/` 很接近：

- 都是典型的 row-wise reduction kernel
- 都是 memory-bound 倾向明显的算子
- 都依赖 shared memory 和同步

## 编译

```bash
make
```

## 运行

```bash
./row_layernorm
./row_rmsnorm
```

更完整的 profile 结果见 [ncu_notes.md](/data/home/tianjianyang/code/aisys-map/experiments/cuda_kernels/layernorm/ncu_notes.md)。

## 现在最值得记住的区别

1. `layernorm` 需要均值和方差，所以要处理 `sum(x)` 和 `sum(x^2)`。
2. `rmsnorm` 只关心均方根，不做中心化，所以更接近“只算尺度，不改中心”。
3. 这两类 kernel 都比 GEMM 更接近很多模型里真实高频出现的 memory-bound 算子。
