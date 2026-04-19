# 06 RoPE Forward in Triton

这个目录对应：

- `triton_rope.py`

默认你已经看过 `01` 到 `05`。
这里不再重复解释前面出现过的：

- `tl.arange`
- `tl.load`
- `tl.store`
- `.to(tl.float32)`

这一节只讲 RoPE 里新增的 Triton 语法。

## 这个 kernel 在算什么

RoPE 把相邻两维看成一个二维向量：

```text
(x_0, x_1), (x_2, x_3), ...
```

然后按 token 位置做旋转：

```text
y_even = x_even * cos(theta) - x_odd * sin(theta)
y_odd  = x_even * sin(theta) + x_odd * cos(theta)
```

## 结合代码看执行流程

先看 kernel 里的这段：

```python
row = tl.program_id(0)
token_idx = row // num_heads
pair_idx = tl.arange(0, BLOCK_D // 2)
even_col = 2 * pair_idx
odd_col = even_col + 1
```

这里说明：

- 一个 program 处理一个 `(token, head)` 行
- 这一行里按二维 pair 处理

然后看数学部分：

```python
exponent = (2.0 * pair_idx.to(tl.float32)) / head_dim
theta = token_idx * tl.exp(-tl.log(base) * exponent)
cos_theta = tl.cos(theta)
sin_theta = tl.sin(theta)
```

最后是旋转：

```python
y0 = x0 * cos_theta - x1 * sin_theta
y1 = x0 * sin_theta + x1 * cos_theta
```

## 新增语法 1：`tl.log` / `tl.cos` / `tl.sin`

这里第一次集中出现了三角和对数函数：

```python
tl.log(base)
tl.cos(theta)
tl.sin(theta)
```

它们和前面见过的 `tl.exp` 一样，都是 Triton 的逐元素数学函数。

可以先把它们理解成：

- 输入是一个标量或向量
- 输出对每个元素分别做同样的数学运算

## 新增语法 2：pair 级索引构造

这一段虽然不是新的 API，但这种写法是第一次出现：

```python
pair_idx = tl.arange(0, BLOCK_D // 2)
even_col = 2 * pair_idx
odd_col = even_col + 1
```

它的意义是：

- 先在逻辑上枚举第几个二维 pair
- 再从 pair 下标推回实际的偶数列和奇数列

后面只要看到这种“先构造逻辑索引，再映射回真实列号”的写法，就要意识到：

- 代码不一定直接按原始列坐标思考
- 也可能先按更适合数学结构的逻辑单位思考

这里的逻辑单位就是：

- 一个二维 pair

## 新增语法 3：整数除法参与索引映射

```python
token_idx = row // num_heads
```

这行也值得单独记一下。

因为它说明 Triton kernel 里不只是“直接拿 program_id 当行号”，也经常会做：

- 除法
- 取模
- 多维下标还原

也就是：

- 先把多维张量压平
- 再在 kernel 里恢复逻辑坐标

这在后面的更复杂 kernel 里会经常出现。

## 这份代码里新增的 Triton 语法

相对前面目录，这里新增的是：

- `tl.log`
- `tl.sin`
- `tl.cos`
- 先构造逻辑索引，再映射回真实列索引
- 用整数除法从一维 program id 恢复逻辑坐标

## 运行

```bash
python triton_rope.py
```

## 现在最值得记住的点

1. RoPE 让你第一次看到 Triton 里比较明显的“数学公式直译”写法。
2. 这类代码经常会先按逻辑单位建索引，比如 pair，而不是直接按原始列号思考。
3. Triton kernel 不只是能做 load/store/reduction，也很适合表达这种逐元素数学变换。
