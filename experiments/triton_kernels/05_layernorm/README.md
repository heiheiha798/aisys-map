# 05 LayerNorm and RMSNorm in Triton

这个目录对应两个脚本：

- `row_layernorm.py`
- `row_rmsnorm.py`

默认你已经看过 `01` 到 `04`。
所以这里不再重复解释：

- `tl.max`
- `tl.sum`
- `tl.exp`

这一节只补 LayerNorm / RMSNorm 里新增的 Triton 语法。

## 这两个 kernel 在算什么

### LayerNorm

```text
mean = sum(x) / N
var  = sum(x^2) / N - mean^2
y    = (x - mean) / sqrt(var + eps)
```

### RMSNorm

```text
mean_sq = sum(x^2) / N
y       = x / sqrt(mean_sq + eps)
```

两者共同点都是：

- 一行一行做 reduction
- 再做逐元素归一化

## 结合代码看新增部分

先看 LayerNorm 的关键几行：

```python
x = tl.load(row_ptr, mask=mask, other=0.0).to(tl.float32)

row_sum = tl.sum(x, axis=0)
row_sq_sum = tl.sum(x * x, axis=0)

mean = row_sum / cols
mean_sq = row_sq_sum / cols
var = tl.maximum(mean_sq - mean * mean, 0.0)
inv_std = tl.rsqrt(var + eps)
```

RMSNorm 的关键几行是：

```python
x = tl.load(row_ptr, mask=mask, other=0.0).to(tl.float32)
row_sq_sum = tl.sum(x * x, axis=0)
mean_sq = row_sq_sum / cols
inv_rms = tl.rsqrt(mean_sq + eps)
```

## 新增语法 1：`.to(tl.float32)`

这里第一次系统性出现：

```python
x = tl.load(...).to(tl.float32)
```

虽然前面在 `scatter` 里见过 `.to(...)`，但那里是整数地址类型转换。

这里的重点是：

- 把参与 reduction 的数据显式转成 `float32`

为什么这么做？

因为 reduction 对数值精度更敏感。

即便原始数据以后换成低精度类型，累加时通常也更希望：

- 在更高精度里做统计

## 新增语法 2：`tl.maximum`

```python
var = tl.maximum(mean_sq - mean * mean, 0.0)
```

`tl.maximum(a, b)` 表示逐元素取较大值。

这里的作用是：

- 防止因为浮点误差导致 `var` 出现一个很小的负数

也就是先做一个数值保护：

```text
var = max(var, 0.0)
```

## 新增语法 3：`tl.rsqrt`

```python
inv_std = tl.rsqrt(var + eps)
```

`tl.rsqrt(x)` 的意思是：

```text
1 / sqrt(x)
```

它在 norm 类 kernel 里非常常见，因为最后都要算：

- `1 / sqrt(var + eps)`
- `1 / sqrt(mean_sq + eps)`

所以和先 `sqrt` 再除相比，直接写 `rsqrt` 更自然。

## 这份代码里新增的 Triton 语法

相对前面目录，这里新增的是：

- `.to(tl.float32)`
  - 把 reduction 放到更高精度里做
- `tl.maximum(a, b)`
  - 逐元素取最大值
- `tl.rsqrt(x)`
  - 倒平方根

## 运行

```bash
python row_layernorm.py
python row_rmsnorm.py
```

## 现在最值得记住的点

1. norm 类 kernel 的重点不只是 reduction，还有数值稳定。
2. `float32` 累加、`maximum` 截断、`rsqrt` 都是这类代码里很常见的固定套路。
3. 从语法上看，Triton 仍然是在“向量 load + reduction + elementwise normalize”这个框架里工作。
