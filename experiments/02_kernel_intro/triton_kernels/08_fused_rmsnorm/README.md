# 08 Fused Residual + RMSNorm in Triton

这个目录对应：

- `fused_residual_rmsnorm.py`

默认你已经看过 `01` 到 `07`。
这里重点不再是新的数学，而是更复杂的 Triton 代码组织方式。

## 这个文件在算什么

目标仍然是：

```text
s = x + residual
mean_sq = mean(s^2)
y = gamma * s * rsqrt(mean_sq + eps)
```

但这版 Triton 代码没有把所有逻辑硬塞进一个 kernel。

它拆成了两个 Triton kernel：

1. `row_mean_sq_kernel`
2. `fused_residual_rmsnorm_kernel`

## 结合代码看执行流程

先看 host 侧：

```python
row_mean_sq_kernel[(rows,)](...)

grid = (rows * triton.cdiv(cols, block_size),)
fused_residual_rmsnorm_kernel[grid](...)
```

这说明：

- 同一个 Python 文件里可以定义多个 Triton kernel
- host 侧可以先后 launch 多个 kernel

这在前面目录里还没有正式出现过。

再看第二个 kernel 开头：

```python
pid = tl.program_id(axis=0)
row = pid // tl.cdiv(cols, BLOCK_SIZE)
block_col = pid % tl.cdiv(cols, BLOCK_SIZE)
```

这里说明第二个 kernel 不是直接用二维 grid，而是：

- 先把 `(row, block_col)` 压成一个一维 `pid`
- 再在 kernel 里用 `//` 和 `%` 还原

## 新增语法 1：一个文件里定义多个 Triton kernel

这是第一次正式出现：

```python
@triton.jit
def row_mean_sq_kernel(...):
    ...

@triton.jit
def fused_residual_rmsnorm_kernel(...):
    ...
```

也就是：

- Triton 不要求一个文件只能有一个 kernel
- 你完全可以把一个算子的不同阶段拆开写

## 新增语法 2：在 kernel 里使用 `tl.cdiv(...)`

这里有一行很值得记：

```python
row = pid // tl.cdiv(cols, BLOCK_SIZE)
block_col = pid % tl.cdiv(cols, BLOCK_SIZE)
```

前面你已经在 host 侧见过：

```python
triton.cdiv(...)
```

这里则是第一次在 Triton kernel 内看到：

```python
tl.cdiv(...)
```

它的作用仍然是向上取整除法，只是这次是在 device 侧参与逻辑坐标恢复。

## 新增语法 3：把二维逻辑坐标压成一维 `pid`

这里非常值得记：

```python
row = pid // tl.cdiv(cols, BLOCK_SIZE)
block_col = pid % tl.cdiv(cols, BLOCK_SIZE)
```

这意味着：

- launch 时只给了一维 grid
- 但逻辑上其实还是 `(row, block_col)` 二维

所以做法是：

1. 先把二维问题压成一维
2. 再在 kernel 内恢复坐标

这种写法在后面更复杂的 kernel 里很常见。

## 这份代码里新增的 Triton 语法

相对前面目录，这里新增的是：

- 一个文件里定义多个 Triton kernel
- host 侧顺序 launch 多个 kernel
- `tl.cdiv(...)`
  - 在 kernel 内参与逻辑坐标恢复
- 用 `//` 和 `%` 从一维 `pid` 还原二维逻辑坐标

## 运行

```bash
python fused_residual_rmsnorm.py
```

## 现在最值得记住的点

1. Triton 文件的组织方式可以很灵活，一个文件里完全可以有多个 kernel。
2. 复杂算子不一定非得一开始就写成单 kernel，有时先拆成几个最小阶段更容易看清数据流。
3. “一维 launch + kernel 内恢复二维逻辑坐标”是很常见的写法。
