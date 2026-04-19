# 07 Triton Online Softmax

这个目录对应：

- `row_softmax_online.py`

默认你已经看过 `01` 到 `06`。
这一节只解释相对普通 softmax 新增的 Triton 写法。

## 这个 kernel 在算什么

普通 softmax 可以理解成：

- 先整行求 `max`
- 再整行求 `sum(exp(...))`

online softmax 则把一整行拆成多个 tile，边扫描边维护状态：

- `running_max`
- `running_sum`

## 结合代码看执行流程

最关键的部分是这段循环：

```python
running_max = -float("inf")
running_sum = 0.0

for tile_idx in range(NUM_TILES):
    start = tile_idx * BLOCK_SIZE
    cols_offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = cols_offsets < cols
    x = tl.load(x_row_ptr + cols_offsets, mask=mask, other=-float("inf"))

    tile_max = tl.max(x, axis=0)
    tile_sum = tl.sum(tl.exp(x - tile_max), axis=0)

    new_max = tl.maximum(running_max, tile_max)
    running_sum = (
        running_sum * tl.exp(running_max - new_max)
        + tile_sum * tl.exp(tile_max - new_max)
    )
    running_max = new_max
```

这说明：

- kernel 不是只处理一个 tile 就结束
- 而是在一个 program 内多次扫描不同 tile

## 新增语法 1：`for range(NUM_TILES)` 里的编译期循环

这里第一次出现：

```python
for tile_idx in range(NUM_TILES):
```

而 `NUM_TILES` 又是：

```python
NUM_TILES: tl.constexpr
```

这意味着：

- 循环次数在编译期就确定了

所以这不是普通 Python 解释执行的“动态循环”，而更接近：

- Triton 编译器已知循环上界的静态循环

这类写法在 tiled kernel 里很常见。

## 新增语法 2：program 内维护标量状态

这里的：

```python
running_max = -float("inf")
running_sum = 0.0
```

也值得单独记一下。

因为前面几个例子更多是：

- 读一段向量
- 立刻算完

而这里第一次明显出现：

- program 内自己维护跨 tile 的状态变量

也就是：

- 当前 program 扫了前几个 tile 之后，已经有自己的中间统计量

这正是 online softmax 的核心。

## 新增语法 3：两遍扫描的 kernel 结构

后面还有第二个循环：

```python
for tile_idx in range(NUM_TILES):
    ...
    y = tl.exp(x - running_max) / running_sum
    tl.store(...)
```

这说明一个 Triton kernel 也可以有很明显的多阶段结构：

1. 第一遍先统计
2. 第二遍再回写结果

这里虽然没有新增某个单独 API，但这是第一次出现这种：

- 在同一个 kernel 里分两个阶段做事

的结构。

## 这份代码里新增的 Triton 语法

相对前面目录，这里新增的是：

- `for range(NUM_TILES)` 这种编译期已知上界的 tile 循环
- program 内维护跨 tile 的标量状态
- 同一个 kernel 里分两遍扫描输入

## 运行

```bash
python row_softmax_online.py
```

## 现在最值得记住的点

1. online softmax 的关键不在某个新 API，而在“一个 program 能维护跨 tile 状态”。
2. Triton 不只适合单次读一块数据，也适合写这种分阶段扫描 kernel。
3. 这一步已经比普通 softmax 更接近 FlashAttention 的思路了。
