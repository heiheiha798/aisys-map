# 01 Elementwise Add in Triton

这个目录是整个 `triton_kernels/` 的起点。

文件只有一个：

- `elementwise_add.py`

它做的事情也最简单：

```text
out[i] = x[i] + y[i]
```

但正因为它足够简单，所以很适合第一次认识 Triton 代码长什么样。

## 先看 host 侧怎么启动 kernel

先看 `main()` 里最关键的几行：

```python
grid = (triton.cdiv(numel, block_size),)
elementwise_add_kernel[grid](
    x,
    y,
    out,
    numel,
    BLOCK_SIZE=block_size,
)
```

这里可以先记住两件事：

1. Triton kernel 仍然是“host 侧启动，device 侧执行”。
2. `kernel[grid](...)` 这种写法就是 Triton 的 launch 语法。

`triton.cdiv(numel, block_size)` 的意思是：

```text
ceil(numel / block_size)
```

也就是需要多少个 program 才能覆盖全部元素。

## 再看 kernel 本体

`elementwise_add_kernel` 的结构非常短：

```python
@triton.jit
def elementwise_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)
```

可以按下面的顺序理解。

### 第一步：`@triton.jit`

```python
@triton.jit
```

这表示下面这个 Python 函数不是普通 Python 函数，而是一个 Triton kernel。

你可以先把它粗略理解成：

- 这个函数会被 Triton 编译成 GPU kernel

## 第二步：`tl.program_id(0)`

```python
pid = tl.program_id(0)
```

这里的 `pid` 就是当前 program 的编号。

在这个例子里，grid 是一维的，所以只需要：

```text
program_id(0)
```

你可以把它先想成 CUDA 里的：

- “这是第几个 block”

但更准确一点的说法是：

- 这是第几个 Triton program

## 第三步：`tl.arange`

```python
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

`tl.arange(0, BLOCK_SIZE)` 会生成一段连续下标：

```text
[0, 1, 2, ..., BLOCK_SIZE - 1]
```

再加上 `pid * BLOCK_SIZE`，就变成当前 program 负责的那一段元素：

```text
program 0 -> [0, 1, 2, ...]
program 1 -> [BLOCK_SIZE, BLOCK_SIZE + 1, ...]
program 2 -> ...
```

这就是 Triton 里最常见的思路：

- 一个 program 处理一个连续 tile

## 第四步：`mask`

```python
mask = offsets < numel
```

最后一个 program 不一定正好填满 `BLOCK_SIZE`。

所以这里要用 `mask` 表示：

- 哪些位置是真的有效元素
- 哪些位置只是越界补位

## 第五步：`tl.load` / `tl.store`

```python
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
tl.store(out_ptr + offsets, out, mask=mask)
```

这里可以先把它理解成：

- `tl.load`
  - 从 global memory 读数据
- `tl.store`
  - 把结果写回 global memory

`other=0.0` 的意思是：

- 如果某个位置被 `mask` 判定为无效
- 那读出来的值就当作 `0.0`

## 第六步：`tl.constexpr`

```python
BLOCK_SIZE: tl.constexpr
```

这表示 `BLOCK_SIZE` 是编译期常量。

也就是 Triton 在编译 kernel 时就知道：

- 这次 block/tile 大小是多少

它不是运行过程中才动态决定的普通张量值。

这类参数通常会影响：

- `tl.arange` 的长度
- 编译出的循环展开和寄存器布局

## 这份代码里第一次出现的 Triton 语法

这个目录第一次出现了下面这些最基础语法：

- `@triton.jit`
  - 把 Python 函数变成 Triton kernel
- `tl.program_id(axis)`
  - 取当前 program 在某个维度上的编号
- `tl.arange(start, end)`
  - 生成一个连续下标向量
- `tl.load(...)`
  - 从显存读取
- `tl.store(...)`
  - 向显存写回
- `mask=...`
  - 做边界保护
- `other=...`
  - 对无效位置给默认值
- `tl.constexpr`
  - 编译期常量参数
- `kernel[grid](...)`
  - Triton kernel 的 launch 语法
- `triton.cdiv(a, b)`
  - 向上取整除法

后面的 README 默认你已经记住这些基础语法。
之后不会再反复从头解释，只讲新增部分。

## 运行

```bash
python elementwise_add.py
```

## 现在最值得记住的点

1. Triton 最基础的写法，就是“一个 program 处理一个连续 tile”。
2. `program_id + arange + mask + load/store` 是后面几乎所有 kernel 都会反复出现的基本骨架。
3. 从这个例子开始，你已经能看懂 Triton kernel 最基本的索引方式了。
