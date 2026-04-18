# NCU Notes: LayerNorm and RMSNorm

这份笔记只记录当前目录里的两条归一化 kernel：

- `row_layernorm`
- `row_rmsnorm`

关注点主要有两个：

- `layernorm` 和 `rmsnorm` 在 kernel 结构上差在哪
- 它们在 profile 画像上有没有明显区别

## Profiling 命令

```bash
/usr/local/cuda-12.4/bin/ncu \
  --target-processes all \
  --set full \
  --launch-count 1 \
  --kernel-name <kernel_name> \
  ./<binary>
```

这里使用：

- `row_layernorm` 对应 `row_layernorm_kernel`
- `row_rmsnorm` 对应 `row_rmsnorm_kernel`

## 当前样本

这两份程序当前都使用：

- `rows = 1024`
- `cols = 256`
- `threads_per_block = 256`

也就是：

- 一个 block 处理一整行
- 一个线程基本对应一个列位置

## NCU 关键指标

| binary | kernel | dur(us) | mem % | compute % | l1/tex % | l2 % | occ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `row_layernorm` | `row_layernorm_kernel` | `6.53` | `50.06` | `50.06` | `67.94` | `9.27` | `79.40` |
| `row_rmsnorm` | `row_rmsnorm_kernel` | `4.61` | `38.44` | `38.44` | `58.06` | `13.23` | `76.33` |

## 先看最直接的结论

- `row_rmsnorm` 比 `row_layernorm` 更快
- 这和公式本身的复杂度差异是一致的

原因不神秘：

- `layernorm` 需要 `sum(x)` 和 `sum(x^2)`
- `rmsnorm` 只需要 `sum(x^2)`

换句话说：

- `rmsnorm` 少了一次均值相关统计，也少了一步减均值

## LayerNorm 画像

`row_layernorm_kernel` 当前的关键指标是：

- `duration = 6.53 us`
- `mem % = 50.06`
- `compute % = 50.06`
- `l1/tex % = 67.94`
- `occ % = 79.40`

这个画像说明：

- 它不是纯 compute-bound
- 也不是像 GEMM 那样明显靠某个计算管线吃满
- 更像一个带 shared-memory reduction 的高频 memory-bound / latency-sensitive kernel

还可以看到：

- `Avg. Not Predicated Off Threads Per Warp ≈ 21.10`

这说明：

- 即使一个 warp 有 32 个线程，真正有效工作的线程数也会因为循环尾部和谓词执行而下降

## RMSNorm 画像

`row_rmsnorm_kernel` 当前的关键指标是：

- `duration = 4.61 us`
- `mem % = 38.44`
- `compute % = 38.44`
- `l1/tex % = 58.06`
- `occ % = 76.33`

和 `layernorm` 对照时，最值得记住的是：

- 它更快，但不代表它是“更偏 compute”
- 更准确的说法是：
  - 它做的统计更少
  - 指令更少
  - shared-memory reduction 也更轻

所以整体 latency 更低。

`ncu` 里也能看到一个典型信号：

- `L1TEX scoreboard dependency` 仍然是重要 stall 来源

这和很多 row-wise normalization kernel 的气质一致：

- 算子本身不大
- 但很容易被访存和依赖链限制

## 两者并排怎么理解

如果只从数学公式看：

- `layernorm`
  - 要处理中心化和尺度
- `rmsnorm`
  - 只处理尺度

如果只从 kernel 结构看：

- `layernorm`
  - 两次 row-wise reduction
  - 先算 `sum(x)`，再算 `sum(x^2)`
- `rmsnorm`
  - 一次 row-wise reduction
  - 只算 `sum(x^2)`

如果只从 `ncu` 看：

- 两者都不是“大算力型” kernel
- 两者都更接近 memory / latency 敏感的小型归一化算子
- `rmsnorm` 的路径更短，所以持续更快

## 这份实验最该记住的结论

1. `layernorm` 和 `rmsnorm` 的最本质区别，不是名字，而是 `rmsnorm` 不做减均值。
2. 对应到 kernel 上，这个差别会直接变成 reduction 次数和指令数的差异。
3. 这类 kernel 比 GEMM 更接近很多模型里真实高频出现的 memory-bound 小算子。
