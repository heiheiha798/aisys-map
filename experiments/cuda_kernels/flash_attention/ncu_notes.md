# NCU Notes: FlashAttention

这份笔记记录当前目录里的最小 `FlashAttention-2` 风格教学版 kernel：

- `flash_attention_kernel`

这里的定位需要先说清楚：

- 这是一个“帮助理解 FA2 工作划分”的学习用例

所以这份 `ncu` 笔记的重点是说明：

- 一个教学版 FlashAttention kernel 为什么不应被当成工程性能代表

## Profiling 命令

```bash
/usr/local/cuda-12.4/bin/ncu \
  --target-processes all \
  --set full \
  --launch-count 1 \
  --kernel-name flash_attention_kernel \
  ./flash_attention
```

## 当前样本

当前程序使用：

- `seq_len = 512`
- `head_dim = 32`
- `threads_per_block = 64`
- `row_tile = 2`
- `col_tile = 64`

当前实现表达的是：

- block 共享同一个 `K/V tile`
- 不同 warp 负责不同的 query rows
- 每个 warp 维护自己负责那一行的 online softmax 状态

也就是：

- 它主要在表达 `sliced-Q`

## NCU 关键指标

| binary | kernel | dur(us) | mem % | compute % | l1/tex % | l2 % | occ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `flash_attention` | `flash_attention_kernel` | `149.54` | `15.66` | `15.66` | `15.80` | `2.48` | `8.33` |

## 和普通 `attention` 的同参对比

同样是 `seq_len = 512, head_dim = 32`：

| kernel | grid | block | dur | mem % | compute % | achieved occ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `attention_kernel` | `512` | `32` | `49.82 us` | `64.13` | `12.91` | `8.34` |
| `flash_attention_kernel` | `256` | `64` | `149.54 us` | `15.66` | `15.66` | `8.33` |

这个结果更准确的理解是：

- 这组参数本身很偏教学规模
- `head_dim = 32` 很小
- 当前实现也远不是生产级 FlashAttention kernel

所以这个目录里的 `.cu` 文件更适合用来理解工作划分，而不适合用来做算法输赢结论。

## 为什么教学版不代表工程性能

主要有三点。

### 1. 高性能 FlashAttention 依赖大量实现细节

真正高性能的 FlashAttention kernel 通常会依赖：

- 更细的寄存器 blocking
- 更强的向量化 load/store
- 更复杂的 tile 映射
- 更精细的 shared memory 布局
- 更接近硬件特性的特化

这些东西一旦补齐，代码会迅速失去教学可读性。

### 2. 当前问题规模本身不一定最适合放大 FA 优势

这里固定的是：

- `seq_len = 512`
- `head_dim = 32`

在这种比较小的 head 维度下，vanilla attention 的显式 `scores` 成本并没有被放大到非常夸张的程度。  
于是教学版 FlashAttention 里：

- online softmax 的额外状态维护
- tile 管理
- 更复杂的控制流

这些成本反而更容易显出来。

### 3. 这个目录的目标不是性能榜

这个目录最重要的是帮助读者看懂：

- 为什么 FA1 的 `sliced-K` 会带来 warp 间通信
- 为什么 FA2 的 `sliced-Q` 能减少同步和中间结果合并

只要这一点讲清楚，这个目录就已经达成目的。

## 这份 profile 最该记住什么

1. 这份 `flash_attention.cu` 是学习用例，不是性能样例。
2. 当前 profile 的价值主要在于说明教学版实现和生产级实现之间还有很大差距。
3. 这个目录真正值得保留的是 `README` 中对 FA2 的循环和工作划分解释，而不是把这份 `.cu` 文件当成性能样例。
