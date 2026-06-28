# NCU Notes: elementwise_add

## 命令

```bash
/usr/local/cuda-12.4/bin/ncu \
  --target-processes all \
  --set full \
  --kernel-name elementwise_add_kernel \
  ./elementwise_add
```

## 结论

这个 kernel 是明显的 `memory-bound`。

## 关键依据

### 1. Memory Throughput 明显高于 Compute Throughput

- `Memory Throughput`: `79.78%`
- `Compute (SM) Throughput`: `8.66%`

这说明瓶颈更接近内存子系统，而不是计算单元。

### 2. NCU 直接给出了 Memory 优先的判断

`ncu` 的提示是：

- `Memory is more heavily utilized than Compute`

这已经是非常直接的信号。

### 3. 这个 kernel 本身的计算量非常小

核心逻辑只是：

```cpp
c[idx] = a[idx] + b[idx];
```

每个元素只做很少的计算，但需要：

- 读取 `a[idx]`
- 读取 `b[idx]`
- 写回 `c[idx]`

这意味着：

- 数据搬运多
- 数据复用少
- arithmetic intensity 低

这是典型的 elementwise kernel 形态。

### 4. Warp Stall 主要在等 memory

profile 里最重要的一条 stall 信息是：

- warp 大部分周期在等待 `L1TEX` 相关 scoreboard dependency

说明线程大量时间花在等数据返回，而不是在做计算。

## 关键指标摘录

- `Duration`: `10.75 us`
- `Memory Throughput`: `780.44 GB/s`
- `Memory Throughput %`: `79.78%`
- `Compute (SM) Throughput %`: `8.66%`
- `L2 Hit Rate`: `33.72%`
- `Achieved Occupancy`: `83.06%`

## 为什么 occupancy 不是主问题

这个 kernel 的 occupancy 不低：

- `Theoretical Occupancy`: `100%`
- `Achieved Occupancy`: `83.06%`

所以这里不是典型的“并发不够导致算不满”。  
更关键的问题是：

- 每次访存之后可做的计算太少
- kernel 的本质工作就是搬数据加做一点点加法

## 这次实验最该记住的结论

`elementwise` kernel 往往天然容易 `memory-bound`。

因为它通常具有这些特征：

- 每个元素计算量小
- 数据复用少
- 主要成本在读写内存

所以后面看到很多小型 elementwise kernel 时，第一直觉就应该是：

- 它大概率不是 compute-bound
- 真正该关注的是访存模式、带宽利用和 kernel fusion
