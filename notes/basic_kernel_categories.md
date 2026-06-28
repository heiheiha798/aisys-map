# Basic Kernel Categories

这份笔记只保留读 `02_kernel_intro/` 和 `05_case_studies/flash-deepseek-v2-lite/` 时需要的 kernel 分类直觉。

## 分类表

| 类型 | 典型操作 | 主要矛盾 | 常见实验位置 |
|---|---|---|---|
| elementwise | activation、bias add、residual add | 算得少、搬得多，常 memory-bound | `02_kernel_intro/*/01_elementwise` |
| reduction | sum/max/mean、softmax、layernorm | 多线程协作、同步、分阶段汇总 | `04_softmax`、`05_layernorm`、`07_online_softmax` |
| GEMM | linear、projection、MLP、attention matmul | 高复用、高吞吐、tile / Tensor Core 路径 | `11_gemm`、`03_kernel_advanced/SGEMM_CUDA` |
| indexed / gather-scatter | embedding、scatter add、token dispatch | 不规则访存、coalescing 差、load imbalance | `02_scatter`、`03_embedding`、MoE case study |
| fused attention | attention score + softmax + value aggregation | 重排 memory access，减少中间写回 | `09_attention`、`12_flash_attention` |

## 按瓶颈看

| 更容易卡住的地方 | 常见类型 |
|---|---|
| memory bandwidth | elementwise、gather/scatter、简单 reduction |
| compute throughput | GEMM、高复用 tensor contraction |
| synchronization | reduction、scan、block-level cooperative kernel |
| irregular memory access | indexed、sparse、MoE routing |
| runtime overhead | 大量小 kernel、unfused operator chain |

## 读一个 kernel 先问

1. 它更像哪一类？
2. 它主要是在算，还是在搬？
3. 它是否需要线程协作或同步？
4. 它的访存是连续、tile 化，还是索引驱动？
5. 当前优化是在提高复用、减少写回、减少同步，还是减少 launch/runtime overhead？

## 最小结论

- elementwise 并行容易，但常常 memory-bound。
- reduction 的难点是协作和同步。
- GEMM 值得做深度优化，因为复用高、算量大。
- indexed kernel 的难点通常不是算，而是不规则访存。
- fused attention 的核心是重构 attention 的 memory access 和 reduction 路径。
