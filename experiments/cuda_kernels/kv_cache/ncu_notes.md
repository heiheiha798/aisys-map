# NCU Notes: KV Cache Append / Update

这份笔记记录当前目录里的最小 `KV cache append / update` kernel：

- `kv_cache_append_update_kernel`

这个实验的重点是语义正确性，不是吞吐极限，所以 profile 也要结合样本规模来解读。

## Profiling 命令

append 第一次 launch：

```bash
/usr/local/cuda-12.4/bin/ncu \
  --target-processes all \
  --set full \
  --launch-count 1 \
  --kernel-name kv_cache_append_update_kernel \
  ./kv_cache_append_update
```

update 第二次 launch：

```bash
/usr/local/cuda-12.4/bin/ncu \
  --target-processes all \
  --set full \
  --launch-skip 1 \
  --launch-count 1 \
  --kernel-name kv_cache_append_update_kernel \
  ./kv_cache_append_update
```

## 当前样本

当前程序使用：

- `num_tokens = 8`
- `num_heads = 2`
- `max_seq_len = 6`
- `head_dim = 16`
- `threads_per_block = 128`
- `append_ops = 6`
- `update_ops = 2`

也就是：

- append 阶段只启动 `6` 个 block
- update 阶段只启动 `2` 个 block

## NCU 关键指标

| phase | grid | dur(us) | mem % | compute % | l1/tex % | l2 % | occ % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `append` | `6` | `2.62` | `0.47` | `0.03` | `6.42` | `0.47` | `5.67` |
| `update` | `2` | `2.56` | `0.45` | `0.01` | `18.98` | `0.45` | `6.05` |

## 最直接的结论

- 这份 profile 基本不能被解读成“KV cache kernel 的真实性能上限”
- 它更像是在 profile 一个极小样本下的教学 demo

原因非常直接：

- append 只有 `6` 个 block
- update 只有 `2` 个 block
- 机器上有 `128` 个 SM

因此这次结果的主导因素不是算法本身，而是：

- grid 极小
- 启动开销和调度空转占比很高

## 这次 profile 真正有价值的信息

虽然这次数据很小，但仍然能看出这版 kernel 的基本性质：

- 它本质是写 cache slot 的小型 copy/store kernel
- 不存在重计算
- 主要成本来自 very small launch 下的访存等待和调度空转

两个 phase 都出现了类似信号：

- `No Eligible ≈ 97.9%`
- `active warps per scheduler ≈ 1`
- `L1TEX scoreboard dependency` 和 `IMC miss` 都很高

这说明：

- warp 几乎一直在等
- 而且不是因为计算太复杂
- 只是工作量太小，GPU 根本没有被有效利用起来

## append 和 update 的差别怎么看

从这次结果看：

- append 和 update 的 `duration` 都在 `2.6 us` 左右
- 它们的画像也几乎一样

这是符合预期的，因为当前教学版里两者共享同一个 kernel：

- 都是在做 `cache[head, slot, d] = src[token_id, head, d]`
- 区别只在操作列表不同

所以在这份最小实现里，append / update 的 profile 差别远小于“样本过小”带来的噪声。

## 这份实验最该记住的结论

1. 这次 `ncu` 更像是验证“这确实是个非常轻的小写入 kernel”，不是验证 KV cache 写入的最终性能。
2. 对这种样本规模，`grid size` 太小，occupancy 太低，几乎所有性能结论都会被 launch 规模主导。
3. 如果后面真的要研究 KV cache append / update 的性能，至少要先把：
   - token 数量
   - head 数量
   - head_dim
   - 操作批量
   明显拉大，否则 `ncu` 的信息密度会很低。

