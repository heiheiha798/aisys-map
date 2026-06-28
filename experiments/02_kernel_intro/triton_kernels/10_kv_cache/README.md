# 10 Triton KV Cache Append and Update

这个目录对应：

- `kv_cache_append_update.py`

默认你已经看过 `01` 到 `09`。
这里不再重复解释：

- 二维 `program_id`
- 复合 mask
- `.to(tl.int64)`

这一节只讲 KV cache 这个例子里第一次比较明确出现的增量写法。

## 这个 kernel 在算什么

当前 step 产生新的 `K/V` 之后，要把它们写进 cache：

```text
K_cache[head, slot, d] = K_src[token_id, head, d]
V_cache[head, slot, d] = V_src[token_id, head, d]
```

如果 `slot` 是新位置，这就是 append。
如果 `slot` 已经存在，这就是 update。

## 结合代码看执行流程

先看 kernel 里怎么拿元数据：

```python
head = tl.load(op_heads_ptr + op_idx, mask=valid_op, other=0).to(tl.int64)
slot = tl.load(op_slots_ptr + op_idx, mask=valid_op, other=0).to(tl.int64)
token_id = tl.load(op_token_ids_ptr + op_idx, mask=valid_op, other=0).to(tl.int64)
```

也就是：

- 一个 program 先读出这次写操作的三元组
- `(head, slot, token_id)` 再决定真正的源地址和目标地址

接着看地址计算：

```python
src_base = (token_id * num_heads + head) * head_dim
cache_base = (head * max_seq_len + slot) * head_dim
```

最后再把 `d_offsets` 这段列块加上去：

```python
k_vals = tl.load(k_src_ptr + src_base + d_offsets, ...)
tl.store(k_cache_ptr + cache_base + d_offsets, k_vals, ...)
```

## 新增语法 1：先读元数据，再决定地址

这不是一个新的 API，但这是第一次非常清楚地看到这种模式：

1. 先从一个元数据数组里读逻辑信息
2. 再根据逻辑信息推导真实地址

前面的 gather 也有一点这个意思，但这里更完整，因为要同时处理：

- `head`
- `slot`
- `token_id`

也就是多级索引恢复。

## 新增语法 2：把多维逻辑下标手动压平成线性地址

这一段是本目录最值得记住的代码：

```python
src_base = (token_id * num_heads + head) * head_dim
cache_base = (head * max_seq_len + slot) * head_dim
```

它说明 Triton kernel 里很常见的一种工作是：

- 自己手动把多维逻辑坐标转成一维线性地址

这和只靠 `row * stride + offsets` 相比更进一步，因为这里的张量逻辑维度更多了。

## 新增语法 3：host 侧同一 kernel 多次 launch

这个文件里 host 侧故意做了：

```python
launch_kv_cache_append_update(...)
torch.cuda.synchronize()

launch_kv_cache_append_update(...)
torch.cuda.synchronize()
```

也就是：

- 先 append
- 再 update

前面 `08_fused_rmsnorm` 也有多个 kernel，但这里是：

- 同一个 kernel
- 配不同的数据
- 分两个 phase 重复 launch

这也是 Triton 代码里很常见的工程写法。

## 这份代码里新增的 Triton 代码模式

相对前面目录，这里新增的是：

- 先读取操作元数据，再决定真正地址
- 手动把多维逻辑坐标压平成线性地址
- 同一个 Triton kernel 在 host 侧按不同 phase 多次 launch

## 运行

```bash
python kv_cache_append_update.py
```

## 现在最值得记住的点

1. KV cache 这类 kernel 的重点通常不在复杂算术，而在地址映射。
2. 这里最值得学的是“怎么从逻辑三元组推导真实内存地址”。
3. Triton 代码里经常不是一次 launch 解决全部问题，而是 host 侧分 phase 组织执行。
