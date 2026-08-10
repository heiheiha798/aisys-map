# Attention Patterns

这个目录用四个独立脚本比较 dense、window、sparse 和 linear attention 的计算组织方式。

概念背景：

- Dense、causal mask 与 sliding window：[Attention 学习笔记 00](../../notes/attention/00-from-vanilla-attention-to-kda.md)
- Feature map、prefix state 与 linear attention：[Attention 学习笔记 01](../../notes/attention/01-linear-attention-background.md)

## 当前脚本

| 文件 | 计算路径 | 运行时重点观察 |
|---|---|---|
| `dense_attention.py` | 每个 query 访问全部可见历史 | 下三角 causal mask 和最后一行 scores |
| `window_attention.py` | 每个 query 访问最近 3 个位置 | sliding-window mask 和最后一个 token 的可见 keys |
| `sparse_attention.py` | 3-token local window 加全局位置 0、2 | local/global mask 形成的稀疏连接 |
| `linear_attention.py` | `ELU(x)+1` feature map 加两个 prefix states | 无完整 causal `QK^T` 时的在线状态更新 |

四个脚本均使用：

```text
sequence length = 6
hidden size = 8
```

其中 dense、window、sparse 仍显式计算选定 query-key 连接；linear 脚本维护：

```text
kv_prefix = sum(phi(k_i) outer v_i)
k_prefix  = sum(phi(k_i))
```

第 `t` 步只读取已经写入的 `0..t` 状态，因此递推顺序提供了因果性。

## 运行

```bash
python 01_model_basics/attention_patterns/dense_attention.py
python 01_model_basics/attention_patterns/window_attention.py
python 01_model_basics/attention_patterns/sparse_attention.py
python 01_model_basics/attention_patterns/linear_attention.py
```

使用仓库的 `aisys` 环境时：

```bash
conda run -n aisys python 01_model_basics/attention_patterns/dense_attention.py
conda run -n aisys python 01_model_basics/attention_patterns/window_attention.py
conda run -n aisys python 01_model_basics/attention_patterns/sparse_attention.py
conda run -n aisys python 01_model_basics/attention_patterns/linear_attention.py
```

这些实现使用 toy input 和 eager PyTorch，负责展示逻辑路径。工程级 sparse/linear kernel、长上下文 benchmark 和真实模型兼容属于后续实验范围。
