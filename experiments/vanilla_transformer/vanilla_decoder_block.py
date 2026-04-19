import math

import torch


torch.set_printoptions(precision=4, sci_mode=False, linewidth=140)


SEQ_LEN = 4
HIDDEN_SIZE = 4
FFN_HIDDEN_SIZE = 8
VOCAB_SIZE = 8


def make_embedding_table() -> torch.Tensor:
    return torch.tensor(
        [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 1, 4, 2],
            [5, 4, 3, 2],
            [1, 1, 2, 3],
            [1, 2, 2, 3],
            [2, 3, 1, 4],
            [3, 1, 4, 1],
        ],
        dtype=torch.float32,
    )


def make_linear_weight(out_dim: int, in_dim: int, base: int) -> torch.Tensor:
    values = []
    for i in range(out_dim):
        row = []
        for j in range(in_dim):
            row.append(((base + i + j) % 3) - 1)
        values.append(row)
    return torch.tensor(values, dtype=torch.float32)


def layernorm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def print_tensor(name: str, tensor: torch.Tensor, integer: bool = False) -> None:
    print(name)
    if integer:
        print(tensor.to(torch.int64))
    else:
        print(tensor)
    print("")


def main() -> None:
    token_ids = torch.tensor([1, 3, 5, 2], dtype=torch.long)
    embedding_table = make_embedding_table()

    w_q = make_linear_weight(HIDDEN_SIZE, HIDDEN_SIZE, 1)
    w_k = make_linear_weight(HIDDEN_SIZE, HIDDEN_SIZE, 2)
    w_v = make_linear_weight(HIDDEN_SIZE, HIDDEN_SIZE, 3)
    w_o = make_linear_weight(HIDDEN_SIZE, HIDDEN_SIZE, 4)
    w_ffn_1 = make_linear_weight(FFN_HIDDEN_SIZE, HIDDEN_SIZE, 5)
    w_ffn_2 = make_linear_weight(HIDDEN_SIZE, FFN_HIDDEN_SIZE, 6)

    x = embedding_table[token_ids]

    print_tensor("token ids:", token_ids, integer=True)
    print_tensor("embedding table:", embedding_table, integer=True)
    print(f"embedding output X: shape={tuple(x.shape)}")
    print(x.to(torch.int64))
    print("")

    print_tensor("W_q:", w_q, integer=True)
    print_tensor("W_k:", w_k, integer=True)
    print_tensor("W_v:", w_v, integer=True)

    q = x @ w_q.T
    k = x @ w_k.T
    v = x @ w_v.T

    print("Q / K / V:")
    print(f"Q shape={tuple(q.shape)}")
    print(q.to(torch.int64))
    print(f"K shape={tuple(k.shape)}")
    print(k.to(torch.int64))
    print(f"V shape={tuple(v.shape)}")
    print(v.to(torch.int64))
    print("")

    scale = 1.0 / math.sqrt(HIDDEN_SIZE)
    raw_scores = (q @ k.T) * scale

    print("raw attention scores:")
    print(f"shape={tuple(raw_scores.shape)}  # [seq_len, seq_len]")
    print(raw_scores)
    print("")

    causal_mask = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), dtype=torch.float32))
    masked_scores = raw_scores.masked_fill(causal_mask == 0, float("-inf"))

    print("causal mask:")
    print(f"shape={tuple(causal_mask.shape)}")
    print(causal_mask.to(torch.int64))
    print("")

    print("masked attention scores:")
    print(masked_scores)
    print("")

    probs = softmax(masked_scores, dim=-1)
    print("attention probabilities after softmax:")
    print(probs)
    print("")

    attn_out = probs @ v
    print("attention output = attention_probs @ V:")
    print(f"shape={tuple(attn_out.shape)}")
    print(attn_out)
    print("")

    attn_proj = attn_out @ w_o.T
    print("output projection after attention:")
    print(attn_proj)
    print("")

    x_after_attn = x + attn_proj
    print("after attention residual:")
    print(x_after_attn)
    print("")

    x_norm_1 = layernorm(x_after_attn)
    print("after first layernorm:")
    print(x_norm_1)
    print("")

    ffn_hidden = torch.relu(x_norm_1 @ w_ffn_1.T)
    print("FFN hidden after first linear + ReLU:")
    print(f"shape={tuple(ffn_hidden.shape)}")
    print(ffn_hidden)
    print("")

    ffn_out = ffn_hidden @ w_ffn_2.T
    print("FFN output:")
    print(ffn_out)
    print("")

    x_after_ffn = x_norm_1 + ffn_out
    print("after FFN residual:")
    print(x_after_ffn)
    print("")

    block_out = layernorm(x_after_ffn)
    print("final block output:")
    print(block_out)
    print("")

    print("===== Step-by-step for the 4th token (t3) =====")
    token_index = 3

    q3 = q[token_index]
    print(f"q3 shape={tuple(q3.shape)}")
    print(q3.to(torch.int64))
    print("")

    print("all key vectors K:")
    print(f"shape={tuple(k.shape)}")
    print(k.to(torch.int64))
    print("")

    dot_products = torch.mv(k, q3) * scale
    print("dot products between q3 and all keys:")
    print(dot_products)
    print("")

    print("mask row for token t3:")
    print(causal_mask[token_index].to(torch.int64))
    print("")

    masked_dot_products = dot_products.masked_fill(causal_mask[token_index] == 0, float("-inf"))
    print("masked dot products for token t3:")
    print(masked_dot_products)
    print("")

    probs_t3 = softmax(masked_dot_products.unsqueeze(0), dim=-1).squeeze(0)
    print("softmax probabilities for token t3:")
    print(probs_t3)
    print("")

    weighted_v = probs_t3.unsqueeze(0) @ v
    print("weighted sum over V for token t3:")
    print(f"shape={tuple(weighted_v.shape)}")
    print(weighted_v)
    print("")


if __name__ == "__main__":
    main()
