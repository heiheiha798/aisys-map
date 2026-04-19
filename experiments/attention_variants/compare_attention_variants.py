import math

import torch


torch.set_printoptions(precision=4, sci_mode=False, linewidth=160)


SEQ_LEN = 4
HIDDEN_SIZE = 8


def make_embedding_table() -> torch.Tensor:
    return torch.tensor(
        [
            [1, 2, 3, 4, 1, 2, 3, 4],
            [2, 3, 4, 5, 2, 3, 4, 5],
            [3, 1, 4, 2, 3, 1, 4, 2],
            [5, 4, 3, 2, 5, 4, 3, 2],
            [1, 1, 2, 3, 1, 1, 2, 3],
            [1, 2, 2, 3, 1, 2, 2, 3],
            [2, 3, 1, 4, 2, 3, 1, 4],
            [3, 1, 4, 1, 3, 1, 4, 1],
        ],
        dtype=torch.float32,
    )


def make_linear_weight(out_dim: int, in_dim: int, base: int) -> torch.Tensor:
    rows = []
    for i in range(out_dim):
        row = []
        for j in range(in_dim):
            row.append(((base + i + j) % 3) - 1)
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def causal_mask(seq_len: int) -> torch.Tensor:
    return torch.tril(torch.ones((seq_len, seq_len), dtype=torch.float32))


def print_section(title: str) -> None:
    print("")
    print("=" * 20, title, "=" * 20)


def print_tensor(name: str, tensor: torch.Tensor, integer: bool = False) -> None:
    print(name)
    if integer:
        print(tensor.to(torch.int64))
    else:
        print(tensor)
    print("")


def split_heads(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    return x.view(SEQ_LEN, num_heads, head_dim).transpose(0, 1).contiguous()


def repeat_kv(kv_heads: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    num_kv_heads = kv_heads.shape[0]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads.")
    group_size = num_q_heads // num_kv_heads
    return kv_heads.repeat_interleave(group_size, dim=0)


def run_grouped_attention_variant(
    name: str,
    x: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    base_q: int,
    base_k: int,
    base_v: int,
) -> None:
    print_section(name)

    q_dim = num_q_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    w_q = make_linear_weight(q_dim, HIDDEN_SIZE, base_q)
    w_k = make_linear_weight(kv_dim, HIDDEN_SIZE, base_k)
    w_v = make_linear_weight(kv_dim, HIDDEN_SIZE, base_v)

    print(f"{name} config:")
    print(f"  num_q_heads={num_q_heads}")
    print(f"  num_kv_heads={num_kv_heads}")
    print(f"  head_dim={head_dim}")
    print(f"  q projection shape={tuple(w_q.shape)}")
    print(f"  k projection shape={tuple(w_k.shape)}")
    print(f"  v projection shape={tuple(w_v.shape)}")
    print("")

    q = x @ w_q.T
    k = x @ w_k.T
    v = x @ w_v.T

    print_tensor("W_q:", w_q, integer=True)
    print_tensor("W_k:", w_k, integer=True)
    print_tensor("W_v:", w_v, integer=True)
    print_tensor("Q before head split:", q, integer=True)
    print_tensor("K before head split:", k, integer=True)
    print_tensor("V before head split:", v, integer=True)

    q_heads = split_heads(q, num_q_heads, head_dim)
    k_heads = split_heads(k, num_kv_heads, head_dim)
    v_heads = split_heads(v, num_kv_heads, head_dim)
    k_for_attn = repeat_kv(k_heads, num_q_heads)
    v_for_attn = repeat_kv(v_heads, num_q_heads)

    print_tensor("Q heads:", q_heads, integer=True)
    print_tensor("KV heads before sharing:", k_heads, integer=True)
    print_tensor("K heads used by attention:", k_for_attn, integer=True)
    print_tensor("V heads used by attention:", v_for_attn, integer=True)

    score_scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q_heads, k_for_attn.transpose(-1, -2)) * score_scale
    mask = causal_mask(SEQ_LEN)
    masked_scores = scores.masked_fill(mask.unsqueeze(0) == 0, float("-inf"))
    probs = softmax(masked_scores, dim=-1)
    out_heads = torch.matmul(probs, v_for_attn)
    out = out_heads.transpose(0, 1).contiguous().view(SEQ_LEN, q_dim)

    print_tensor("raw attention scores:", scores)
    print_tensor("causal mask:", mask, integer=True)
    print_tensor("masked attention scores:", masked_scores)
    print_tensor("attention probabilities:", probs)
    print_tensor("attention output per head:", out_heads)
    print_tensor("attention output after concat:", out)

    token_index = 3
    head_index = min(1, num_q_heads - 1)
    mapped_kv_head = head_index if num_q_heads == num_kv_heads else head_index // (num_q_heads // num_kv_heads)
    print(f"{name} focus: token t3, q head {head_index}")
    print(f"  this q head uses kv head {mapped_kv_head}")
    print(f"  q_t3_head shape={tuple(q_heads[head_index, token_index].shape)}")
    print(q_heads[head_index, token_index].to(torch.int64))
    print("")
    print("  score row for token t3 on this head:")
    print(scores[head_index, token_index])
    print("")


def run_mla_variant(x: torch.Tensor) -> None:
    print_section("MLA")

    num_heads = 2
    head_dim = 4
    q_dim = num_heads * head_dim
    latent_dim = 3

    w_q = make_linear_weight(q_dim, HIDDEN_SIZE, 4)
    w_down_kv = make_linear_weight(latent_dim, HIDDEN_SIZE, 5)
    w_up_k = make_linear_weight(q_dim, latent_dim, 6)
    w_up_v = make_linear_weight(q_dim, latent_dim, 7)

    print("MLA config:")
    print(f"  num_heads={num_heads}")
    print(f"  head_dim={head_dim}")
    print(f"  latent_dim={latent_dim}")
    print("  teaching simplification:")
    print("    cache latent c_kv instead of full K/V")
    print("    then reconstruct K/V from c_kv")
    print("")

    q = x @ w_q.T
    c_kv = x @ w_down_kv.T
    k = c_kv @ w_up_k.T
    v = c_kv @ w_up_v.T

    print_tensor("W_q:", w_q, integer=True)
    print_tensor("W_down_kv:", w_down_kv, integer=True)
    print_tensor("W_up_k:", w_up_k, integer=True)
    print_tensor("W_up_v:", w_up_v, integer=True)
    print_tensor("Q:", q, integer=True)
    print_tensor("latent KV cache c_kv:", c_kv, integer=True)
    print_tensor("reconstructed K:", k, integer=True)
    print_tensor("reconstructed V:", v, integer=True)

    q_heads = split_heads(q, num_heads, head_dim)
    k_heads = split_heads(k, num_heads, head_dim)
    v_heads = split_heads(v, num_heads, head_dim)

    score_scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q_heads, k_heads.transpose(-1, -2)) * score_scale
    mask = causal_mask(SEQ_LEN)
    masked_scores = scores.masked_fill(mask.unsqueeze(0) == 0, float("-inf"))
    probs = softmax(masked_scores, dim=-1)
    out_heads = torch.matmul(probs, v_heads)
    out = out_heads.transpose(0, 1).contiguous().view(SEQ_LEN, q_dim)

    print("MLA cache comparison:")
    print(f"  full K cache logical shape would be: ({SEQ_LEN}, {q_dim})")
    print(f"  full V cache logical shape would be: ({SEQ_LEN}, {q_dim})")
    print(f"  latent cache shape is: {tuple(c_kv.shape)}")
    print("")

    print_tensor("raw attention scores:", scores)
    print_tensor("causal mask:", mask, integer=True)
    print_tensor("masked attention scores:", masked_scores)
    print_tensor("attention probabilities:", probs)
    print_tensor("attention output per head:", out_heads)
    print_tensor("attention output after concat:", out)

    token_index = 3
    head_index = 1
    print("MLA focus: token t3, head 1")
    print(q_heads[head_index, token_index].to(torch.int64))
    print("")
    print("score row for token t3 on this head:")
    print(scores[head_index, token_index])
    print("")


def main() -> None:
    token_ids = torch.tensor([1, 3, 5, 2], dtype=torch.long)
    embedding_table = make_embedding_table()
    x = embedding_table[token_ids]

    print_tensor("token ids:", token_ids, integer=True)
    print_tensor("embedding table:", embedding_table, integer=True)
    print_tensor("selected hidden states X:", x, integer=True)

    run_grouped_attention_variant(
        name="MHA",
        x=x,
        num_q_heads=2,
        num_kv_heads=2,
        head_dim=4,
        base_q=1,
        base_k=2,
        base_v=3,
    )

    run_grouped_attention_variant(
        name="MQA",
        x=x,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
        base_q=2,
        base_k=3,
        base_v=4,
    )

    run_grouped_attention_variant(
        name="GQA",
        x=x,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=2,
        base_q=3,
        base_k=4,
        base_v=5,
    )

    run_mla_variant(x)


if __name__ == "__main__":
    main()
