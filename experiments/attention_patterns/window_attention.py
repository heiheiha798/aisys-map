import math

import torch


torch.set_printoptions(precision=4, sci_mode=False)


def build_input(seq_len: int = 6, hidden_size: int = 8):
    token_ids = torch.arange(seq_len)
    embedding = torch.linspace(-1.0, 1.0, steps=seq_len * hidden_size).reshape(
        seq_len, hidden_size
    )
    x = embedding.clone()
    w_q = torch.linspace(-0.7, 0.9, steps=hidden_size * hidden_size).reshape(
        hidden_size, hidden_size
    )
    w_k = torch.linspace(0.8, -0.6, steps=hidden_size * hidden_size).reshape(
        hidden_size, hidden_size
    )
    w_v = torch.linspace(-0.5, 0.7, steps=hidden_size * hidden_size).reshape(
        hidden_size, hidden_size
    )
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v
    return token_ids, x, q, k, v


def sliding_window_mask(seq_len: int, window_size: int):
    rows = torch.arange(seq_len).unsqueeze(1)
    cols = torch.arange(seq_len).unsqueeze(0)
    return (cols <= rows) & (cols >= rows - window_size + 1)


def main():
    window_size = 3
    token_ids, x, q, k, v = build_input()
    mask = sliding_window_mask(q.shape[0], window_size)
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = q @ k.transpose(0, 1) * scale
    masked_scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(masked_scores, dim=-1)
    out = probs @ v

    print("token_ids:")
    print(token_ids)
    print(f"\nSliding-window mask (window_size={window_size}):")
    print(mask.to(torch.int32))
    print("\nAttention probabilities:")
    print(probs)
    print("\nOutput:")
    print(out)
    print("\nLast-token visible keys:")
    print(mask[-1].to(torch.int32))


if __name__ == "__main__":
    main()
