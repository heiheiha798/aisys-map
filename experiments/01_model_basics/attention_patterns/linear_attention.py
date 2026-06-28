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


def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    q_phi = torch.nn.functional.elu(q) + 1.0
    k_phi = torch.nn.functional.elu(k) + 1.0

    kv_prefix = torch.zeros(q.shape[-1], v.shape[-1])
    k_prefix = torch.zeros(q.shape[-1])
    outputs = []

    for i in range(q.shape[0]):
        kv_prefix = kv_prefix + torch.outer(k_phi[i], v[i])
        k_prefix = k_prefix + k_phi[i]
        denom = torch.dot(q_phi[i], k_prefix).clamp_min(1e-6)
        out_i = (q_phi[i] @ kv_prefix) / denom
        outputs.append(out_i)

    return q_phi, k_phi, torch.stack(outputs, dim=0)


def main():
    token_ids, x, q, k, v = build_input()
    q_phi, k_phi, out = linear_attention(q, k, v)

    print("token_ids:")
    print(token_ids)
    print("\nFeature-mapped Q:")
    print(q_phi)
    print("\nFeature-mapped K:")
    print(k_phi)
    print("\nOutput:")
    print(out)
    print("\nSummary:")
    print("- This path does not explicitly build a full causal QK^T matrix.")
    print("- It changes compute organization instead of pruning connectivity with a mask.")


if __name__ == "__main__":
    main()
