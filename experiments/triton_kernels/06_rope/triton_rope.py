import math

import torch
import triton
import triton.language as tl


BASE = 10000.0


@triton.jit
def rope_forward_kernel(
    x_ptr,
    y_ptr,
    seq_len,
    num_heads,
    head_dim,
    base,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    total_rows = seq_len * num_heads
    if row >= total_rows:
        return

    token_idx = row // num_heads
    pair_idx = tl.arange(0, BLOCK_D // 2)
    even_col = 2 * pair_idx
    odd_col = even_col + 1

    pair_mask = odd_col < head_dim
    row_base = row * head_dim

    x0 = tl.load(x_ptr + row_base + even_col, mask=pair_mask, other=0.0)
    x1 = tl.load(x_ptr + row_base + odd_col, mask=pair_mask, other=0.0)

    exponent = (2.0 * pair_idx.to(tl.float32)) / head_dim
    theta = token_idx * tl.exp(-tl.log(base) * exponent)
    cos_theta = tl.cos(theta)
    sin_theta = tl.sin(theta)

    y0 = x0 * cos_theta - x1 * sin_theta
    y1 = x0 * sin_theta + x1 * cos_theta

    tl.store(y_ptr + row_base + even_col, y0, mask=pair_mask)
    tl.store(y_ptr + row_base + odd_col, y1, mask=pair_mask)


def build_inputs(seq_len: int, num_heads: int, head_dim: int, device: torch.device):
    x = torch.empty((seq_len, num_heads, head_dim), device=device, dtype=torch.float32)
    for token_idx in range(seq_len):
        for head_idx in range(num_heads):
            for dim_idx in range(head_dim):
                a = math.sin((token_idx + 1) * 0.11)
                b = math.cos((head_idx + 2) * 0.07)
                c = (((dim_idx * 13 + token_idx * 5 + head_idx * 3) % 29) - 14) * 0.04
                x[token_idx, head_idx, dim_idx] = 0.5 * a + 0.35 * b + c
    return x


def cpu_rope_forward(x: torch.Tensor, base: float) -> torch.Tensor:
    seq_len, num_heads, head_dim = x.shape
    out = torch.empty_like(x, device="cpu")
    x_cpu = x.cpu()

    for token_idx in range(seq_len):
        for head_idx in range(num_heads):
            for pair_idx in range(head_dim // 2):
                even_col = 2 * pair_idx
                odd_col = even_col + 1
                x0 = float(x_cpu[token_idx, head_idx, even_col])
                x1 = float(x_cpu[token_idx, head_idx, odd_col])
                exponent = (2.0 * pair_idx) / head_dim
                theta = token_idx / (base ** exponent)
                cos_theta = math.cos(theta)
                sin_theta = math.sin(theta)
                out[token_idx, head_idx, even_col] = x0 * cos_theta - x1 * sin_theta
                out[token_idx, head_idx, odd_col] = x0 * sin_theta + x1 * cos_theta

    return out


def torch_rope_reference(x: torch.Tensor, base: float) -> torch.Tensor:
    seq_len, _, head_dim = x.shape
    pair_idx = torch.arange(head_dim // 2, device=x.device, dtype=torch.float32)
    exponent = (2.0 * pair_idx) / head_dim
    inv_freq = 1.0 / (base ** exponent)
    positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
    theta = positions[:, None] * inv_freq[None, :]
    cos_theta = torch.cos(theta)[:, None, :]
    sin_theta = torch.sin(theta)[:, None, :]

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    y_even = x_even * cos_theta - x_odd * sin_theta
    y_odd = x_even * sin_theta + x_odd * cos_theta

    out = torch.empty_like(x)
    out[..., 0::2] = y_even
    out[..., 1::2] = y_odd
    return out


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this Triton RoPE example.")

    seq_len = 128
    num_heads = 8
    head_dim = 64
    block_d = 64

    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head_dim.")
    if head_dim != block_d:
        raise ValueError("This teaching kernel expects head_dim == BLOCK_D.")

    device = torch.device("cuda")
    x = build_inputs(seq_len, num_heads, head_dim, device)
    y = torch.empty_like(x)

    total_rows = seq_len * num_heads
    rope_forward_kernel[(total_rows,)](
        x,
        y,
        seq_len,
        num_heads,
        head_dim,
        BASE,
        BLOCK_D=block_d,
    )

    ref_torch = torch_rope_reference(x, BASE)
    ref_cpu = cpu_rope_forward(x, BASE).to(device)

    max_abs_vs_torch = torch.max(torch.abs(y - ref_torch)).item()
    max_abs_vs_cpu = torch.max(torch.abs(y - ref_cpu)).item()

    ok = max_abs_vs_torch < 2e-5 and max_abs_vs_cpu < 2e-5
    if ok:
        sample = y[3, 1]
        print(
            f"triton_rope passed. seq_len={seq_len}, num_heads={num_heads}, "
            f"head_dim={head_dim}, max_abs_vs_torch={max_abs_vs_torch:.8e}, "
            f"max_abs_vs_cpu={max_abs_vs_cpu:.8e}"
        )
        print(
            f"sample output: y[0]={y.flatten()[0].item():.8f}, "
            f"y[1]={y.flatten()[1].item():.8f}, "
            f"y[sample_even]={sample[0].item():.8f}, "
            f"y[sample_odd]={sample[1].item():.8f}"
        )
    else:
        print(
            f"triton_rope failed. max_abs_vs_torch={max_abs_vs_torch:.8e}, "
            f"max_abs_vs_cpu={max_abs_vs_cpu:.8e}"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
