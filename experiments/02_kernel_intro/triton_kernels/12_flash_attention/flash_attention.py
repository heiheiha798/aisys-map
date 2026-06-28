import math

import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    seq_len,
    head_dim,
    stride_qm,
    stride_qd,
    stride_km,
    stride_kd,
    stride_vm,
    stride_vd,
    stride_om,
    stride_od,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    q_rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = tl.arange(0, BLOCK_D)

    q_ptrs = q_ptr + q_rows[:, None] * stride_qm + d_offsets[None, :] * stride_qd
    q = tl.load(q_ptrs)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):
        k_cols = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = k_ptr + k_cols[:, None] * stride_km + d_offsets[None, :] * stride_kd
        v_ptrs = v_ptr + k_cols[:, None] * stride_vm + d_offsets[None, :] * stride_vd
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        scores = tl.dot(q, tl.trans(k)) * scale
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)

        acc = acc * alpha[:, None] + tl.dot(p, v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_ij

    out = acc / l_i[:, None]
    out_ptrs = out_ptr + q_rows[:, None] * stride_om + d_offsets[None, :] * stride_od
    tl.store(out_ptrs, out)


def build_inputs(seq_len: int, head_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    cols = torch.arange(head_dim, device=device, dtype=torch.float32).unsqueeze(0)

    q = 0.7 * torch.sin((rows + 1.0) * (cols + 2.0) * 0.011)
    q = q + (((rows * 5.0 + cols * 3.0) % 19.0) - 9.0) * 0.03

    k = 0.5 * torch.cos((rows + 3.0) * (cols + 1.0) * 0.009)
    k = k + (((rows * 7.0 + cols * 11.0) % 23.0) - 11.0) * 0.025

    v = 0.6 * torch.sin((rows + 5.0) * (cols + 4.0) * 0.007)
    v = v - (((rows * 13.0 + cols * 2.0) % 29.0) - 14.0) * 0.02

    return q.contiguous(), k.contiguous(), v.contiguous()


def torch_attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[1])
    scores = torch.matmul(q, k.transpose(0, 1)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this Triton Flash Attention example.")

    device = torch.device("cuda")
    seq_len = 128
    head_dim = 64
    block_m = 32
    block_n = 32
    block_d = 64

    if seq_len % block_m != 0 or seq_len % block_n != 0:
        raise ValueError("This teaching kernel expects seq_len to be divisible by BLOCK_M and BLOCK_N.")
    if head_dim != block_d:
        raise ValueError("This teaching kernel expects head_dim == BLOCK_D.")

    q, k, v = build_inputs(seq_len, head_dim, device)
    out = torch.empty_like(q)
    scale = 1.0 / math.sqrt(head_dim)

    flash_attention_kernel[(seq_len // block_m,)](
        q,
        k,
        v,
        out,
        seq_len,
        head_dim,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        out.stride(0),
        out.stride(1),
        scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
    )

    ref = torch_attention_reference(q, k, v)
    max_abs_diff = torch.max(torch.abs(out - ref)).item()
    if max_abs_diff >= 3e-4:
        flat_idx = int(torch.argmax(torch.abs(out - ref)).item())
        row = flat_idx // head_dim
        col = flat_idx % head_dim
        raise AssertionError(
            f"flash_attention failed. max_abs_diff={max_abs_diff}, "
            f"row={row}, col={col}, got={out[row, col].item()}, ref={ref[row, col].item()}"
        )

    print(
        f"flash_attention passed. seq_len={seq_len}, head_dim={head_dim}, "
        f"block_m={block_m}, block_n={block_n}, block_d={block_d}, "
        f"max_abs_diff={max_abs_diff}"
    )
    print(
        "sample output: "
        f"out[0,0]={out[0, 0].item()}, out[0,1]={out[0, 1].item()}, out[last]={out[-1, -1].item()}"
    )


if __name__ == "__main__":
    main()
