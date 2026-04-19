import math

import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    seq_len,
    head_dim,
    scale,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_row = tl.program_id(0)

    offs_d = tl.arange(0, BLOCK_D)
    offs_seq = tl.arange(0, BLOCK_SEQ)

    q_ptrs = q_ptr + query_row * head_dim + offs_d
    q_mask = offs_d < head_dim
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    scores = tl.zeros([BLOCK_SEQ], dtype=tl.float32)

    for key_row in range(0, BLOCK_SEQ):
        k_ptrs = k_ptr + key_row * head_dim + offs_d
        k = tl.load(k_ptrs, mask=q_mask, other=0.0)
        dot = tl.sum(q * k, axis=0)
        scores = tl.where(offs_seq == key_row, dot * scale, scores)

    row_max = tl.max(scores, axis=0)
    probs = tl.exp(scores - row_max)
    row_sum = tl.sum(probs, axis=0)
    probs = probs / row_sum

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for key_row in range(0, BLOCK_SEQ):
        v_ptrs = v_ptr + key_row * head_dim + offs_d
        v = tl.load(v_ptrs, mask=q_mask, other=0.0)
        weight = tl.sum(tl.where(offs_seq == key_row, probs, 0.0), axis=0)
        acc += weight * v

    out_ptrs = out_ptr + query_row * head_dim + offs_d
    tl.store(out_ptrs, acc, mask=q_mask)


def build_inputs(seq_len: int, head_dim: int, device: torch.device):
    q = torch.empty((seq_len, head_dim), device=device, dtype=torch.float32)
    k = torch.empty((seq_len, head_dim), device=device, dtype=torch.float32)
    v = torch.empty((seq_len, head_dim), device=device, dtype=torch.float32)

    for row in range(seq_len):
        for d in range(head_dim):
            q[row, d] = 0.07 * math.sin((row + 1) * (d + 1) * 0.13) + 0.03 * (
                (row + d) % 5 - 2
            )
            k[row, d] = 0.05 * math.cos((row + 3) * (d + 1) * 0.11) + 0.04 * (
                (row * 3 + d) % 7 - 3
            )
            v[row, d] = 0.06 * math.sin((row + 5) * (d + 2) * 0.09) + 0.02 * (
                (row * 5 + d * 2) % 9 - 4
            )
    return q, k, v


def cpu_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    seq_len, head_dim = q.shape
    out = torch.empty_like(q, device="cpu")
    scale = 1.0 / math.sqrt(head_dim)

    q_cpu = q.cpu()
    k_cpu = k.cpu()
    v_cpu = v.cpu()

    for query_row in range(seq_len):
        scores = []
        for key_row in range(seq_len):
            dot = 0.0
            for d in range(head_dim):
                dot += float(q_cpu[query_row, d]) * float(k_cpu[key_row, d])
            scores.append(dot * scale)

        row_max = max(scores)
        probs = [math.exp(score - row_max) for score in scores]
        row_sum = sum(probs)

        for d in range(head_dim):
            acc = 0.0
            for key_row in range(seq_len):
                acc += probs[key_row] / row_sum * float(v_cpu[key_row, d])
            out[query_row, d] = acc

    return out


def torch_attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[1])
    scores = torch.matmul(q, k.transpose(0, 1)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def main():
    if not torch.cuda.is_available():
      raise RuntimeError("CUDA is required to run this Triton attention example.")

    seq_len = 64
    head_dim = 32
    block_seq = 64
    block_d = 32

    if seq_len != block_seq:
        raise ValueError("This teaching kernel expects seq_len == BLOCK_SEQ.")
    if head_dim != block_d:
        raise ValueError("This teaching kernel expects head_dim == BLOCK_D.")

    device = torch.device("cuda")
    q, k, v = build_inputs(seq_len, head_dim, device)
    out = torch.empty_like(q)

    scale = 1.0 / math.sqrt(head_dim)
    attention_kernel[(seq_len,)](
        q,
        k,
        v,
        out,
        seq_len,
        head_dim,
        scale,
        BLOCK_SEQ=block_seq,
        BLOCK_D=block_d,
    )

    ref_torch = torch_attention_reference(q, k, v)
    ref_cpu = cpu_attention(q, k, v).to(device)

    max_abs_vs_torch = torch.max(torch.abs(out - ref_torch)).item()
    max_abs_vs_cpu = torch.max(torch.abs(out - ref_cpu)).item()

    ok = max_abs_vs_torch < 1e-4 and max_abs_vs_cpu < 1e-4
    if ok:
        print(
            f"triton_attention passed. seq_len={seq_len}, head_dim={head_dim}, "
            f"max_abs_vs_torch={max_abs_vs_torch:.8e}, max_abs_vs_cpu={max_abs_vs_cpu:.8e}"
        )
        print(
            f"sample output: out[0]={out.flatten()[0].item():.8f}, "
            f"out[1]={out.flatten()[1].item():.8f}, "
            f"out[last]={out.flatten()[-1].item():.8f}"
        )
    else:
        print(
            f"triton_attention failed. max_abs_vs_torch={max_abs_vs_torch:.8e}, "
            f"max_abs_vs_cpu={max_abs_vs_cpu:.8e}"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
