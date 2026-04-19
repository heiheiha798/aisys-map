import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, k, BLOCK_K):
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k_start + offs_k)[None, :] * stride_ak
        b_ptrs = b_ptr + (k_start + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a_mask = (offs_m[:, None] < m) & ((k_start + offs_k)[None, :] < k)
        b_mask = ((k_start + offs_k)[:, None] < k) & (offs_n[None, :] < n)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)
        acc = tl.dot(a, b, acc=acc, input_precision="ieee", out_dtype=tl.float32)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    tl.store(c_ptrs, acc, mask=c_mask)


def build_inputs(m: int, n: int, k: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    row_a = torch.arange(m, device=device, dtype=torch.float32).unsqueeze(1)
    col_a = torch.arange(k, device=device, dtype=torch.float32).unsqueeze(0)
    row_b = torch.arange(k, device=device, dtype=torch.float32).unsqueeze(1)
    col_b = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(0)

    a = 0.5 * torch.sin((row_a + 1.0) * 0.021) + 0.3 * torch.cos((col_a + 3.0) * 0.017)
    a = a + (((row_a * 5.0 + col_a * 7.0) % 29.0) - 14.0) * 0.02

    b = 0.4 * torch.cos((row_b + 2.0) * 0.019) - 0.6 * torch.sin((col_b + 5.0) * 0.013)
    b = b + (((row_b * 11.0 + col_b * 3.0) % 31.0) - 15.0) * 0.015

    return a.contiguous(), b.contiguous()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this Triton GEMM example.")

    device = torch.device("cuda")
    m = 192
    n = 160
    k = 224
    block_m = 64
    block_n = 64
    block_k = 32

    a, b = build_inputs(m, n, k, device)
    c = torch.empty((m, n), device=device, dtype=torch.float32)

    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    matmul_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )

    ref = torch.matmul(a, b)
    max_abs_diff = torch.max(torch.abs(c - ref)).item()
    if max_abs_diff >= 2e-4:
        flat_idx = int(torch.argmax(torch.abs(c - ref)).item())
        row = flat_idx // n
        col = flat_idx % n
        raise AssertionError(
            f"triton_gemm failed. max_abs_diff={max_abs_diff}, "
            f"row={row}, col={col}, got={c[row, col].item()}, ref={ref[row, col].item()}"
        )

    print(
        f"triton_gemm passed. m={m}, n={n}, k={k}, "
        f"block_m={block_m}, block_n={block_n}, block_k={block_k}, "
        f"max_abs_diff={max_abs_diff}"
    )
    print(
        "sample output: "
        f"c[0,0]={c[0, 0].item()}, c[0,1]={c[0, 1].item()}, c[last]={c[-1, -1].item()}"
    )


if __name__ == "__main__":
    main()
