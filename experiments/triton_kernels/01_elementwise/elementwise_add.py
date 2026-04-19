import torch
import triton
import triton.language as tl


@triton.jit
def elementwise_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


def build_inputs(numel: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.arange(numel, device=device, dtype=torch.float32)
    x = 0.5 * torch.sin((idx + 1.0) * 0.013) + (((idx * 7.0) % 19.0) - 9.0) * 0.1
    y = 0.3 * torch.cos((idx + 5.0) * 0.017) - (((idx * 11.0) % 23.0) - 11.0) * 0.07
    return x.contiguous(), y.contiguous()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this Triton elementwise example.")

    device = torch.device("cuda")
    numel = 65536
    block_size = 256

    x, y = build_inputs(numel, device)
    out = torch.empty_like(x)

    grid = (triton.cdiv(numel, block_size),)
    elementwise_add_kernel[grid](
        x,
        y,
        out,
        numel,
        BLOCK_SIZE=block_size,
    )

    ref = x + y
    max_abs_diff = torch.max(torch.abs(out - ref)).item()
    if max_abs_diff >= 1e-6:
        idx = int(torch.argmax(torch.abs(out - ref)).item())
        raise AssertionError(
            f"elementwise_add failed. max_abs_diff={max_abs_diff}, "
            f"idx={idx}, got={out[idx].item()}, ref={ref[idx].item()}"
        )

    print(
        f"elementwise_add passed. numel={numel}, block_size={block_size}, "
        f"max_abs_diff={max_abs_diff}"
    )
    print(
        "sample output: "
        f"out[0]={out[0].item()}, out[1]={out[1].item()}, out[last]={out[-1].item()}"
    )


if __name__ == "__main__":
    main()
