import torch
import triton
import triton.language as tl


def cpu_row_softmax(x: torch.Tensor) -> torch.Tensor:
    row_max = x.max(dim=1, keepdim=True).values
    exp_x = torch.exp(x - row_max)
    return exp_x / exp_x.sum(dim=1, keepdim=True)


@triton.jit
def row_softmax_kernel(
    x_ptr,
    y_ptr,
    rows,
    cols,
    stride_x_row,
    stride_y_row,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= rows:
        return

    cols_offsets = tl.arange(0, BLOCK_SIZE)
    mask = cols_offsets < cols

    x_row_ptr = x_ptr + row * stride_x_row
    y_row_ptr = y_ptr + row * stride_y_row

    x = tl.load(x_row_ptr + cols_offsets, mask=mask, other=-float("inf"))
    row_max = tl.max(x, axis=0)
    exp_x = tl.exp(x - row_max)
    row_sum = tl.sum(exp_x, axis=0)
    y = exp_x / row_sum
    tl.store(y_row_ptr + cols_offsets, y, mask=mask)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; Triton softmax needs an NVIDIA GPU.")

    rows = 128
    cols = 256
    device = "cuda"
    dtype = torch.float32
    block_size = triton.next_power_of_2(cols)

    x = torch.empty((rows, cols), device=device, dtype=dtype)
    for r in range(rows):
        for c in range(cols):
            x[r, c] = float((r + c) % 17 - 8)

    y = torch.empty_like(x)

    row_softmax_kernel[(rows,)](
        x,
        y,
        rows,
        cols,
        x.stride(0),
        y.stride(0),
        BLOCK_SIZE=block_size,
    )

    ref = cpu_row_softmax(x.cpu()).to(device)
    max_abs_diff = torch.max(torch.abs(y - ref)).item()

    if max_abs_diff >= 1e-4:
        flat_idx = torch.argmax(torch.abs(y - ref)).item()
        got = y.flatten()[flat_idx].item()
        expected = ref.flatten()[flat_idx].item()
        raise AssertionError(
            f"row_softmax failed. max_abs_diff={max_abs_diff}, index={flat_idx}, "
            f"got={got}, expected={expected}"
        )

    print(
        f"row_softmax passed. rows={rows}, cols={cols}, "
        f"BLOCK_SIZE={block_size}, max_abs_diff={max_abs_diff}"
    )
    print(
        "sample output: "
        f"y[0]={y.flatten()[0].item()}, "
        f"y[1]={y.flatten()[1].item()}, "
        f"y[255]={y.flatten()[255].item()}"
    )


if __name__ == "__main__":
    main()
