import torch
import triton
import triton.language as tl


def cpu_row_softmax(x: torch.Tensor) -> torch.Tensor:
    row_max = x.max(dim=1, keepdim=True).values
    exp_x = torch.exp(x - row_max)
    return exp_x / exp_x.sum(dim=1, keepdim=True)


@triton.jit
def row_softmax_online_kernel(
    x_ptr,
    y_ptr,
    rows,
    cols,
    stride_x_row,
    stride_y_row,
    BLOCK_SIZE: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= rows:
        return

    x_row_ptr = x_ptr + row * stride_x_row
    y_row_ptr = y_ptr + row * stride_y_row

    running_max = -float("inf")
    running_sum = 0.0

    for tile_idx in range(NUM_TILES):
        start = tile_idx * BLOCK_SIZE
        cols_offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = cols_offsets < cols
        x = tl.load(x_row_ptr + cols_offsets, mask=mask, other=-float("inf"))

        tile_max = tl.max(x, axis=0)
        tile_sum = tl.sum(tl.exp(x - tile_max), axis=0)

        new_max = tl.maximum(running_max, tile_max)
        running_sum = (
            running_sum * tl.exp(running_max - new_max)
            + tile_sum * tl.exp(tile_max - new_max)
        )
        running_max = new_max

    for tile_idx in range(NUM_TILES):
        start = tile_idx * BLOCK_SIZE
        cols_offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = cols_offsets < cols
        x = tl.load(x_row_ptr + cols_offsets, mask=mask, other=-float("inf"))
        y = tl.exp(x - running_max) / running_sum
        tl.store(y_row_ptr + cols_offsets, y, mask=mask)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available; Triton online softmax needs an NVIDIA GPU."
        )

    rows = 4096
    cols = 256
    device = "cuda"
    dtype = torch.float32
    block_size = 128
    num_tiles = triton.cdiv(cols, block_size)

    x = torch.empty((rows, cols), device=device, dtype=dtype)
    for r in range(rows):
        for c in range(cols):
            x[r, c] = float((r + c) % 17 - 8)

    y = torch.empty_like(x)

    row_softmax_online_kernel[(rows,)](
        x,
        y,
        rows,
        cols,
        x.stride(0),
        y.stride(0),
        BLOCK_SIZE=block_size,
        NUM_TILES=num_tiles,
    )

    ref = cpu_row_softmax(x.cpu()).to(device)
    max_abs_diff = torch.max(torch.abs(y - ref)).item()

    if max_abs_diff >= 1e-4:
        flat_idx = torch.argmax(torch.abs(y - ref)).item()
        got = y.flatten()[flat_idx].item()
        expected = ref.flatten()[flat_idx].item()
        raise AssertionError(
            f"row_softmax_online failed. max_abs_diff={max_abs_diff}, index={flat_idx}, "
            f"got={got}, expected={expected}"
        )

    print(
        f"row_softmax_online passed. rows={rows}, cols={cols}, "
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
