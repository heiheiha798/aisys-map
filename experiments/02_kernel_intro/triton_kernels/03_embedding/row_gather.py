import os

import torch
import triton
import triton.language as tl


@triton.jit
def row_gather_kernel(table_ptr, ids_ptr, out_ptr, batch, dim, table_stride, out_stride, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    if row >= batch:
        return

    token_id = tl.load(ids_ptr + row)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < dim

    src_ptr = table_ptr + token_id * table_stride + offsets
    dst_ptr = out_ptr + row * out_stride + offsets

    values = tl.load(src_ptr, mask=mask, other=0.0)
    tl.store(dst_ptr, values, mask=mask)


def cpu_row_gather(table: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return table[ids]


def make_inputs(vocab: int, dim: int, batch: int, device: str):
    r = torch.arange(vocab, device=device, dtype=torch.float32).unsqueeze(1)
    c = torch.arange(dim, device=device, dtype=torch.float32).unsqueeze(0)
    x = torch.sin((r + 1.0) * 0.0013) + torch.cos((c + 5.0) * 0.017)
    y = (((r * 7.0 + c * 11.0) % 31.0) - 15.0) * 0.03
    table = 0.6 * x + y

    repeated_mode = os.getenv("GATHER_ID_MODE") == "repeated"
    idx = torch.arange(batch, device=device, dtype=torch.int32)
    if repeated_mode:
        ids = ((idx // 64) % 32).to(torch.int32)
    else:
        ids = ((idx * 37 + (idx // 7) * 17) % vocab).to(torch.int32)
    return table, ids, repeated_mode


def run() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this Triton example.")

    vocab = 8192
    dim = 256
    batch = 4096
    block_size = triton.next_power_of_2(dim)

    table, ids, repeated_mode = make_inputs(vocab, dim, batch, "cuda")
    out = torch.empty((batch, dim), device="cuda", dtype=table.dtype)

    grid = (batch,)
    row_gather_kernel[grid](
        table,
        ids,
        out,
        batch,
        dim,
        table.stride(0),
        out.stride(0),
        BLOCK_SIZE=block_size,
    )

    ref = cpu_row_gather(table, ids.to(torch.long))
    max_abs_diff = torch.max(torch.abs(out - ref)).item()
    ok = max_abs_diff < 1e-6

    if ok:
        mode = "repeated" if repeated_mode else "random"
        print(
            f"row_gather passed. vocab={vocab}, batch={batch}, dim={dim}, "
            f"block_size={block_size}, id_mode={mode}, max_abs_diff={max_abs_diff}"
        )
        print(
            f"sample output: out[0]={out.flatten()[0].item()}, "
            f"out[1]={out.flatten()[1].item()}, out[last]={out.flatten()[-1].item()}"
        )
    else:
        idx = torch.argmax(torch.abs(out - ref)).item()
        got = out.flatten()[idx].item()
        expected = ref.flatten()[idx].item()
        raise AssertionError(
            f"row_gather failed. max_abs_diff={max_abs_diff}, idx={idx}, got={got}, ref={expected}"
        )


if __name__ == "__main__":
    run()
