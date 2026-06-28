"""
Persistent Matmul
=====================
这个脚本演示如何用 Triton 实现 persistent matrix multiplication kernels。
这里保留最核心的两条路径：

* naive tiled matmul
* persistent matmul

脚本会 benchmark Triton、torch 和 cuBLAS 的实现，
并用 proton profiler 做分析。

.. code-block:: bash

    python 09-persistent-matmul.py --K_range 128 1024 --K_step 128
"""

import argparse
from contextlib import contextmanager

import torch
import triton
import triton.language as tl
import triton.profiler as proton


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


if is_cuda():
    from triton._C.libtriton import nvidia

    device_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    device_blas = nvidia.cublas.CublasLt(device_workspace)
elif is_hip():
    from triton._C.libtriton import amd

    device_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    device_blas = amd.hipblas.HipblasLt(device_workspace)
else:
    device_blas = None


def device_blas_name():
    return "cuBLAS" if is_cuda() else "hipBLAS"


# 给 proton profiler 生成更容易读的名字和统计信息。
# 这里会把 M/N/K 以及估算的 bytes / FLOPs 挂到 kernel 记录上。
def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    bytes_per_elem = args["c_ptr"].element_size()
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


def matmul_get_configs():
    return [
        triton.Config(
            {"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": 8},
            num_stages=s,
            num_warps=w,
        )
        for BM in [128]
        for BN in [128, 256]
        for BK in [64, 128]
        for s in [2, 3, 4]
        for w in [4, 8]
    ]


# -----------------------------------------------------------------------------
# Naive Matmul
# -----------------------------------------------------------------------------


@triton.autotune(
    configs=matmul_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # `pid` 是当前 program 的线性编号。
    # 在这个 naive 版本里，一个 program 只负责一个 output tile。
    pid = tl.program_id(axis=0)

    # 整个输出矩阵在 M / N 方向上分别会被切成多少个 tiles。
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # 这里沿用 03 matmul 里的 grouped ordering。
    # 它的作用是调整 tile 遍历顺序，改善 cache / L2 locality，
    # 不是改变数学计算本身。
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 当前 program 负责的 output tile 左上角坐标。
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    # 生成这个 tile 覆盖到的全局行 / 列索引。
    # 如果 tile 在边界处超出矩阵尺寸，先用 0 占位，后续通过 mask 避免非法访存。
    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)

    # 这是给编译器的 hint，帮助它生成更好的连续访存代码。
    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

    # 当前 K 子块内部的偏移。
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # 构造 A / B 子块的指针。
    # 可以直接理解成：
    # - A tile shape = [BLOCK_M, BLOCK_K]
    # - B tile shape = [BLOCK_K, BLOCK_N]
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 用 fp32 accumulator 做累加，避免精度太差。
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 沿着 K 维逐块推进。
    # 每一轮加载一块 A、一块 B，做一次 dot，并累加到 accumulator。
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)

        # 指针前进到 K 维的下一块。
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 累加结束后，把结果转成输出 dtype，并写回输出矩阵对应位置。
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # naive 版本的 grid 很直接：
    # 输出总共多少个 tiles，就开多少个 programs。
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


# -----------------------------------------------------------------------------
# Persistent Matmul
# -----------------------------------------------------------------------------


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    # 把线性的 tile_id 还原成二维 tile 坐标 (pid_m, pid_n)。
    # 这里仍然使用 grouped ordering，所以映射逻辑和 naive kernel 一致。
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.autotune(
    configs=matmul_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    # persistent kernel 的关键区别：
    # 一个 program 不再“只做一个 tile 就退出”，
    # 而是作为一个常驻 worker，连续处理多个 tiles。
    #
    # `start_pid` 可以理解成当前 worker 的编号。
    start_pid = tl.program_id(axis=0)

    # 整个输出矩阵一共有多少个 tiles。
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    # `tile_id_c` 是写回阶段维护的一份 tile counter。
    # 它和计算阶段的 tile_id 对应同一批 tiles，
    # 只是拆开写，让 prologue / epilogue 的依赖关系更简单。
    tile_id_c = start_pid - NUM_SMS
    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # 这就是 persistent 的核心：
    # 当前 worker 从 start_pid 开始，每次跨 NUM_SMS 继续拿下一个 tile。
    #
    # 例如有 84 个 SM：
    # - worker 0 做 tile 0, 84, 168, ...
    # - worker 1 做 tile 1, 85, 169, ...
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        # 先把线性的 tile_id 映射回二维 tile 坐标。
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N

        # 下面这部分和 naive kernel 基本一致：
        # 生成当前 tile 对应的行 / 列索引，并处理边界。
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # 沿着 K 维逐块加载并累加。
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)

        # 更新写回阶段使用的 tile 计数器，得到当前 tile 的输出坐标。
        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        c = accumulator.to(tl.float16)
        tl.store(c_ptrs, c, mask=c_mask)


def matmul_persistent(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"

    # persistent 版本通常只开接近 SM 数量的 programs。
    # 因为每个 program 不只是算一个 tile，而是会持续处理多个 tiles。
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # 如果总 tile 数比 SM 数还少，那就没必要开超过 tile 数的 programs。
    grid = lambda META: (min(num_sms, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])),)
    matmul_kernel_persistent[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        NUM_SMS=num_sms,
    )
    return c


# -----------------------------------------------------------------------------
# Reference / Profiling Helpers
# -----------------------------------------------------------------------------


def device_blas_matmul(a, b):
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    M, K = a.shape
    N, K = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    bytes_per_elem = a.element_size()
    flops_str = f"flops{bytes_per_elem * 8}"
    blas_name = device_blas_name()
    with proton.scope(
        f"{blas_name} [M={M}, N={N}, K={K}]",
        {"bytes": bytes_per_elem * (M * K + N * K + M * N), flops_str: 2.0 * M * N * K},
    ):
        device_blas.matmul(a, b, c)
    return c


# 这里传入的 `b` 是转置后的 [N, K] 视图，
# 所以 torch 侧用 `b.T` 恢复成常规 matmul 需要的 [K, N]。
def torch_matmul(a, b):
    M, K = a.shape
    N, K = b.shape
    bytes_per_elem = a.element_size()
    flops_str = f"flops{bytes_per_elem * 8}"
    with proton.scope(
        f"torch [M={M}, N={N}, K={K}]",
        {"bytes": bytes_per_elem * (M * K + N * K + M * N), flops_str: 2.0 * M * N * K},
    ):
        c = torch.matmul(a, b.T)
    return c


@contextmanager
def proton_context():
    proton.activate(0)
    try:
        yield
    finally:
        proton.deactivate(0)


def bench_fn(label, reps, warmup_reps, fn, *args):
    print(f"Benchmarking {label}: ...", end="")
    for _ in range(warmup_reps):
        fn(*args)
    with proton_context():
        for _ in range(reps):
            fn(*args)
    print(f"\rBenchmarking {label}: done")


def bench(K, reps=10000, warmup_reps=10000):
    M = 8192
    N = 8192
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    # 先转成 [N, K] contiguous，方便 torch / cuBLAS 这两条参考路径复用。
    # Triton 的两个 kernel 仍然接收常规的 [K, N]，所以调用时再传 `b.T`。
    b = b.T.contiguous()

    if device_blas is not None:
        bench_fn(device_blas_name(), reps, warmup_reps, device_blas_matmul, a, b)
    bench_fn("torch", reps, warmup_reps, torch_matmul, a, b)
    bench_fn("naive", reps, warmup_reps, matmul, a, b.T)
    bench_fn("persistent", reps, warmup_reps, matmul_persistent, a, b.T)


def run_test(expect, fn, a, b, label, enabled=True):
    print(f"  {label}: ...", end="")
    if enabled:
        actual = fn(a, b)
        passed = torch.allclose(expect, actual.to(expect.dtype), atol=1.0)
        icon = "✅" if passed else "❌"
    else:
        icon = "⭕"
    print(f"\r  {label}: {icon}  ")


def validate(M, N, K):
    print(f"{M=}, {N=}, {K=}, verification naive vs: ")
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    b = b.T.contiguous()

    # 统一用 naive Triton 结果作为参考值。
    naive_result = matmul(a, b.T).to(torch.float16)
    run_test(naive_result, torch_matmul, a, b, "Torch")
    run_test(naive_result, device_blas_matmul, a, b, device_blas_name(), enabled=device_blas is not None)
    run_test(naive_result, matmul_persistent, a, b.T, "Persistent")
    print()


def show_profile(profile_name):
    import triton.profiler.viewer as proton_viewer

    metric_names = ["tflop16/s", "time/ms"]
    file_name = f"{profile_name}.hatchet"
    tree, metrics = proton_viewer.parse(metric_names, file_name)
    proton_viewer.print_tree(tree, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-K", type=int, required=False, default=512)
    parser.add_argument("--K_range", type=int, nargs=2)
    parser.add_argument("--K_step", type=int, default=512)
    args = parser.parse_args()

    if args.K and args.K_range is None:
        args.K_range = [args.K, args.K]
        args.K_step = 1

    torch.manual_seed(0)

    validate(32, 32, 32)
    validate(8192, 8192, args.K_range[0])

    proton.start("matmul", hook="triton")
    proton.deactivate()
    for K in range(args.K_range[0], args.K_range[1] + 1, args.K_step):
        bench(K)
    proton.finalize()
    show_profile("matmul")
