"""
Fused Softmax
=============

在这个 tutorial 里，你会实现一个 fused softmax kernel。
对于行能放进 GPU SRAM 的那类矩阵，它会明显快于 PyTorch 的 native op。

你会顺带学到：

* 对 bandwidth-bound operations 来说，kernel fusion 为什么重要。

* Triton 里的 reduction operators 应该怎么用。

"""

# %%
# Motivations 动机
# -----------
#
# 自己写 elementwise addition 这种 GPU kernel 很适合入门，但在真实场景里帮助有限。
# 所以这里换成一个更有代表性的例子：带 numerical stabilization 的 softmax。

import torch

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                   'gfx90a', 'gfx908')


def naive_softmax(x):
    """用原生 PyTorch 计算 X 的 row-wise softmax。

    这里先减去每一行的最大值来避免 overflow。
    因为 softmax 对这个 shift 不敏感，所以结果不变。
    """
    # 读 MN 个 elements；写 M 个 elements。
    x_max = x.max(dim=1)[0]
    # 读 MN + M 个 elements；写 MN 个 elements。
    z = x - x_max[:, None]
    # 读 MN 个 elements；写 MN 个 elements。
    numerator = torch.exp(z)
    # 读 MN 个 elements；写 M 个 elements。
    denominator = numerator.sum(dim=1)
    # 读 MN + M 个 elements；写 MN 个 elements。
    ret = numerator / denominator[:, None]
    # 总计：读取 5MN + 2M 个 elements；写回 3MN + 2M 个 elements。
    return ret


# %%
# 如果直接用 PyTorch 按朴素方式实现 :code:`y = naive_softmax(x)`，
# 对于 :math:`x \in R^{M \times N}`，需要从 DRAM 读取 :math:`5MN + 2M` 个 elements，
# 并写回 :math:`3MN + 2M` 个 elements。
# 这显然很浪费；更理想的做法是写一个 fused kernel，只读一次 X，
# 然后在 on-chip 完成所有计算。
# 这样理论上只需要读写 :math:`MN` 量级的数据，因此可以期待大约 4x 的理论加速
# （也就是 :math:`(8MN + 4M) / 2MN`）。
# `torch.jit.script` 也想自动做这类 kernel fusion，
# 但后面会看到，它离理想情况还有距离。

# %%
# Compute Kernel 计算内核
# --------------
#
# softmax kernel 的思路是：每个 program 按 program 数量做 stride，
# 处理若干行输入矩阵 X，完成 normalization 后再写回输出 Y。
#
# Triton 有个重要限制：每个 block 的元素数必须是 power-of-two。
# 所以如果想支持任意输入形状，就要在内部对每一行做 padding，
# 并用 mask 正确保护 memory operations。


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # 当前 program 起始处理的 row。
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # stride 表示 pointer 沿着行方向前进一行需要增加多少。
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # BLOCK_SIZE 取不小于 n_cols 的下一个 power-of-two，
        # 这样一整行就能装进一个 block。
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # 把一整行读进 SRAM；由于 BLOCK_SIZE 可能大于 n_cols，需要配合 mask。
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # 先减最大值，保证 numerical stability。
        row_minus_max = row - tl.max(row, axis=0)
        # Triton 的 exponentiation 很快，但它是近似实现，可以类比 CUDA 里的 `__expf`。
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # 把结果写回 DRAM。
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


# %%
# 下面封装一个 helper function，给任意输入 tensor 组装 kernel launch 和对应的 (meta-)arguments。

properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}


def softmax(x):
    n_rows, n_cols = x.shape

    # 每次 loop 的 block size 取为不小于 `x` 列数的最小 power-of-two。
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # 另一个常见技巧是增加 `num_warps`，让每一行分摊到更多 threads 上。
    # 下一个 tutorial 会展示如何更自然地 auto-tune 这个值，
    # 而不是手工拍 heuristic。
    num_warps = 8

    # software pipelining 的 stage 数。
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # 分配输出 tensor。
    y = torch.empty_like(x)

    # 先预编译 kernel，拿到 register usage，并估算 thread occupancy。
    kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages, num_warps=num_warps, grid=(1, ))
    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    if is_hip():
        # NUM_REGS 表示 regular purpose registers 的数量。
        # 在 CDNA 上，它可能只占全部 registers 的一半；但并不是所有架构都如此。
        # 参考 ISA SECTION 3.6.4（CDNA3）：
        # VGPRs 分成 regular VGPRs 和 accumulation VGPRs 两个池。
        # accumulation VGPRs 会被 matrix VALU instructions 使用，也能直接从 memory 加载。
        # 一个 wave 最多可以有 512 个 VGPRs，其中两类各最多 256 个；
        # 如果总数少于 512，则两类的分配比例可以灵活变化，并不要求严格对半。
        NUM_GPRS = NUM_REGS
        if is_cdna():
            NUM_GPRS = NUM_REGS * 2

        # MAX_NUM_THREADS 表示每个 multi-processor 上最多能常驻多少 threads。
        # 除以 WARP_SIZE 后，就能得到一个 CU 上最多能并行跑多少个 waves。
        MAX_NUM_THREADS = properties["max_threads_per_sm"]
        max_num_waves = MAX_NUM_THREADS // WARP_SIZE
        occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
    else:
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy

    num_programs = min(num_programs, n_rows)

    # 启动一组 persistent programs。
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)
    return y


# %%
# Unit Test 单元测试
# ---------

# %%
# 这里故意用一个行列数都不规则的矩阵来测，
# 这样可以验证前面 padding + mask 的处理确实正确。

torch.manual_seed(0)
x = torch.randn(1823, 781, device=DEVICE)
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)

# %%
# 结果和预期一致，两边输出相同。

# %%
# Benchmark 性能测试
# ---------
#
# 这里固定 4096 行，只把列数当作自变量做 benchmark。
# 然后把 Triton kernel 的性能和
# (1) :code:`torch.softmax`
# (2) 上面定义的 :code:`naive_softmax`
# 做对比。


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # 作为 plot x-axis 的 argument 名字。
        x_vals=[128 * i for i in range(2, 100)],  # `x_name` 的候选取值。
        line_arg='provider',  # 决定不同曲线的 argument。
        line_vals=['triton', 'torch', 'naive_softmax'],  # `line_arg` 的候选取值。
        line_names=["Triton", "Torch", "Naive Softmax"],  # 曲线显示名。
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # 曲线样式。
        ylabel="GB/s",  # y-axis 标签。
        plot_name="softmax-performance",  # plot 名字，也会作为保存文件名。
        args={'M': 4096},  # 不在 `x_names` / `y_name` 里的固定参数。
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    if provider == 'naive_softmax':
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


benchmark.run(show_plots=True, print_data=True)

# %%
# 从上面的 plot 可以看出：
#  - Triton 比 Torch JIT 快约 4x，这说明这里 Torch JIT 基本没有做有效 fusion。
#  - Triton 也明显快于 :code:`torch.softmax`，而且代码通常更容易阅读、理解和维护。
#    当然，PyTorch 的 `softmax` 更通用，它可以处理任意形状的 tensor。
