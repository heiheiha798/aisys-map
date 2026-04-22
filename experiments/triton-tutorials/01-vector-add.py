"""
Vector Addition
===============

在这个 tutorial 里，你会用 Triton 写一个简单的 vector addition kernel。

你会顺带熟悉：

* Triton 的基础 programming model。

* 用来定义 Triton kernels 的 `triton.jit` decorator。

* 如何把 custom op 和 native reference implementation 做 validation 与 benchmark。

"""

# %%
# Compute Kernel 计算部分
# --------------

import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # 指向 second input vector 的 pointer。
               output_ptr,  # 指向 output vector 的 pointer。
               n_elements,  # vector 的元素总数。
               BLOCK_SIZE: tl.constexpr,  # 每个 program 处理多少个 elements。
               # NOTE: 这里用 `constexpr`，这样它才能作为 shape value 参与编译期推导。
               ):
    # 会有多个 programs 并行处理不同数据块；这里先拿到当前是第几个 program。
    pid = tl.program_id(axis=0)  # 我们使用 1D launch grid，所以 axis 是 0。
    # 当前 program 负责从 `block_start` 开始的一段连续元素。
    # 例如 vector 长度是 256、block_size 是 64 时，4 个 programs 会分别处理
    # [0:64, 64:128, 128:192, 192:256]。
    # 注意这里的 offsets 本质上是一组 element pointers。
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 用 mask 保护 memory operations，避免越界访问。
    mask = offsets < n_elements
    # 从 DRAM 读取 x 和 y；如果输入长度不是 block size 的整数倍，超出的 lanes 会被 mask 掉。
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # 把 x + y 写回 DRAM。
    tl.store(output_ptr + offsets, output, mask=mask)


# %%
# 下面再写一个 helper function，用来
# (1) 分配输出 `z` tensor；
# (2) 用合适的 grid / block size 启动上面的 kernel。


def add(x: torch.Tensor, y: torch.Tensor):
    # 先预分配 output tensor。
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    # SPMD launch grid 表示并行启动多少个 kernel instances。
    # 它和 CUDA launch grid 很像，既可以直接给 Tuple[int]，
    # 也可以给一个 Callable(meta-parameters) -> Tuple[int]。
    # 这里我们使用 1D grid，大小就是 block 的数量。
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - 每个 `torch.tensor` 都会隐式转换成指向首元素的 pointer。
    #  - `triton.jit` 函数可以通过 `fn[grid]` 这种形式变成 callable GPU kernel。
    #  - meta-parameters 记得用 keyword arguments 传入。
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # 这里返回的是 output handle；由于还没调用 `torch.cuda.synchronize()`，
    # 所以 kernel 此时仍然可能在异步执行。
    return output


# %%
# 现在可以直接调用上面的函数，计算两个 `torch.tensor` 的 element-wise sum，
# 并顺手检查结果是否正确。

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')

# %%
# 结果看起来没问题。

# %%
# Benchmark 性能测试
# ---------
#
# 接下来对不同 size 的 vector 做 benchmark，看看这个 custom op 相比 PyTorch 表现如何。
# Triton 自带了一套 utilities，可以比较简洁地画出不同 problem size 下的性能曲线。


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # 作为 plot x-axis 的 argument 名字。
        x_vals=[2**i for i in range(12, 28, 1)],  # `x_name` 的候选取值。
        x_log=True,  # x-axis 使用 log scale。
        line_arg='provider',  # 哪个 argument 决定 plot 中的不同曲线。
        line_vals=['triton', 'torch'],  # `line_arg` 的候选取值。
        line_names=['Triton', 'Torch'],  # 每条曲线显示出来的 label。
        styles=[('blue', '-'), ('green', '-')],  # 曲线样式。
        ylabel='GB/s',  # y-axis 标签。
        plot_name='vector-add-performance',  # plot 名字，同时也会作为保存文件名。
        args={},  # 不在 `x_names` 和 `y_name` 中的其他固定参数。
    ))
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# %%
# 现在可以直接运行上面这个带 decorator 的函数。
# `print_data=True` 会打印性能数字，`show_plots=True` 会显示图，
# `save_path='/path/to/results/'` 会把图和原始 CSV 数据一起保存到磁盘。
benchmark.run(print_data=True, show_plots=True)
