"""
Low-Memory Dropout
==================

在这个 tutorial 里，你会实现一个 memory-efficient 的 dropout。
它的状态只需要一个 int32 seed，而不是像传统实现那样维护一个与输入同 shape 的 bit mask tensor。

你会顺带学到：

* 用 PyTorch 朴素实现 Dropout 的局限。

* Triton 里 parallel pseudo-random number generation 的基本写法。

"""

# %%
# Baseline 基线实现
# --------
#
# *dropout* 最早在 [SRIVASTAVA2014]_ 中提出，
# 主要用于 low-data regime 下提升深度网络表现，本质上是一种 regularization。
#
# 它接收一个 vector 作为输入，并输出同 shape 的 vector。
# 输出中的每个 scalar 都有 :math:`p` 的概率被置零，否则就直接拷贝输入值。
# 这样网络即使只看到 :math:`1 - p` 比例的输入，也必须学会稳定工作。
#
# 在 evaluation 时，我们希望使用完整网络，因此会设 :math:`p=0`。
# 但如果训练时直接丢值、不做缩放，输出 norm 会发生变化，
# 这通常不是好事，例如可能导致 output softmax temperature 被人为拉低。
# 所以常见做法是在保留元素上乘 :math:`\frac{1}{1 - p}`，
# 这样无论 dropout probability 多大，输出的尺度都更一致。
#
# 先看一个 baseline implementation。

import tabulate
import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _dropout(
    x_ptr,  # 指向输入的 pointer。
    x_keep_ptr,  # 指向 0/1 mask 的 pointer。
    output_ptr,  # 指向输出的 pointer。
    n_elements,  # `x` tensor 的元素总数。
    p,  # `x` 中每个元素被置零的概率。
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # 读取输入数据。
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    # 这一行就是前面讲的核心缩放逻辑。
    output = tl.where(x_keep, x / (1 - p), 0.0)
    # 把结果写回 output。
    tl.store(output_ptr + offsets, output, mask=mask)


def dropout(x, x_keep, p):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output


# 输入 tensor。
x = torch.randn(size=(10, ), device=DEVICE)
# Dropout mask。
p = 0.5
x_keep = (torch.rand(size=(10, ), device=DEVICE) > p).to(torch.int32)
#
output = dropout(x, x_keep=x_keep, p=p)
print(tabulate.tabulate([
    ["input"] + x.tolist(),
    ["keep mask"] + x_keep.tolist(),
    ["output"] + output.tolist(),
]))

# %%
# Seeded dropout 基于种子的实现
# --------------
#
# 上面的 dropout 实现虽然能用，但工程上会有些 awkward。
# 第一，它需要把 dropout mask 存下来，供 backpropagation 使用。
# 第二，在 recompute / checkpointing 场景里，dropout state 的管理会变得很麻烦，
# 例如 PyTorch 文档里关于 `preserve_rng_state` 的说明就很典型。
# 所以这里换一种实现方式：它
# (1) memory footprint 更小；
# (2) data movement 更少；
# (3) 更容易在多次 kernel 调用之间保持随机性状态的一致性。
#
# Triton 里的 pseudo-random number generation 用起来很直接。
# 这里会用 :code:`triton.language.rand`，它接受一个 seed 和一组 :code:`int32` offsets，
# 返回落在 [0, 1) 区间内、均匀分布的 :code:`float32` block。
# 如果你需要更复杂的方案，Triton 也提供了其他
# :ref:`random number generation strategies<Random Number Generation>`。
#
# .. note::
#    Triton 的 PRNG 实现基于 Philox algorithm，见 [SALMON2011]_。
#
# 下面把这些 pieces 组合起来。


@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # 计算当前 program instance 负责的 element offsets。
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 从 x 读取输入。
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # 基于 seed 生成随机数，并据此执行 dropout。
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # 写回输出。
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output


x = torch.randn(size=(10, ), device=DEVICE)
# 和 baseline 对比时可以注意：这里根本不会显式实例化 dropout mask tensor。
output = seeded_dropout(x, p=0.5, seed=123)
output2 = seeded_dropout(x, p=0.5, seed=123)
output3 = seeded_dropout(x, p=0.5, seed=512)

print(
    tabulate.tabulate([
        ["input"] + x.tolist(),
        ["output (seed = 123)"] + output.tolist(),
        ["output (seed = 123)"] + output2.tolist(),
        ["output (seed = 512)"] + output3.tolist(),
    ]))

# %%
# 这样就完成了：只要 seed 相同，这个 Triton kernel 就会生成相同的 dropout mask。
# 如果你想继续看 GPU programming 里 pseudorandomness 的更多用法，
# 可以继续读 `python/triton/language/random.py`。

# %%
# Exercises 练习
# ---------
#
# 1. 把 kernel 扩展到 matrix 输入，并改成每行一个 seed。
# 2. 增加对 striding 的支持。
# 3. （挑战）实现一个 sparse Johnson-Lindenstrauss transform kernel，
#    每次根据 seed 在线生成 projection matrix。

# %%
# References 参考文献
# ----------
#
# .. [SALMON2011] John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw, "Parallel Random Numbers: As Easy as 1, 2, 3", 2011
# .. [SRIVASTAVA2014] Nitish Srivastava and Geoffrey Hinton and Alex Krizhevsky and Ilya Sutskever and Ruslan Salakhutdinov, "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", JMLR 2014
