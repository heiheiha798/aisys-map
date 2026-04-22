"""
Layer Normalization
====================
在这个 tutorial 里，你会实现一个高性能的 layer normalization kernel，
它会比 PyTorch 实现更快。

你会顺带学到：

* 如何在 Triton 里实现 layer normalization 的 forward kernel。

* 如何在 feature 维度上做 reduction，并完成 fused normalization + affine transform。

"""

# %%
# Motivations 动机
# -----------
#
# *LayerNorm* 最早在 [BA2016]_ 中提出，
# 常用于提升 sequential models（例如 Transformers）或小 batch 神经网络的表现。
# 它接收一个向量 :math:`x` 作为输入，并输出同 shape 的向量 :math:`y`。
# 归一化过程会先减去均值，再除以标准差。
# 归一化之后，还会接一个带可学习参数的线性变换，参数是权重 :math:`w` 和偏置 :math:`b`。
# forward pass 可以写成：
#
# .. math::
#    y = \frac{ x - \text{E}[x] }{ \sqrt{\text{Var}(x) + \epsilon} } * w + b
#
# 其中 :math:`\epsilon` 是一个加到分母上的小常数，用于保证 numerical stability。
# 先看 forward pass 的实现。

import torch

import triton
import triton.language as tl

try:
    # 这里指的是 https://github.com/NVIDIA/apex，
    # 不是 PyPI 上的那个 apex，所以不应该加入 setup.py 的 extras_require。
    import apex
    HAS_APEX = True
except ModuleNotFoundError:
    HAS_APEX = False

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # 把 program id 映射到它要处理的 X / Y 行。
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # 计算均值。
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # 计算方差。
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # 写回 mean / rstd。
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # 做 normalization，并应用线性变换。
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # 写回输出。
        tl.store(Y + cols, y, mask=mask)


def layer_norm(x, normalized_shape, weight, bias, eps):
    # 分配输出。
    y = torch.empty_like(x)
    # 把输入 reshape 成 2D tensor。
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
    # 每个 feature 小于 64KB 时，直接走 fused kernel。
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # 为 num_warps 选择一个简单 heuristic。
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    # 启动 kernel。
    _layer_norm_fwd_fused[(M, )](  #
        x_arg, y, weight, bias, mean, rstd,  #
        x_arg.stride(0), N, eps,  #
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
    return y


# %%
# Benchmark 性能测试
# ---------
#
# 现在只关注 inference / forward 路径，把这个 kernel 的前向性能和 PyTorch 做对比。
# 这里主要关注每个 feature 不超过 64KB 的输入。


def test_layer_norm(M, N, dtype, eps=1e-5, device=DEVICE):
    # 构造输入数据。
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device)
    bias = torch.rand(w_shape, dtype=dtype, device=device)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    # forward pass。
    y_tri = layer_norm(x, w_shape, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
    # 比较结果。
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['triton', 'torch'] + (['apex'] if HAS_APEX else []),
        line_names=['Triton', 'Torch'] + (['Apex'] if HAS_APEX else []),
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='GB/s',
        plot_name='layer-norm-forward',
        args={'M': 4096, 'dtype': torch.float16},
    ))
def bench_layer_norm(M, N, dtype, provider, eps=1e-5, device=DEVICE):
    # 构造输入数据。
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device)
    bias = torch.rand(w_shape, dtype=dtype, device=device)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():

        if provider == "triton":
            return layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

        if provider == "torch":
            return torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

        if provider == "apex":
            apex_layer_norm = (apex.normalization.FusedLayerNorm(w_shape).to(x.device).to(x.dtype))
            return apex_layer_norm(x)  # noqa: F811, E704

    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


test_layer_norm(1151, 8192, torch.float16)
bench_layer_norm.run(save_path='.', print_data=True)

# %%
# References 参考文献
# ----------
#
# .. [BA2016] Jimmy Lei Ba and Jamie Ryan Kiros and Geoffrey E. Hinton, "Layer Normalization", Arxiv 2016
