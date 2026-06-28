"""
Libdevice (`tl.extra.libdevice`) function
==============================
Triton 可以调用 external library 里的自定义函数。
这个例子里，我们会使用 `libdevice` library，对 tensor 应用 `asin`。

关于 libdevice 函数的完整语义，可以参考
`CUDA libdevice-users-guide <https://docs.nvidia.com/cuda/libdevice-users-guide/index.html>`_
以及
`HIP device-lib source code <https://github.com/ROCm/llvm-project/tree/amd-staging/amd/device-libs/ocml/src>`_。

在 `libdevice.py` 里，Triton 会把“计算逻辑相同但 data type 不同”的函数聚合到一起。
例如 `__nv_asin` 和 `__nv_asinf` 都是在计算反正弦主值，
区别只在于前者操作 `double`，后者操作 `float`。
Triton 会根据输入和输出类型，自动选择底层应该调用哪个 device function。
"""

# %%
# asin Kernel 计算内核
# ------------

import torch

import triton
import triton.language as tl
import inspect
import os
from triton.language.extra import libdevice

from pathlib import Path

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def asin_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x = libdevice.asin(x)
    tl.store(y_ptr + offsets, x, mask=mask)


# %%
# 使用默认 libdevice library path
# -----------------------------------------
# 可以直接使用 `triton/language/math.py` 中编码好的默认 libdevice path。

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
output_triton = torch.zeros(size, device=DEVICE)
output_torch = torch.asin(x)
assert x.is_cuda and output_triton.is_cuda
n_elements = output_torch.numel()
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
asin_kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=1024)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')


# %%
# 自定义 libdevice library path
# -------------------------------------
# 也可以通过 `extern_libs` 参数，把 libdevice 的实际路径显式传给 `asin` kernel。
def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


current_file = inspect.getfile(inspect.currentframe())
current_dir = Path(os.path.dirname(os.path.abspath(current_file)))

if is_cuda():
    libdir = current_dir.parent.parent / 'third_party/nvidia/backend/lib'
    extern_libs = {'libdevice': str(libdir / 'libdevice.10.bc')}
elif is_hip():
    libdir = current_dir.parent.parent / 'third_party/amd/backend/lib'
    extern_libs = {}
    libs = ["ocml", "ockl"]
    for lib in libs:
        extern_libs[lib] = str(libdir / f'{lib}.bc')
else:
    raise RuntimeError('unknown backend')

output_triton = torch.empty_like(x)
asin_kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=1024, extern_libs=extern_libs)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
