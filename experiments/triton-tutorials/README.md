# Triton Official Tutorials Mirror

这个目录只服务于：

- [experiments/triton-tutorials](/data/home/tianjianyang/code/aisys-map/experiments/triton-tutorials)

也就是：

- 放一份本地可读、可改、可对照的 Triton 官方 tutorial 镜像
- 作为 `aisys-map` 里的学习参考材料

不把这些文件当成“本 repo 自己从零写的教学 kernel”。

## 这些文件来自哪里

这个目录里的 `01~09` 文件，来源于 Triton 官方仓库：

- `/data/home/tianjianyang/code/triton/python/tutorials/`

拷贝时保留了原始文件名，方便和官方教程一一对应。

## 为什么要在这里再放一份

原因很简单：

- 希望在 `aisys-map` 里保留一份可直接阅读的官方参考实现
- 方便和本 repo 自己写的 `experiments/triton_kernels/` 教学版例子对照
- 方便做少量本地修改，而不影响“这是官方教程”的事实

所以这个目录更像：

- 官方教程镜像
- 本地阅读副本

而不是：

- 重新设计的一套 Triton 教材

## 当前包含哪些文件

- `01-vector-add.py`
- `02-fused-softmax.py`
- `03-matrix-multiplication.py`
- `04-low-memory-dropout.py`
- `05-layer-norm.py`
- `06-fused-attention.py`
- `07-extern-functions.py`
- `08-grouped-gemm.py`
- `09-persistent-matmul.py`

## 本地做过哪些修改

这些文件不是完全原封不动地复制过来，当前做过一些面向本地学习环境的修改。

主要包括：

- 把大量注释改成了中英混合版本，方便直接阅读
- `05-layer-norm.py` 删掉了 backward，只保留更适合当前学习目标的 forward / inference 主线
- `08-grouped-gemm.py` 删掉了 TMA 路径，并整理了文件结构和注释
- `09-persistent-matmul.py` 删掉了 Hopper / TMA / descriptor 路径，只保留 A6000 相关主线，并补了更多解释性注释

所以如果后面你要做“和官方源码逐行严格比对”，请优先回到原始目录：

- `/data/home/tianjianyang/code/triton/python/tutorials/`

## 怎么使用这个目录

更推荐把这里当成：

- 官方实现参考
- 学习时的本地注释版

而把下面这个目录当成：

- 自己写的教学版最小 kernel

对应目录：

- [triton_kernels](/data/home/tianjianyang/code/aisys-map/experiments/triton_kernels)

一句话说清楚：

这个目录是 **official Triton tutorials 的本地镜像，并带少量面向 A6000 / 中文学习路径的修改**。
