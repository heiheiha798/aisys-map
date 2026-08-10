# AISys Map

一张围绕 inference systems 的学习地图，配套最小实验和开源 case study。

这个仓库不追求覆盖完整 AI systems 版图，而是把当前学习材料收束到一条可执行主线：先建立模型和 kernel 的基础直觉，再进入真实 inference runtime、scheduler、KV cache、量化、并行和 decode path 优化。

## 入口

| 文档 | 用途 |
|---|---|
| [roadmap.md](roadmap.md) | 当前学习顺序、范围边界和完成标准 |
| [experiments.md](experiments.md) | 5 组实验和 case study 的索引 |
| [notes/README.md](notes/README.md) | 长期复用的基础概念笔记索引 |

## 当前边界

主线关注：

- decoder-only 模型数据流与 attention 变体
- CUDA / Triton 教学 kernel、GPU 存储层级、Tensor Core / WMMA
- kernel profiling、GEMM 优化和 kernel DSL case study
- prefill / decode、continuous batching、KV cache、CUDA Graph decode
- weight-only quantization、推理侧 `TP / EP`
- nano-vLLM、nanoPD、flash-deepseek-v2-lite 这类完整工程和真实 decode path case study

暂不展开：

- 训练系统全景和训练并行细节
- 集群调度、HTTP service、可靠性工程
- 通用分布式通信专题
- speculative decoding、MoE 系统、edge inference 的完整展开
- 权重格式和部署产物的专门路线

## 目录结构

| 路径 | 内容 |
|---|---|
| [notes](notes) | Attention 学习主线、GPU 组织、CUDA 编程对象、Tensor Core / WMMA、kernel 分类等概念笔记 |
| [01_model_basics](01_model_basics) | 模型入门：最朴素的数据流和 attention 变体 |
| [02_kernel_intro](02_kernel_intro) | kernel 入门：CUDA / Triton 教学版 kernel |
| [03_kernel_advanced](03_kernel_advanced) | kernel 深入：DSL puzzles 与 GEMM 优化 submodule |
| [04_inference_system](04_inference_system) | inference 系统机制：真实推理路径、调度、量化、并行 |
| [05_case_studies](05_case_studies) | 实战演练：完整开源 engine、PD 分离、真实 decode path 优化 |
| [roadmap.md](roadmap.md) | 当前学习主线 |
| [experiments.md](experiments.md) | 5 组实验和 case study 索引 |
| [requirements.txt](requirements.txt) | 实验脚本共用 Python 依赖 |

## 5 组实验

| 分组 | 目录 | 用途 |
|---|---|---|
| 模型入门 | [01_model_basics](01_model_basics) | `vanilla_transformer`、`attention_variants`、`attention_patterns` |
| Kernel 入门 | [02_kernel_intro](02_kernel_intro) | `cuda_kernels`、`triton_kernels`、`triton-tutorials` |
| Kernel 深入 | [03_kernel_advanced](03_kernel_advanced) | `tilelang-puzzles`*、`KDT-DSL`*、`SGEMM_CUDA`* |
| Inference 机制 | [04_inference_system](04_inference_system) | `hf_inference`、`continuous_batching`、`quantization`、`parallel` |
| 实战演练 | [05_case_studies](05_case_studies) | `nano-vllm`*、`nanoPD`*、`flash-deepseek-v2-lite`* |

`*` 为 git submodule。初始化全部 submodule：

```bash
git submodule update --init --recursive
```

只拉某一个 case study：

```bash
git submodule update --init 05_case_studies/flash-deepseek-v2-lite
git submodule update --init 03_kernel_advanced/SGEMM_CUDA
```

## 使用方式

建议按 [roadmap.md](roadmap.md) 的顺序推进：

1. 先读 [notes/README.md](notes/README.md) 里的基础概念笔记。
2. 再按 `01` 到 `05` 跑实验和读 case study。
3. 每个实验先回答：它依赖哪篇笔记、属于哪组实验、主要瓶颈是 compute、memory、sync、runtime、scheduler 还是 state。

具体每组实验的材料说明放在 [experiments.md](experiments.md)，README 只作为入口。
