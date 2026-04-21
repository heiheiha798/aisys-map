# llama.cpp 常用 API 整理

这份文档整理 `llama.cpp` 最常用的 API 面。

和 `vLLM` 不同，`llama.cpp` 的“常用 API”可以明显分成三块：

- CLI 二进制接口
- `llama-server` HTTP 接口
- `libllama` C API

如果你只是使用它，常接触前两块。
如果你想理解它怎么执行，最关键的是第三块。

## 1. 最常见的三层 API 面

### 1.1 CLI

最常见二进制：

- `llama-cli`
- `llama-server`

其中：

- `llama-cli` 适合本地直接生成
- `llama-server` 适合 HTTP / OpenAI-compatible serving

### 1.2 C API

核心头文件：

- `include/llama.h`

这层适合：

- 写自己的推理程序
- 理解 `llama.cpp` 的运行主线
- 对接自定义上层系统

### 1.3 更底层的 ggml / backend API

这层更偏内部实现，不是通常说的“常用 API”。这里先不展开。

## 2. `llama-cli`

最常见命令行程序：

- `llama-cli`

相关文档：

- `tools/cli/README.md`

它的作用就是：

- 加载模型
- 读 prompt
- 直接在本地执行生成

如果只是快速体验 `llama.cpp`，这是最直接的入口。

### 2.1 常见参数

最常见的一批参数：

- `-m`
  模型路径
- `-p`
  prompt
- `-n`
  生成 token 数
- `-c`
  context size
- `-b`
  logical batch size
- `-ub`
  physical batch size
- `-fa`
  Flash Attention 开关
- `-ngl`
  GPU offload 相关

然后是采样相关：

- `--temp`
- `--top-k`
- `--top-p`
- `--repeat-penalty`
- `--mirostat`
- `--grammar`
- `--json-schema`

所以从“使用 API”角度看，`llama-cli` 的 API 本质上就是：

- 一组非常完整的命令行参数面

## 3. `llama-server`

第二个最常用入口：

- `llama-server`

相关文档：

- `tools/server/README.md`

它的定位可以理解成：

- 一个轻量的 HTTP server
- 同时支持原生接口和 OpenAI-compatible 接口

这也是现在很多人把 `llama.cpp` 当成服务框架来用的主要方式。

## 4. `llama-server` 常见 HTTP 接口

最常见的 endpoint 包括：

- `GET /health`
- `POST /completion`
- `POST /embedding`
- `POST /reranking`
- `GET /props`
- `POST /props`
- `GET /slots`

以及 OpenAI-compatible 一组：

- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/embeddings`

其中最常用的通常是：

- `/v1/chat/completions`
- `/v1/completions`
- `/v1/embeddings`

所以如果你是从系统角度使用 `llama.cpp`，重点记住 `llama-server` 和这几个 endpoint 就够了。

## 5. `libllama` C API 的核心对象

真正最值得理解的常用 C API 都在：

- `include/llama.h`

先记住四个核心对象：

- `llama_model`
- `llama_context`
- `llama_batch`
- `llama_sampler`

这四个对象几乎就把主流程串起来了。

## 6. 模型与上下文初始化 API

最常用的一组初始化 API：

- `llama_model_default_params()`
- `llama_context_default_params()`
- `llama_model_load_from_file()`
- `llama_init_from_model()`
- `llama_free_model()`
- `llama_free()`

可以理解成两步：

1. 先加载 `model`
2. 再基于 `model` 创建 `context`

最典型的 C API 使用套路就是：

```c
llama_model_params model_params = llama_model_default_params();
llama_context_params ctx_params = llama_context_default_params();

struct llama_model * model = llama_model_load_from_file(path, model_params);
struct llama_context * ctx = llama_init_from_model(model, ctx_params);
```

## 7. 查询运行时配置的 API

一些非常常用的小查询接口：

- `llama_n_ctx(ctx)`
- `llama_n_ctx_seq(ctx)`
- `llama_n_batch(ctx)`

这些 API 的作用比较直接：

- 看当前 context 支持的上下文和 batch 能力

虽然简单，但阅读代码时经常会遇到。

## 8. Tokenization API

最常见的是：

- `llama_tokenize(...)`

作用：

- 把文本变成 token ids

虽然很多上层程序会包一层更方便的 helper，但底层最核心的还是这个 API。

如果你自己写一个最小推理程序，它通常会出现在很前面。

## 9. `llama_batch`

和推理主线最相关的一组 API：

- `llama_batch_init(...)`
- `llama_batch_free(...)`

作用：

- 创建和释放一次 decode / encode 所需的 batch 结构

它是一次执行请求的直接载体。

如果只看最常见路径，几乎可以把它理解成：

- “这一步准备喂给模型的 token 包”

## 10. 执行 API：`llama_decode`

最关键的执行 API：

- `llama_decode(...)`

这是 `llama.cpp` 生成闭环里最重要的一个函数。

作用：

- 执行一轮 causal decode
- 更新 KV cache
- 产出当前轮 logits

如果你只想抓一个核心函数名，那就是它。

## 11. `llama_encode`

另一个相关 API：

- `llama_encode(...)`

它更适合：

- encoder-only
- embedding
- 非纯 causal decode 场景

对于最典型的 LLM 文本生成主线，`llama_decode` 的优先级更高。

## 12. 取 logits 的 API

decode 执行后，最常见的是读取 logits：

- `llama_get_logits(ctx)`
- `llama_get_logits_ith(ctx, i)`

作用：

- 从 context 中取出当前输出 logits

通常下一步会把它们交给 sampler。

## 13. Sampler API

和生成循环直接相关的另一组核心 API：

- `llama_sampler_chain_init(...)`
- `llama_sampler_chain_add(...)`
- `llama_sampler_sample(...)`
- `llama_sampler_accept(...)`

这是 `llama.cpp` 采样主线最值得记住的一组函数。

它们分别大致对应：

- 创建 sampler 链
- 往链里加 temperature / top-k / top-p 等策略
- 从 logits 中采一个 token
- 接受这个 token，更新内部状态

所以一个最小生成循环，通常就是：

1. `llama_decode`
2. `llama_sampler_sample`
3. `llama_sampler_accept`
4. 再构造 batch
5. 再 `llama_decode`

## 14. 最小 C API 心智模型

如果把最常用的 C API 压到最小，可以记成下面这组：

- `llama_model_default_params`
- `llama_context_default_params`
- `llama_model_load_from_file`
- `llama_init_from_model`
- `llama_tokenize`
- `llama_batch_init`
- `llama_decode`
- `llama_get_logits`
- `llama_sampler_sample`
- `llama_sampler_accept`
- `llama_batch_free`
- `llama_free`
- `llama_free_model`

这基本就覆盖了“从加载到生成”的最短路径。

## 15. 如果从使用角度出发，最值得先记什么

如果你只是“用” `llama.cpp`，建议优先记：

- `llama-cli`
- `llama-server`
- `/v1/chat/completions`
- `/v1/completions`
- `/v1/embeddings`

如果你是“读代码”，建议优先记：

- `llama_model`
- `llama_context`
- `llama_batch`
- `llama_decode`
- `llama_sampler_*`

## 16. 和 vLLM 的 API 风格差异

这一点很值得提前记住。

`vLLM` 的常用 API 更偏：

- Python class
- engine abstraction
- request-oriented

`llama.cpp` 的常用 API 更偏：

- CLI / server
- C struct + function
- decode loop oriented

这也是为什么两个项目都叫“推理框架”，但读起来感觉完全不同。

## 17. 一个最小总结

如果只记一组最重要的 `llama.cpp` 常用 API：

- 使用层：
  `llama-cli`, `llama-server`
- 服务层：
  `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`
- 执行层：
  `llama_model_load_from_file`, `llama_init_from_model`, `llama_batch_init`, `llama_decode`, `llama_get_logits`, `llama_sampler_sample`

这组就足够支撑你继续往主流程里读。
