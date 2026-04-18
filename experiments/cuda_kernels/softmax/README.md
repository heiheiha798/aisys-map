# Reduction Kernels

这个目录放基础的 `reduction` CUDA kernel 实验。

当前包含：

- `row_softmax.cu`

这份说明会尽量把读者当成：

- 会写 Python
- 知道 softmax 是什么
- 但几乎没认真写过 C++ / CUDA kernel

也就是说，这里不会默认你已经熟悉：

- `threadIdx`
- `blockIdx`
- `shared memory`
- `__syncthreads()`
- reduction pattern

---

## 1. 这个实验到底在干什么

这个例子实现的是：

- `row-wise softmax`

意思是：

- 输入是一个二维矩阵
- 我们对每一行单独做 softmax

如果一行输入是：

```text
[x0, x1, x2, ...]
```

softmax 的输出是：

```text
exp(xi) / sum(exp(xj))
```

但实际实现时，几乎不会直接这样算。  
更常见、也更稳定的做法是：

```text
exp(xi - row_max) / sum(exp(xj - row_max))
```

也就是：

1. 先找到这一行的最大值 `row_max`
2. 每个元素减去这个最大值
3. 再做 `exp`
4. 再把这一行的 `exp` 值加起来
5. 最后每个元素除以总和

所以 softmax 不是一个“每个元素各算各的”操作。  
它至少需要：

- 一次 `max reduction`
- 一次 `sum reduction`

这就是它被放在 `reduction/` 目录的原因。

---

## 2. 为什么 softmax 不是普通的 elementwise kernel

如果是普通的 `elementwise add`：

```cpp
c[i] = a[i] + b[i];
```

每个位置只依赖同一个位置的输入。  
所以一个 thread 很自然地处理一个元素就行。

但 softmax 不一样。

比如你要算第 17 个元素的输出，你不仅需要：

- 第 17 个元素自己的值

还需要：

- 整行的最大值
- 整行所有 `exp` 之后的总和

这就意味着：

- 同一行里的线程必须合作
- 不能真的“每个线程各算各的”

这就是 reduction kernel 最核心的味道：

> 很多线程各自先算一部分，再一起把中间结果汇总起来。

---

## 3. 这份 kernel 的整体分工

这份代码的设计很简单：

- 一个 `block` 负责一整行
- 一个 `thread` 负责这一行里的一个或多个元素

在当前这个例子里：

- `rows = 128`
- `cols = 256`
- `threads_per_block = 256`

所以你可以先粗糙理解成：

- 一行有 256 个元素
- 一个 block 里有 256 个线程
- 所以基本就是“一个线程盯一个列位置”

当然，这份代码写成了更通用的形式：

```cpp
for (int col = tid; col < cols; col += blockDim.x)
```

这表示：

- 如果列数比线程数更多
- 一个 thread 也可以负责多个位置

---

## 4. 先看最重要的 launch 方式

代码里 launch kernel 的地方是：

```cpp
row_softmax_kernel<<<rows, threads_per_block, shared_mem_bytes>>>(d_x, d_y, rows, cols);
```

你可以先这样理解这句：

- 一共启动 `rows` 个 block
- 每个 block 有 `threads_per_block` 个线程
- 每个 block 还额外分配 `shared_mem_bytes` 大小的 shared memory

代入当前参数就是：

- 启动 128 个 block
- 每个 block 256 个线程
- 每个 block 额外拿一块 shared memory

因为每个 block 负责一行，所以：

- block 0 处理第 0 行
- block 1 处理第 1 行
- ...

---

## 5. 从代码第一行开始讲 kernel

### 5.1 `int row = blockIdx.x;`

意思是：

- 当前这个 block 在处理第几行

因为我们是一行一个 block，所以：

- `blockIdx.x` 就直接拿来当行号

### 5.2 `int tid = threadIdx.x;`

意思是：

- 当前 thread 在 block 里的编号

因为 block 里有 256 个线程，所以：

- `tid` 的范围是 `0 ~ 255`

---

## 6. `extern __shared__ float shared[];` 到底是什么

这是很多第一次看 CUDA 的人最容易懵的一句。

先说结论：

> 这句是在声明“一块属于当前 block 的、运行时动态分配的 shared memory”。

大白话理解：

- shared memory 是 block 内线程共享的一小块快内存
- 这块内存不是全局的
- 每个 block 都有自己的一份
- block 内所有线程都能读写它

### 为什么这里要写 `extern`

因为这块 shared memory 的大小不是在代码里写死的，  
而是在 launch kernel 时，从这句里传进去的：

```cpp
<<<rows, threads_per_block, shared_mem_bytes>>>
```

也就是第三个参数：

- `shared_mem_bytes`

它告诉 CUDA：

- 每个 block 需要多少字节的 shared memory

### 这里到底分了多少

代码里写的是：

```cpp
constexpr int shared_mem_bytes = threads_per_block * sizeof(float);
```

也就是：

- `256 * sizeof(float)`

所以每个 block 有一块能放 256 个 `float` 的共享内存。

### 你可以怎么理解它

你可以把它理解成：

> 每个 block 开工之前，先领一块“小白板”，block 里的线程可以一起往上写中间结果。

这块 shared memory 在这份 softmax 里就是用来做 reduction 的中转站。

---

## 7. `float* s_max = shared;` 和 `float* s_sum = shared;` 是什么意思

这两句很容易让人误会成：

- 有两块不同的 shared memory

其实不是。

它们都指向同一块 shared memory。

这是什么意思？

意思是：

- 在第一阶段，这块内存被当成 `s_max`
- 在第二阶段，这块内存又被复用成 `s_sum`

所以你可以理解成：

- 同一块小白板
- 先拿来写“每个线程的局部最大值”
- 后面擦掉再拿来写“每个线程的局部和”

这是节省 shared memory 的做法。

---

## 8. 第一阶段：每个 thread 先找自己的 `local_max`

代码是：

```cpp
float local_max = -INFINITY;
for (int col = tid; col < cols; col += blockDim.x) {
  float val = x[row * cols + col];
  if (val > local_max) {
    local_max = val;
  }
}
```

大白话解释：

- 每个 thread 不可能一上来就知道整行最大值
- 所以先看自己负责的那些元素
- 在自己手里先找一个“局部最大值”

在当前这个例子里：

- 因为 `cols = 256`
- `blockDim.x = 256`

所以大多数 thread 实际上只负责一个元素。

但代码写成这种 `for (col += blockDim.x)` 的形式，是为了更通用。  
以后如果一行更长，一个 thread 也可以负责多个位置。

---

## 9. `s_max[tid] = local_max;` 在做什么

这句的意思是：

- 每个 thread 把自己的局部最大值写到 shared memory 里

所以这时：

- `s_max[0]` 放 thread 0 的局部最大值
- `s_max[1]` 放 thread 1 的局部最大值
- ...

你可以把它理解成：

- 每个人先把自己手里的最好成绩写到白板上

---

## 10. 为什么这里必须 `__syncthreads()`

代码是：

```cpp
__syncthreads();
```

大白话解释：

> 大家先停一下，等所有线程都把自己的局部结果写完，再继续。

如果没有这句，会发生什么？

- 有的线程可能已经开始读 `s_max`
- 但别的线程还没来得及把值写进去

这样读到的结果就可能是错的。

所以在 CUDA 里：

- block 内线程要合作用 shared memory 时
- 很多时候都必须用 `__syncthreads()` 保证大家步调一致

---

## 11. 这段 reduction 到底在干什么

代码是：

```cpp
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
  if (tid < stride) {
    s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
  }
  __syncthreads();
}
```

这段是最经典的 tree reduction。

如果你完全不熟，直接用淘汰赛理解最容易。

### 第一轮

- 前 128 个线程分别和后 128 个线程配对
- 每一对比较大小
- 较大的那个留下来

### 第二轮

- 前 64 个线程再和后 64 个线程配对

### 后面继续

- 32
- 16
- 8
- 4
- 2
- 1

最后：

- `s_max[0]` 就是整行最大值

所以这段的本质就是：

> 把很多线程各自的局部结果，一轮一轮合并成一个全局结果。

这就是 reduction 的最核心模式。

---

## 12. `float row_max = s_max[0];`

现在整行最大值已经算出来了，放在：

- `s_max[0]`

所以接下来 block 里的每个线程都可以把它当成：

- 当前这一行的最大值

---

## 13. 第二阶段：先算 `exp(x - row_max)`，再做局部求和

代码是：

```cpp
float local_sum = 0.0f;
for (int col = tid; col < cols; col += blockDim.x) {
  float exp_val = expf(x[row * cols + col] - row_max);
  y[row * cols + col] = exp_val;
  local_sum += exp_val;
}
```

这里做了两件事：

### 第一件事

把每个位置的：

```text
exp(x - row_max)
```

算出来。

### 第二件事

每个 thread 同时把自己负责的那些 `exp` 值加起来，得到自己的：

- `local_sum`

所以这一阶段结束后：

- `y` 里先暂时存了还没归一化的 `exp` 值
- 每个 thread 手里有一个自己的局部和

你可以理解成：

- 大家先各自把手里的数做 `exp`
- 顺手算出自己这部分的小计

---

## 14. 为什么这里先把值写到 `y`

很多人第一次看会疑惑：

- 为什么不等 `row_sum` 有了再一起写？

这里的写法是：

- 先把 `exp(x - row_max)` 存到 `y`
- 后面知道 `row_sum` 之后，再除一次

这样做的好处是：

- 不用重新算一遍 `exp`

否则你后面归一化时还得重新读输入、重新做 `exp`。

所以这是一个很常见的折中：

- 先把中间结果落一下
- 后面再做最后一步

---

## 15. 第二次 reduction：求整行 `sum`

代码是：

```cpp
s_sum[tid] = local_sum;
__syncthreads();

for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
  if (tid < stride) {
    s_sum[tid] += s_sum[tid + stride];
  }
  __syncthreads();
}
```

这和前面的 `max reduction` 一模一样，只不过：

- 前面是取最大值
- 这里是做加法

最后：

- `s_sum[0]` 就是整行的总和，也就是 `row_sum`

所以 softmax 里为什么有两个 reduction，现在就很清楚了：

1. 先求行最大值
2. 再求行和

---

## 16. 最后归一化

代码是：

```cpp
for (int col = tid; col < cols; col += blockDim.x) {
  y[row * cols + col] /= row_sum;
}
```

现在：

- `y` 里已经存了 `exp(x - row_max)`
- `row_sum` 也已经有了

所以最后每个 thread 再回去，把自己那部分除以总和，就完成了 softmax。

---

## 17. host 侧代码在做什么

除了 kernel 本体，`main()` 里还有几类事情。

### 17.1 准备 host 数据

```cpp
std::vector<float> h_x(...);
std::vector<float> h_y(...);
std::vector<float> h_ref(...);
```

这里：

- `h_x` 是 CPU 上的输入
- `h_y` 是从 GPU 拿回来的输出
- `h_ref` 是 CPU 上自己算出来的参考答案

### 17.2 分配 device memory

```cpp
cudaMalloc(&d_x, bytes);
cudaMalloc(&d_y, bytes);
```

这就是：

- 在 GPU 上开内存

### 17.3 把输入拷到 GPU

```cpp
cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice);
```

就是：

- CPU -> GPU 拷贝

### 17.4 启动 kernel

```cpp
row_softmax_kernel<<<...>>>(...);
```

### 17.5 等 GPU 算完

```cpp
cudaDeviceSynchronize();
```

### 17.6 把结果拷回 CPU

```cpp
cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost);
```

### 17.7 用 CPU 版本做 correctness check

```cpp
cpu_row_softmax(...)
```

也就是：

- 我们自己在 CPU 上再算一次标准答案
- 然后和 GPU 的输出逐项比较

这一步非常重要，因为：

- CUDA kernel 很容易“跑了但结果错了”
- 所以实验阶段一定要做 correctness check

---

## 18. 这个版本为什么是“教学版”，不是高性能版

这份代码的目标不是快，而是清楚。

所以它故意保留了一些“对教学友好、对性能不最优”的点：

- 每个 block 只处理一行
- 用最基础的 shared-memory tree reduction
- 没有做 warp-level reduction
- 没有做 vectorized load/store
- 没有做更激进的 softmax 优化

但正因为它不复杂，才适合你现在这个阶段。

你应该先看懂：

- 为什么要 reduction
- shared memory 怎么当中转站
- `__syncthreads()` 为什么不能少

再去看更复杂的高性能版本。

---

## 19. 一句话总结这份 kernel

这份 `row_softmax` 的核心思路可以压成一句话：

> 一个 block 负责一整行；每个 thread 先处理自己那部分元素；block 内先一起求出整行最大值，再一起求出整行总和，最后再各自完成归一化。

---

## 20. 现在你最该记住的几个点

1. softmax 不是纯 elementwise
2. softmax 至少包含两次 reduction
3. shared memory 是 block 内线程交换中间结果的地方
4. `__syncthreads()` 是为了保证 shared memory 的读写时序正确
5. reduction 的本质就是“大家先各算一部分，再一轮轮合并”

---

## 21. 下一步最自然的问题

如果你已经大致接受这份代码，下一步最自然会问：

- 为什么 tree reduction 要一轮轮减半？
- 为什么这个 kernel 的 occupancy 不高？
- 为什么 `row_softmax` 不像 `elementwise_add` 那么明显 memory-bound？
- warp-level reduction 和 block-level reduction 有什么差别？
- 更高性能的 softmax 会怎么改写？

这些问题就可以作为后续继续扩展的方向。
