NVCC ?= /usr/local/cuda-12.4/bin/nvcc
NVCCFLAGS := -O3 -std=c++17 -arch=sm_89
LDFLAGS := -lcublas

CUDA_CORE_TARGETS := fp16_gemm_cuda_core bf16_gemm_cuda_core int8_gemm_cuda_core int4_gemm_cuda_core
TENSOR_CORE_TARGETS := fp16_gemm_tensor_core bf16_gemm_tensor_core int8_gemm_tensor_core int4_gemm_tensor_core
CUBLAS_TARGETS := bf16_gemm_cublas
TARGETS := $(CUDA_CORE_TARGETS) $(TENSOR_CORE_TARGETS) $(CUBLAS_TARGETS)

all: $(TARGETS)

fp16_gemm_cuda_core: fp16_gemm_cuda_core.cu gemm_cuda_core_common.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ fp16_gemm_cuda_core.cu

bf16_gemm_cuda_core: bf16_gemm_cuda_core.cu gemm_cuda_core_common.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ bf16_gemm_cuda_core.cu

int8_gemm_cuda_core: int8_gemm_cuda_core.cu gemm_cuda_core_common.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ int8_gemm_cuda_core.cu

int4_gemm_cuda_core: int4_gemm_cuda_core.cu gemm_cuda_core_common.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ int4_gemm_cuda_core.cu

fp16_gemm_tensor_core: fp16_gemm_tensor_core.cu gemm_tensor_core_common.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ fp16_gemm_tensor_core.cu

bf16_gemm_tensor_core: bf16_gemm_tensor_core.cu gemm_tensor_core_common.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ bf16_gemm_tensor_core.cu

int8_gemm_tensor_core: int8_gemm_tensor_core.cu gemm_tensor_core_common.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ int8_gemm_tensor_core.cu

int4_gemm_tensor_core: int4_gemm_tensor_core.cu gemm_tensor_core_common.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ int4_gemm_tensor_core.cu

bf16_gemm_cublas: bf16_gemm_cublas.cu
	$(NVCC) $(NVCCFLAGS) -o $@ bf16_gemm_cublas.cu $(LDFLAGS)

clean:
	rm -f $(TARGETS)

.PHONY: all clean
