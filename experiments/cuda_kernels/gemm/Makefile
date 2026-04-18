NVCC ?= /usr/local/cuda-12.4/bin/nvcc
NVCCFLAGS := -O3 -std=c++17 -arch=sm_89
LDFLAGS := -lcublas

CUDA_CORE_TARGETS := bf16_gemm_cuda_core
TENSOR_CORE_TARGETS := bf16_gemm_tensor_core
CUBLAS_TARGETS := bf16_gemm_cublas
TARGETS := $(CUDA_CORE_TARGETS) $(TENSOR_CORE_TARGETS) $(CUBLAS_TARGETS)

all: $(TARGETS)

bf16_gemm_cuda_core: bf16_gemm_cuda_core.cu
	$(NVCC) $(NVCCFLAGS) -o $@ bf16_gemm_cuda_core.cu

bf16_gemm_tensor_core: bf16_gemm_tensor_core.cu
	$(NVCC) $(NVCCFLAGS) -o $@ bf16_gemm_tensor_core.cu

bf16_gemm_cublas: bf16_gemm_cublas.cu
	$(NVCC) $(NVCCFLAGS) -o $@ bf16_gemm_cublas.cu $(LDFLAGS)

clean:
	rm -f $(TARGETS)

.PHONY: all clean
