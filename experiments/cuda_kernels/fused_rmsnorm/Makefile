NVCC ?= /usr/local/cuda-12.4/bin/nvcc
NVCCFLAGS := -O3 -std=c++17

TARGETS := fused_residual_rmsnorm

all: $(TARGETS)

fused_residual_rmsnorm: fused_residual_rmsnorm.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
