NVCC ?= /usr/local/cuda-12.4/bin/nvcc
NVCCFLAGS := -O3 -std=c++17

TARGETS := row_layernorm row_rmsnorm

all: $(TARGETS)

row_layernorm: row_layernorm.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

row_rmsnorm: row_rmsnorm.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
