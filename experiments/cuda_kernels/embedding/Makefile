NVCC ?= /usr/local/cuda-12.4/bin/nvcc
NVCCFLAGS := -O3 -std=c++17

TARGET := row_gather
SRC := row_gather.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
