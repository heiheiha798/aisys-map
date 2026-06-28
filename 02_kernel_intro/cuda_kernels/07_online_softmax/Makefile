NVCC ?= /usr/local/cuda-12.4/bin/nvcc
TARGET := row_softmax_online
SRC := row_softmax_online.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -O3 -std=c++17 -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
