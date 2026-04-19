NVCC ?= /usr/local/cuda-12.4/bin/nvcc
TARGET := attention
SRC := attention.cu

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -O3 -std=c++17 $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
