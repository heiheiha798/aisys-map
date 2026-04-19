NVCC ?= nvcc
TARGET := elementwise_add
SRC := elementwise_add.cu

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -O2 -std=c++17 $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
