NVCC ?= nvcc
TARGET := row_softmax
SRC := row_softmax.cu

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -O2 -std=c++17 $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
