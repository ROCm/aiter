# Makefile for Householder tridiagonalization

# Compiler and flags
CXX = hipcc
ROCBLAS_DIR = /opt/rocm
ROCSOLVER_DIR = /opt/rocm

CFLAGS = -O3 -Wall -Wno-unused-result -I${ROCSOLVER_DIR}/include -I${ROCBLAS_DIR}/include --offload-arch=gfx942
LDFLAGS = -L${ROCSOLVER_DIR}/lib -lrocsolver -L${ROCBLAS_DIR}/lib -lrocblas

# Targets
TARGET = sytrd_bench

# Source files
SRC = sytrd_benchmark.cu sytrd_kernels.cu

# Default target
all: $(TARGET)

# Build
$(TARGET): $(SRC)
	$(CXX) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

# Clean
clean:
	rm -f $(TARGET)

# Test
test: $(TARGET)
	./$(TARGET) -n 4 8 16 32 64 128 256 512 1024 2048 4096 8192 -v -i 20 -w 10

.PHONY: all clean test
