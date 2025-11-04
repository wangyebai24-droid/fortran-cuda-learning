# Makefile for Matrix-Vector Multiplication Benchmark

# 编译器
CC = gcc
CXX = g++
NVCC = nvcc

# 编译选项 - 针对AMD EPYC优化
CFLAGS = -O3 -fopenmp -march=znver3 -mtune=znver3  # znver3 for EPYC 7xx3
NVCCFLAGS = -O3 -arch=sm_89 -Xcompiler -fopenmp  # sm_89 for RTX 4090
LDFLAGS = -lm -fopenmp

# Intel MKL (已确认安装在 /opt/intel/mkl)
MKLROOT = /opt/intel/mkl
BLAS_FLAGS = -m64 -I${MKLROOT}/include
BLAS_LIBS = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl

# CUDA库 - 使用CUDA 11.8以匹配nvcc版本
CUDA_PATH = /usr/local/cuda-11.8
CUDA_LIBS = -L$(CUDA_PATH)/targets/x86_64-linux/lib -lcudart -lcublas
# 目标文件
TARGET = matrix_vector_test
OBJS = main.o matrix_vector_cpu.o matrix_vector_gpu.o

# 默认目标
all: $(TARGET)

# 链接
$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(BLAS_LIBS) $(CUDA_LIBS)

# 编译主程序
main.o: main.c
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -c $< -o $@

# 编译CPU版本
matrix_vector_cpu.o: matrix_vector_cpu.c
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -c $< -o $@

# 编译GPU版本
matrix_vector_gpu.o: matrix_vector_gpu.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# 清理
clean:
	rm -f $(OBJS) $(TARGET)

# 运行测试 - 调整为更大规模
test: $(TARGET)
	OMP_NUM_THREADS=64 ./$(TARGET) 50000 50000 2

# 运行小规模测试
test_small: $(TARGET)
	OMP_NUM_THREADS=32 ./$(TARGET) 20000 20000 2

# 运行大规模测试 - 充分利用256GB内存
test_large: $(TARGET)
	OMP_NUM_THREADS=128 ./$(TARGET) 100000 100000 2

# 运行超大规模测试 - 挑战极限
test_huge: $(TARGET)
	OMP_NUM_THREADS=128 ./$(TARGET) 150000 150000 2

# 运行单GPU测试
test_single: $(TARGET)
	OMP_NUM_THREADS=64 ./$(TARGET) 60000 60000 1

# 性能基准测试套件
benchmark: $(TARGET)
	@echo "Running comprehensive benchmark suite..."
	@echo "Test 1: 30k x 30k"
	@OMP_NUM_THREADS=32 ./$(TARGET) 30000 30000 2
	@echo "\nTest 2: 60k x 60k"
	@OMP_NUM_THREADS=64 ./$(TARGET) 60000 60000 2
	@echo "\nTest 3: 90k x 90k"
	@OMP_NUM_THREADS=128 ./$(TARGET) 90000 90000 2

.PHONY: all clean test test_small test_large test_huge test_single benchmark