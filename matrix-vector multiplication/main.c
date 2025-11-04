#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

// 外部函数声明
extern void matrix_vector_mult_cpu(double* result, const double* vector, 
                                  int m, int n, int seed);
extern void matrix_vector_mult_gpu(double* result, const double* vector,
                                  int m, int n, int num_gpus, int seed);
extern int get_gpu_count();

// 获取当前时间（秒）
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1.0e-6;
}

// 计算相对误差
double compute_relative_error(const double* vec1, const double* vec2, int n) {
    double max_err = 0.0;
    double max_val = 0.0;
    
    for (int i = 0; i < n; i++) {
        double err = fabs(vec1[i] - vec2[i]);
        double val = fabs(vec1[i]);
        if (err > max_err) max_err = err;
        if (val > max_val) max_val = val;
    }
    
    return (max_val > 0) ? (max_err / max_val) : max_err;
}

int main(int argc, char* argv[]) {
    // 默认参数
    int m = 50000;      // 矩阵行数
    int n = 50000;      // 矩阵列数
    int num_gpus = 2;   // GPU数量（默认2个4090）
    int seed = 42;      // 随机种子
    
    // 解析命令行参数
    if (argc > 1) m = atoi(argv[1]);
    if (argc > 2) n = atoi(argv[2]);
    if (argc > 3) num_gpus = atoi(argv[3]);
    if (argc > 4) seed = atoi(argv[4]);
    
    printf("========================================\n");
    printf("Matrix-Vector Multiplication Benchmark\n");
    printf("========================================\n");
    printf("Matrix size: %d x %d\n", m, n);
    printf("Matrix memory size: %.2f GB\n", 
           (double)m * n * sizeof(double) / (1024.0 * 1024.0 * 1024.0));
    printf("Random seed: %d\n", seed);
    printf("\n");
    
    // 检查GPU可用性
    int available_gpus = get_gpu_count();
    printf("Available GPUs: %d\n", available_gpus);
    if (num_gpus > available_gpus) {
        printf("Warning: Requested %d GPUs, but only %d available. Using %d GPUs.\n", 
               num_gpus, available_gpus, available_gpus);
        num_gpus = available_gpus;
    }
    printf("Using %d GPUs for computation\n\n", num_gpus);
    
    // 分配内存
    double *x = (double*)malloc(n * sizeof(double));
    double *y_cpu = (double*)calloc(m, sizeof(double));
    double *y_gpu = (double*)calloc(m, sizeof(double));
    
    if (!x || !y_cpu || !y_gpu) {
        fprintf(stderr, "Error: Failed to allocate memory\n");
        exit(1);
    }
    
    // 初始化输入向量
    printf("Initializing input vector...\n");
    srand(seed);
    for (int i = 0; i < n; i++) {
        x[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;  // [-1, 1]范围
    }
    
    // CPU计算
    printf("\n--- CPU Computation ---\n");
    double cpu_start = get_time();
    matrix_vector_mult_cpu(y_cpu, x, m, n, seed);
    double cpu_end = get_time();
    double cpu_time = cpu_end - cpu_start;
    printf("CPU time: %.3f seconds\n", cpu_time);
    
    // GPU计算
    printf("\n--- GPU Computation ---\n");
    double gpu_start = get_time();
    matrix_vector_mult_gpu(y_gpu, x, m, n, num_gpus, seed);
    double gpu_end = get_time();
    double gpu_time = gpu_end - gpu_start;
    printf("GPU time: %.3f seconds\n", gpu_time);
    
    // 验证结果
    printf("\n--- Verification ---\n");
    double relative_error = compute_relative_error(y_cpu, y_gpu, m);
    printf("Relative error: %.6e\n", relative_error);
    
    if (relative_error < 1e-6) {
        printf("✓ Results match within acceptable tolerance\n");
    } else {
        printf("✗ Warning: Large error detected!\n");
    }
    
    // 性能总结
    printf("\n--- Performance Summary ---\n");
    printf("CPU time: %.3f seconds\n", cpu_time);
    printf("GPU time: %.3f seconds\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    
    // 清理内存
    free(x);
    free(y_cpu);
    free(y_gpu);
    
    return 0;
}