#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", \
                    __FILE__, __LINE__, status); \
            exit(1); \
        } \
    } while(0)

// LCG随机数生成器（与CPU版本一致）
__device__ double lcg_random(unsigned int* seed) {
    *seed = (1103515245 * (*seed) + 12345) & 0x7fffffff;
    return (double)(*seed) / 0x7fffffff;
}

// CUDA核函数：生成随机矩阵
__global__ void generate_random_matrix(double* d_A, int m_local, int n, int m_start, int seed) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m_local && j < n) {
        unsigned int local_seed = seed + (m_start + i) * n + j;
        d_A[i * n + j] = lcg_random(&local_seed);
    }
}

// 每个GPU生成自己负责的矩阵行并计算
void compute_on_gpu(int gpu_id, int m_start, int m_local, int n,
                   const double* h_x, double* h_y_local, int seed) {
    
    CHECK_CUDA(cudaSetDevice(gpu_id));
    
    // 创建cuBLAS句柄
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    
    // 分配设备内存
    double *d_A, *d_x, *d_y;
    size_t matrix_size = (size_t)m_local * n * sizeof(double);
    
    CHECK_CUDA(cudaMalloc(&d_A, matrix_size));
    CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_y, m_local * sizeof(double)));
    
    // 生成随机矩阵（使用LCG）
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m_local + blockDim.y - 1) / blockDim.y);
    generate_random_matrix<<<gridDim, blockDim>>>(d_A, m_local, n, m_start, seed);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 复制输入向量到GPU
    CHECK_CUDA(cudaMemcpy(d_x, h_x, n * sizeof(double), cudaMemcpyHostToDevice));
    
    // 执行矩阵向量乘法: y = A * x
    double alpha = 1.0, beta = 0.0;
    CHECK_CUBLAS(cublasDgemv(cublas_handle,
                            CUBLAS_OP_T,      // 转置操作
                            n,                // A^T的行数（原A的列数）
                            m_local,          // A^T的列数（原A的行数）
                            &alpha,
                            d_A, n,           // leading dimension
                            d_x, 1,
                            &beta,
                            d_y, 1));
    
    // 将结果复制回主机
    CHECK_CUDA(cudaMemcpy(h_y_local, d_y, m_local * sizeof(double), 
                         cudaMemcpyDeviceToHost));
    
    // 清理资源
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    
    printf("GPU %d: Processed %d rows (rows %d to %d)\n", 
           gpu_id, m_local, m_start, m_start + m_local - 1);
}

extern "C" {

void matrix_vector_mult_gpu(double* h_y, const double* h_x, 
                           int m, int n, int num_gpus, int seed) {
    
    // 计算每个GPU负责的行数
    int rows_per_gpu = m / num_gpus;
    int extra_rows = m % num_gpus;
    
    printf("Starting GPU computation with %d GPUs\n", num_gpus);
    printf("Matrix partitioning: %d rows per GPU, %d extra rows\n", 
           rows_per_gpu, extra_rows);
    
    // 为每个GPU创建流和处理
    #pragma omp parallel for num_threads(num_gpus)
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        int m_start = gpu * rows_per_gpu + (gpu < extra_rows ? gpu : extra_rows);
        int m_local = rows_per_gpu + (gpu < extra_rows ? 1 : 0);
        
        // 每个GPU计算其负责的部分
        compute_on_gpu(gpu, m_start, m_local, n, h_x, h_y + m_start, seed);
    }
    
    // 同步所有GPU
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        CHECK_CUDA(cudaSetDevice(gpu));
        CHECK_CUDA(cudaDeviceSynchronize());
    }
}

int get_gpu_count() {
    int count;
    CHECK_CUDA(cudaGetDeviceCount(&count));
    
    // 打印GPU信息
    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, i));
        printf("GPU %d: %s (%.1f GB)\n", i, prop.name, 
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }
    
    return count;
}

} // extern "C"