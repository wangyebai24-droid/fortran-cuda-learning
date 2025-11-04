#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mkl.h>  // 使用MKL头文件，包含了cblas

// 简单的线性同余随机数生成器（用于可重现的并行随机数）
double lcg_random(unsigned int* seed) {
    *seed = (1103515245 * (*seed) + 12345) & 0x7fffffff;
    return (double)(*seed) / 0x7fffffff;
}

void matrix_vector_mult_cpu(double* result, const double* vector, 
                           int m, int n, int seed) {
    
    int num_threads = omp_get_max_threads();
    printf("CPU: Using %d OpenMP threads with Intel MKL\n", num_threads);
    
    // 初始化结果向量
    memset(result, 0, m * sizeof(double));
    
    // 使用更大的块以减少开销
    int rows_per_thread = (m + num_threads - 1) / num_threads;
    
    // 每个线程处理一部分行
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int start_row = tid * rows_per_thread;
        int end_row = (start_row + rows_per_thread > m) ? m : start_row + rows_per_thread;
        
        if (start_row < m) {
            int local_rows = end_row - start_row;
            
            // 为每个线程分配本地矩阵块（考虑cache友好性）
            double* local_matrix = (double*)aligned_alloc(64, local_rows * n * sizeof(double));
            unsigned int local_seed = seed + tid;
            
            // 生成该线程负责的所有行
            for (int i = 0; i < local_rows; i++) {
                for (int j = 0; j < n; j++) {
                    local_seed = seed + (start_row + i) * n + j;
                    local_matrix[i * n + j] = lcg_random(&local_seed); 
                }
            }
            
            // 使用MKL的批量GEMV操作（更高效）
            // 计算 result[start_row:end_row] = local_matrix * vector
            cblas_dgemv(CblasRowMajor, CblasNoTrans, 
                       local_rows, n,
                       1.0,                    // alpha
                       local_matrix, n,        // A, lda
                       vector, 1,              // x, incx
                       0.0,                    // beta
                       result + start_row, 1); // y, incy
            
            free(local_matrix);
            
            // 进度报告（仅主线程）
            if (tid == 0) {
                printf("CPU: Thread 0 completed %d rows\n", local_rows);
            }
        }
    }
    
    printf("CPU: Computation completed\n");
}