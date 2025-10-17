/*
Code refined from FlashLLM (https://github.com/AlibabaResearch/flash-llm)
*/
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>
#include <vector>
#include <algorithm>

// Performance Benchmark

#define WARM_UP_ITERATION 100
#define BENCHMARK_ITERATION 10000


void checkCublasError(cublasStatus_t status, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Cublas Error at line %d, Error Code: %d\n", line, status);
        exit(EXIT_FAILURE);
    }
}


#define CHECK_CUDA(func)                                                                                               \
    {                                                                                                                  \
        cudaError_t status = (func);                                                                                   \
        if (status != cudaSuccess) {                                                                                   \
            printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, cudaGetErrorString(status), status);  \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    }


#define CUDA_CALL(code)                                                                                                \
    do {                                                                                                               \
        cudaError_t status = code;                                                                                     \
        std::string err    = cudaGetErrorString(status);                                                               \
        CHECK_EQ(status, cudaSuccess) << "CUDA Error: " << err;                                                        \
    } while (0)

void checkCudaError(cudaError_t error, int line)
{
    if (error != cudaSuccess) {
        printf("Cuda Error at line %d, Error Code: %d\n", line, error);
        exit(EXIT_FAILURE);
    }
}

void checkLastCudaError(int line)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Last Cuda Error Detected at line: %d, Error: %s.\n", line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}


__host__ void //row major (as is expected by cuBLAS Kernel.)
init_host_matrices(half* a, half* b, int M_GLOBAL, int K_GLOBAL, int N_GLOBAL, int MATRIX_A_PRUNING_PERCENTAGE)
{

    for (int i = 0; i < M_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j++) {
            int r = rand() % 100;
            if (r >= MATRIX_A_PRUNING_PERCENTAGE)
                a[j + i * K_GLOBAL] = __float2half_rn(static_cast<float>((rand() % 5)) / 5 - 0.5f);
            else
                a[j + i * K_GLOBAL] = __float2half_rn(0.0f);
        }
    }

    for (int i = 0; i < N_GLOBAL * K_GLOBAL; i++)
        b[i] = __float2half_rn(static_cast<float>((rand() % 5)) / 5 - 0.5f);
}

double ComputeTotalError(half* CuBlas, half* Other, int m, int n)
{
    double totalError = 0.0;
    for (int i = 0; i < m * n; i++){
        totalError += fabs(__half2float(CuBlas[i]) - __half2float(Other[i]));
    }
    return totalError;
}

void PrintPerformance(const char* KernelName, float milliseconds, float tflops, double error)
{
    printf("%-10s \t -> \t\t Time/ms: %5.3f \t Performance/TFLOPs: %4.2f \t TotalError: %.2lf\n",
           KernelName,
           milliseconds,
           tflops,
           error);
}
