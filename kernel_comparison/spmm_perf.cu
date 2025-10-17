#include "./spmm_test_utils.h"
#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "FlashLLM_SpMM_API.cuh"
#include "Coruscant_SpMM_API.cuh"
#include "Coruscant_STC_SpMM_API.cuh"

void pre_swizzle_A_fp16(const uint16_t* A,   // input  (M×K)
                        uint16_t* A_swz,     // output (M×K)
                        int M, int K)
{
    const int COPY_UNIT_FP16_ROWS = 8;  // 8-row group for ldmatrix.m8n8
    const int SWIZZLE_SHIFT_ELEMS = 3;  // 16B / 2B = 8 elements

    for (int row = 0; row < M; ++row) {
        int mask_elems = (row % COPY_UNIT_FP16_ROWS) << SWIZZLE_SHIFT_ELEMS;
        for (int col = 0; col < K; ++col) {
            int col_swz = col ^ mask_elems;
            if (col_swz < K)
                A_swz[row * K + col_swz] = A[row * K + col];
        }
    }
}


half* create_swizzled_copy(const half* A_h, int M, int K)
{
    // allocate host buffer for swizzled matrix
    half* A_swiz_h = (half*)malloc(sizeof(half) * M * K);
    if (!A_swiz_h) {
        fprintf(stderr, "Error: malloc failed for A_swiz_h\n");
        return NULL;
    }

    // reinterpret pointers as uint16_t (same bit-width)
    const uint16_t* A_u16 = reinterpret_cast<const uint16_t*>(A_h);
    uint16_t* A_swiz_u16  = reinterpret_cast<uint16_t*>(A_swiz_h);

    // perform pre-swizzle
    pre_swizzle_A_fp16(A_u16, A_swiz_u16, M, K);

    return A_swiz_h;
}

int main(int argc, char** argv)
{

if (argc != 6) {
    printf("Wrong Inputs! Correct input format: ./spmm_perf M K N Sparsity SplitK\n");
    return 1;
}
int M_GLOBAL                    = atoi(argv[1]);
int K_GLOBAL                    = atoi(argv[2]);
int N_GLOBAL                    = atoi(argv[3]);
int MATRIX_A_PRUNING_PERCENTAGE = atoi(argv[4]);
int SPLIT_K                     = atoi(argv[5]);

int NEW_WARM_UP_ITERATION = 4;
int NEW_BENCHMARK_ITERATION  = 10;


cublasStatus_t cublas_status;
// cusparseStatus_t  cusparse_status;
// cudaError_t       cuda_error;
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
// Host memory
half* A_h            = NULL;  // row major
half* B_h            = NULL;  // col major huh. 
half* B_Transposed_h = NULL;  // row major
// Device memory
half* A            = NULL;
half* B            = NULL;
half* B_Transposed = NULL;
//
A_h            = (half*)malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
B_h            = (half*)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
B_Transposed_h = (half*)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
if (A_h == NULL || B_h == NULL || B_Transposed_h == NULL) {
    printf("Error in CPU Malloc!\n");
    exit(-1);
}
//cudaMalloc(reinterpret_cast<void**>(&A), sizeof(half) * M_GLOBAL * K_GLOBAL);
cudaMalloc(reinterpret_cast<void**>(&A), 4*sizeof(half) * M_GLOBAL * K_GLOBAL);
cudaMalloc(reinterpret_cast<void**>(&B), sizeof(half) * N_GLOBAL * K_GLOBAL);
cudaMalloc(reinterpret_cast<void**>(&B_Transposed), sizeof(half) * N_GLOBAL * K_GLOBAL);
checkLastCudaError(__LINE__);
if (A == NULL || B == NULL || B_Transposed == NULL) {
    printf("Error in cudaMalloc!\n");
    exit(-1);
}
//
printf("Creating matrix of size %d, %d\n", M_GLOBAL, K_GLOBAL);
init_host_matrices(A_h, B_h, M_GLOBAL, K_GLOBAL, N_GLOBAL, MATRIX_A_PRUNING_PERCENTAGE); 


for (int i = 0; i < K_GLOBAL; i++)
    for (int j = 0; j < N_GLOBAL; j++)
        B_Transposed_h[i * N_GLOBAL + j] = B_h[i + j * K_GLOBAL];
//
// printf("Preparing dense data for GPU...\n");
//cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
cudaError_t err;
for (int r = 0; r < 4; ++r) {
    err = cudaMemcpy(A + r * M_GLOBAL * K_GLOBAL, A_h,
                     M_GLOBAL * K_GLOBAL * sizeof(half),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying A round %d: %s\n", r, cudaGetErrorString(err));
    }
}
err = cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
if (err != cudaSuccess) {
    fprintf(stderr, "Error copying B: %s\n", cudaGetErrorString(err));
}
err = cudaMemcpy(B_Transposed, B_Transposed_h, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
if (err != cudaSuccess) {
    fprintf(stderr, "Error copying B_Transposed: %s\n", cudaGetErrorString(err));
}
checkLastCudaError(__LINE__);
/////////////////////////////////////////////////////////////////////////////////////////////////
printf("Launching CuBlas...\n");
half* D_cublas = NULL;
cudaMalloc(reinterpret_cast<void**>(&D_cublas), sizeof(half) * M_GLOBAL * N_GLOBAL);
if (D_cublas == NULL) {
    printf("Error in spmm_test.cu: line %d cudaMalloc falied\n", __LINE__);
    exit(-1);
}
cudaMemset(D_cublas, 0, sizeof(half) * M_GLOBAL * N_GLOBAL);
cublasHandle_t handle;
cublasCreate(&handle);
cublasSetStream(handle, 0);
cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
cudaDeviceSynchronize();
int              m = M_GLOBAL, n = N_GLOBAL, k = K_GLOBAL;
const float      alpha     = 1.0;
const float      beta      = 0.0;
cublasGemmAlgo_t CuBlasALG = static_cast<cublasGemmAlgo_t>(0); //0 states default algorithm
    
// Warm-up
for (int i = 0; i < NEW_WARM_UP_ITERATION; i++) {
    cublas_status = cublasGemmEx(handle,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     m,
                                     n,
                                     k,
                                     &alpha,
                                     A,
                                     CUDA_R_16F,
                                     k,
                                     B,
                                     CUDA_R_16F,
                                     k,
                                     &beta,
                                     D_cublas,
                                     CUDA_R_16F,
                                     m,
                                     CUDA_R_32F,
                                     CuBlasALG);
        checkCublasError(cublas_status, __LINE__);
    }

// Timing with cache clearing
float cublas_total_time = 0.0f;
for (int i = 0; i < NEW_BENCHMARK_ITERATION; i++) {
    half* A_iter = A + (i % 4) * M_GLOBAL * K_GLOBAL;
    
    cudaEventRecord(start, 0);
    cublasGemmEx(handle,
                     CUBLAS_OP_T, //row major input
                     CUBLAS_OP_N, //col major input
                     m,
                     n,
                     k,
                     &alpha,
                     A_iter,
                     CUDA_R_16F,
                     k, //A's leading dimension: should be m if A is col major
                     B,
                     CUDA_R_16F,
                     k,
                     &beta,
                     D_cublas,
                     CUDA_R_16F,
                     m, //C's leading dimension. m because C is col major by default.
                     CUDA_R_32F,
                     CuBlasALG);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float iteration_time = 0.0f;
    cudaEventElapsedTime(&iteration_time, start, stop);
    cublas_total_time += iteration_time;
}

float milliseconds_cublas_tc = cublas_total_time / NEW_BENCHMARK_ITERATION;
    float tflops_cublas_tc = static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2)
                                                 / (milliseconds_cublas_tc / 1000.))
                             / 1e12;
    half* D_cublas_h = NULL;  // col major
    D_cublas_h       = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (D_cublas_h == NULL) {
        printf("Error in spmm_test.cu: line %d CPU Malloc falied\n", __LINE__);
        exit(-1);
    }
    printf("cublas COMPLETE\n");
    cudaMemcpy(D_cublas_h, D_cublas, sizeof(half) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);  // Col Major
    cudaFree(D_cublas);

int Split_K = SPLIT_K;

///////////////////////////////////////////////////////////////////////////

// Coruscant_STC
printf("Launching Coruscant_STC...\n");
//
half* D_SpMM_Coruscant_STC = NULL;
cudaMalloc(reinterpret_cast<void**>(&D_SpMM_Coruscant_STC), sizeof(half) * M_GLOBAL * N_GLOBAL);
if (D_SpMM_Coruscant_STC == NULL) {
    printf("Error in spmm_test.cu: line %d cudaMalloc falied\n", __LINE__);
    exit(-1);
}
cudaMemset(D_SpMM_Coruscant_STC, 0, sizeof(half) * M_GLOBAL * N_GLOBAL);

uint32_t* idx_STC   = NULL;
uint64_t* bmp_STC   = NULL;
uint32_t* NZ_STC    = NULL;


int nnz_length_STC = InitSparseMatrixA_API_bmp_real(A_h, M_GLOBAL, K_GLOBAL, &bmp_STC, &NZ_STC, &idx_STC);//so this works. 


uint32_t* idx_GPU_STC   = NULL;
uint64_t* bmp_GPU_STC   = NULL;
uint32_t* NZ_GPU_STC    = NULL;

const size_t intArrayLength_STC = (M_GLOBAL * K_GLOBAL / 64)+1;
const size_t uint64ArrayLength_STC = M_GLOBAL * K_GLOBAL / 64;

cudaMalloc((void**)&idx_GPU_STC, intArrayLength_STC * sizeof(uint32_t) * 4);
cudaMalloc((void**)&bmp_GPU_STC, uint64ArrayLength_STC * sizeof(uint64_t) * 4);
cudaMalloc((void**)&NZ_GPU_STC, nnz_length_STC * sizeof(uint32_t) * 4);

// Copy the arrays from host to device
for (int r = 0; r < 4; ++r) {
    err = cudaMemcpy(idx_GPU_STC + r * intArrayLength_STC, idx_STC,
                     intArrayLength_STC * sizeof(uint32_t),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying idx round %d: %s\n", r, cudaGetErrorString(err));
    }
}

for (int r = 0; r < 4; ++r) {
    err = cudaMemcpy(bmp_GPU_STC + r * uint64ArrayLength_STC, bmp_STC,
                     uint64ArrayLength_STC * sizeof(uint64_t),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying bmp round %d: %s\n", r, cudaGetErrorString(err));
    }
}

for (int r = 0; r < 4; ++r) {
    err = cudaMemcpy(NZ_GPU_STC + r * nnz_length_STC, NZ_STC,
                     nnz_length_STC * sizeof(uint32_t),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying NZ round %d: %s\n", r, cudaGetErrorString(err));
    }
}
half* Reduction_Workspace_STC = NULL;
cudaMalloc(reinterpret_cast<void**>(&Reduction_Workspace_STC), sizeof(half) * M_GLOBAL * N_GLOBAL * Split_K);
if (Reduction_Workspace_STC == NULL) {
    printf("Error in cudaMalloc\n");
    exit(-1);
}

// Warm-up
for (int i = 0; i < NEW_WARM_UP_ITERATION; i++) {
    Coruscant_STC_SpMM_SplitK_API(0, A, bmp_GPU_STC, reinterpret_cast<uint4*>(NZ_GPU_STC), idx_GPU_STC, B, D_SpMM_Coruscant_STC, M_GLOBAL, N_GLOBAL, K_GLOBAL, Reduction_Workspace_STC, Split_K);
}

// Timing with cache clearing
float coruscant_stc_total_time = 0.0f;
for (int i = 0; i < NEW_BENCHMARK_ITERATION; i++) {
    uint32_t* NZ_iter_STC = NZ_GPU_STC + (i % 4) * nnz_length_STC;
    uint64_t* bmp_iter_STC = bmp_GPU_STC + (i % 4) * uint64ArrayLength_STC;
    uint32_t* idx_iter_STC = idx_GPU_STC + (i % 4) * intArrayLength_STC;
    
    cudaEventRecord(start, 0);
    Coruscant_STC_SpMM_SplitK_API(0, A, bmp_iter_STC, reinterpret_cast<uint4*>(NZ_iter_STC), idx_iter_STC, B, D_SpMM_Coruscant_STC, M_GLOBAL, N_GLOBAL, K_GLOBAL, Reduction_Workspace_STC, Split_K);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    checkLastCudaError(__LINE__);
    
    float iteration_time = 0.0f;
    cudaEventElapsedTime(&iteration_time, start, stop);
    coruscant_stc_total_time += iteration_time;
}

float milliseconds_SpMM_Coruscant_STC = coruscant_stc_total_time / NEW_BENCHMARK_ITERATION;
float tflops_SpMM_Coruscant_STC =
    static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds_SpMM_Coruscant_STC / 1000.))
    / 1e12;
half* D_SpMM_Coruscant_STC_h = NULL;  // col major
D_SpMM_Coruscant_STC_h       = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
cudaMemcpy(D_SpMM_Coruscant_STC_h, D_SpMM_Coruscant_STC, sizeof(half) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);  // Col Major
cudaFree(D_SpMM_Coruscant_STC);

cudaFree(idx_GPU_STC);
cudaFree(NZ_GPU_STC);
cudaFree(bmp_GPU_STC);
cudaFree(Reduction_Workspace_STC);



/////////////////////////////////////////////////////////////////////////////////////////////////
// Coruscant
printf("Launching Coruscant...\n");
// printf("Preparing Compressed A matrix for GPU kernel: MM_Sparse_TC...\n");
half* D_SpMM_Coruscant = NULL;
cudaMalloc(reinterpret_cast<void**>(&D_SpMM_Coruscant), sizeof(half) * M_GLOBAL * N_GLOBAL);
if (D_SpMM_Coruscant == NULL) {
    printf("Error in spmm_test.cu: line %d cudaMalloc falied\n", __LINE__);
    exit(-1);
}
cudaMemset(D_SpMM_Coruscant, 0, sizeof(half) * M_GLOBAL * N_GLOBAL);

uint32_t* idx   = NULL;
uint64_t* bmp   = NULL;
uint32_t* NZ    = NULL;

 half* A_swiz_h = create_swizzled_copy(A_h, M_GLOBAL, K_GLOBAL);

int nnz_length = InitSparseMatrixA_API_bmp_real(A_swiz_h, M_GLOBAL, K_GLOBAL, &bmp, &NZ, &idx);


uint32_t* idx_GPU   = NULL;
uint64_t* bmp_GPU   = NULL;
uint32_t* NZ_GPU    = NULL;
const size_t intArrayLength = (M_GLOBAL * K_GLOBAL / 64)+1;
const size_t uint64ArrayLength = M_GLOBAL * K_GLOBAL / 64;

cudaMalloc((void**)&idx_GPU, intArrayLength * sizeof(uint32_t) * 4);
cudaMalloc((void**)&bmp_GPU, uint64ArrayLength * sizeof(uint64_t) * 4);
//Round robin access for no L2 caching
cudaMalloc((void**)&NZ_GPU, nnz_length * sizeof(uint32_t) * 4);


// Copy the arrays from host to device
for (int r = 0; r < 4; ++r) {
    err = cudaMemcpy(idx_GPU + r * intArrayLength, idx,
                     intArrayLength * sizeof(uint32_t),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying idx round %d: %s\n", r, cudaGetErrorString(err));
    }
}

for (int r = 0; r < 4; ++r) {
    err = cudaMemcpy(bmp_GPU + r * uint64ArrayLength, bmp,
                     uint64ArrayLength * sizeof(uint64_t),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying bmp round %d: %s\n", r, cudaGetErrorString(err));
    }
}


for (int r = 0; r < 4; ++r) {
    err = cudaMemcpy(NZ_GPU + r * nnz_length, NZ,
                     nnz_length * sizeof(uint32_t),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying NZ round %d: %s\n", r, cudaGetErrorString(err));
    }
}

half* Reduction_Workspace = NULL;
cudaMalloc(reinterpret_cast<void**>(&Reduction_Workspace), sizeof(half) * M_GLOBAL * N_GLOBAL * Split_K);
if (Reduction_Workspace == NULL) {
    printf("Error in cudaMalloc\n");
    exit(-1);
}

// Warm-up
for (int i = 0; i < NEW_WARM_UP_ITERATION; i++) {
    SpMM_SplitK_API(0, A, bmp_GPU, reinterpret_cast<uint4*>(NZ_GPU), idx_GPU, B, D_SpMM_Coruscant, M_GLOBAL, N_GLOBAL, K_GLOBAL, Reduction_Workspace, Split_K);
}

// Timing with cache clearing
float coruscant_total_time = 0.0f;
for (int i = 0; i < NEW_BENCHMARK_ITERATION; i++) {
    uint32_t* NZ_iter = NZ_GPU + (i % 4) * nnz_length;
    uint64_t* bmp_iter = bmp_GPU + (i % 4) * uint64ArrayLength;
    uint32_t* idx_iter = idx_GPU + (i % 4) * intArrayLength;
    cudaEventRecord(start, 0);
    SpMM_SplitK_API(0, A, bmp_iter, reinterpret_cast<uint4*>(NZ_iter), idx_iter, B, D_SpMM_Coruscant, M_GLOBAL, N_GLOBAL, K_GLOBAL, Reduction_Workspace, Split_K);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    checkLastCudaError(__LINE__);
    
    float iteration_time = 0.0f;
    cudaEventElapsedTime(&iteration_time, start, stop);
    coruscant_total_time += iteration_time;
}

float milliseconds_SpMM_Coruscant = coruscant_total_time / NEW_BENCHMARK_ITERATION;
float tflops_SpMM_Coruscant =
    static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds_SpMM_Coruscant / 1000.))
    / 1e12;
half* D_SpMM_Coruscant_h = NULL;  // col major
D_SpMM_Coruscant_h       = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
cudaMemcpy(D_SpMM_Coruscant_h, D_SpMM_Coruscant, sizeof(half) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);  // Col Major
cudaFree(D_SpMM_Coruscant);
double totalError_SpMM_Coruscant  = ComputeTotalError(D_cublas_h, D_SpMM_Coruscant_h, M_GLOBAL, N_GLOBAL);

cudaFree(idx_GPU);
cudaFree(NZ_GPU);
cudaFree(bmp_GPU);
cudaFree(Reduction_Workspace);

/////////////////////////////////////////////////////////////////////////////////////////////////
//FLASHLLM 

half* D_SpMM_FlashLLM = NULL;
cudaMalloc(reinterpret_cast<void**>(&D_SpMM_FlashLLM), sizeof(half) * M_GLOBAL * N_GLOBAL);
if (D_SpMM_FlashLLM == NULL) {
    printf("Error in spmm_test.cu: line %d cudaMalloc falied\n", __LINE__);
    exit(-1);
}
cudaMemset(D_SpMM_FlashLLM, 0, sizeof(half) * M_GLOBAL * N_GLOBAL);
uint32_t* NZWeights_CPU   = NULL;
int*      TileOffsets_CPU = NULL;
int       NumOffsets = InitSparseMatrixA_API(A_h, M_GLOBAL, N_GLOBAL, K_GLOBAL, &NZWeights_CPU, &TileOffsets_CPU);
int       NNZ        = TileOffsets_CPU[NumOffsets - 1] * 4;  // VectorSize = 4
// printf("NumOffsets: %d, NNZ: %d\n", NumOffsets, NNZ);
//
uint32_t* NZWeights_GPU   = NULL;
int*      TileOffsets_GPU = NULL;
cudaMalloc(&TileOffsets_GPU, sizeof(int) * NumOffsets * 4);
if (NNZ == 0)
    NNZ = 1;  // For 100% sparsity, NNZ = 0, malloc will return NULL
//cudaMalloc(&NZWeights_GPU, sizeof(uint32_t) * NNZ);
cudaMalloc(&NZWeights_GPU, sizeof(uint32_t) * NNZ * 4);
if (TileOffsets_GPU == NULL || NZWeights_GPU == NULL) {
    printf("Error in malloc memory from device memory!\n");
    exit(-1);
}
//cudaMemcpy(NZWeights_GPU, NZWeights_CPU, sizeof(uint32_t) * NNZ, cudaMemcpyHostToDevice);
for (int r = 0; r < 4; ++r) {
    err = cudaMemcpy(NZWeights_GPU + r * NNZ, NZWeights_CPU,
                     NNZ * sizeof(uint32_t),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying NZ round %d: %s\n", r, cudaGetErrorString(err));
    }
}
for (int r = 0; r < 4; ++r) {
    err = cudaMemcpy(TileOffsets_GPU + r * NumOffsets, TileOffsets_CPU,
                     NumOffsets * sizeof(int),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying TileOffsets round %d: %s\n", r, cudaGetErrorString(err));
    }
}

// printf("Done! Compressed A matrix for GPU kernel: MM_Sparse_TC.\n");
//
printf("Launching Flash-LLM...\n");
// printf("Split_K = %d\n", Split_K);
Reduction_Workspace = NULL;
cudaMalloc(reinterpret_cast<void**>(&Reduction_Workspace), sizeof(half) * M_GLOBAL * N_GLOBAL * Split_K);
if (Reduction_Workspace == NULL) {
    printf("Error in cudaMalloc\n");
    exit(-1);
}
//
// Warm-up
for (int i = 0; i < NEW_WARM_UP_ITERATION; i++) {
    FlashLLM_SpMM_SplitK_API(0, A, reinterpret_cast<uint4*>(NZWeights_GPU), TileOffsets_GPU, B, D_SpMM_FlashLLM, M_GLOBAL, N_GLOBAL, K_GLOBAL, Reduction_Workspace, Split_K);
}

// Timing with cache clearing
float flashllm_total_time = 0.0f;
for (int i = 0; i < NEW_BENCHMARK_ITERATION; i++) {
    uint32_t* NZWeights_iter = NZWeights_GPU + (i % 4) * NNZ;
    int* TileOffsets_iter = TileOffsets_GPU + (i % 4) * NumOffsets;
    cudaEventRecord(start, 0);
    FlashLLM_SpMM_SplitK_API(0, A, reinterpret_cast<uint4*>(NZWeights_iter), TileOffsets_iter, B, D_SpMM_FlashLLM, M_GLOBAL, N_GLOBAL, K_GLOBAL, Reduction_Workspace, Split_K);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    checkLastCudaError(__LINE__);

    float iteration_time = 0.0f;
    cudaEventElapsedTime(&iteration_time, start, stop);
    flashllm_total_time += iteration_time;
}

float milliseconds_SpMM_FlashLLM = flashllm_total_time / NEW_BENCHMARK_ITERATION;
float tflops_SpMM_FlashLLM =
    static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds_SpMM_FlashLLM / 1000.))
    / 1e12;
half* D_SpMM_FlashLLM_h = NULL;  // col major
D_SpMM_FlashLLM_h       = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
cudaMemcpy(D_SpMM_FlashLLM_h, D_SpMM_FlashLLM, sizeof(half) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);  // Col Major
cudaFree(D_SpMM_FlashLLM);
cudaFree(NZWeights_GPU);
cudaFree(TileOffsets_GPU);
cudaFree(Reduction_Workspace);
free(TileOffsets_CPU);
free(NZWeights_CPU);


/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////


printf("******************************************Problem Size******************************************\n");
printf("M: %d N: %d K: %d Pruning Rate: %d SplitK: %d\n",
        M_GLOBAL,
        N_GLOBAL,
        K_GLOBAL,
        MATRIX_A_PRUNING_PERCENTAGE,
        SPLIT_K);


printf("******************************************Performance*******************************************\n");

//PrintPerformance("CuBlas_SIMT", milliseconds_cublas, tflops_cublas, 0.0);
PrintPerformance("CuBlas_TC", milliseconds_cublas_tc, tflops_cublas_tc, -1);
double totalError_SpMM_FlashLLM  = ComputeTotalError(D_cublas_h, D_SpMM_FlashLLM_h, M_GLOBAL, N_GLOBAL);
PrintPerformance("FlashLLM", milliseconds_SpMM_FlashLLM, tflops_SpMM_FlashLLM, totalError_SpMM_FlashLLM);
PrintPerformance("Coruscant", milliseconds_SpMM_Coruscant, tflops_SpMM_Coruscant, totalError_SpMM_Coruscant);
PrintPerformance("Coruscant_STC", milliseconds_SpMM_Coruscant_STC, tflops_SpMM_Coruscant_STC, -1);

free(idx);
free(bmp);
free(NZ);

free(D_cublas_h);
free(A_h);
free(A_swiz_h);
free(B_h);
free(D_SpMM_FlashLLM_h);  
free(D_SpMM_Coruscant_h);
free(D_SpMM_Coruscant_STC_h);
free(B_Transposed_h);
cudaFree(A);
cudaFree(B);
cudaFree(B_Transposed);
return 0;
}
