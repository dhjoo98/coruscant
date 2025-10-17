//Code refined from FlashLLM (https://github.com/AlibabaResearch/flash-llm)

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

template<typename TilingConfig, typename SparseKernelConfig>
static void Coruscant_STC_SpMM_SplitK_Kernel_Ex(cudaStream_t stream,
                                  const half*  A,
                                  const uint64_t* bmp, 
                                  const uint4* NZ,
                                  //const uint32_t* NZ, 
                                  const uint32_t* idx,
                                  //const uint4* Compressed_A,
                                  //const int*   TileOffsets,
                                  const half*  B,
                                  half*        Reduction_Workspace,
                                  const int    M_Global,
                                  const int    N_Global,
                                  const int    K_Global,
                                  int          Split_K);



cudaError_t Coruscant_STC_SpMM_SplitK_API(cudaStream_t stream,
                            const half*  A,
                            const uint64_t* bmp, 
                            const uint4* NZ, 
                            //const uint32_t* NZ, 
                            const uint32_t* idx,
                            //const uint4* Compressed_A,
                            //const int*   TileOffsets,
                            const half*  B,
                            half*        C,
                            const int    M_Global,
                            const int    N_Global,
                            const int    K_Global,
                            half*        Reduction_Workspace,  // Identical workspace for all SpMM kernel launches
                            int          Split_K);



__host__ int InitSparseMatrixA_API_bmp_real(half *matrix, int rows, int cols, 
                     uint64_t **bitmaps, uint32_t **packed_nonzeros, uint32_t **num_nonzeros);


// Used by ft-tools
extern "C" void GenSparseMatrixBinFile(char* DenseMatrixFileName,
                                       int   M,
                                       int   N,
                                       int   K,
                                       char* NZWeightsFileName,
                                       char* TileOffsetsFileName,
                                       char* OutputSizesFileName);