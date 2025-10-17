//Code refined from FlashLLM (https://github.com/AlibabaResearch/flash-llm)


#include "./MatMulUtilities.cuh"
#include "./Reduction_Kernel.cuh"
#include "./SpMM_Kernel.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

void print_packed_halfs(uint32_t packed_value) {
    // Extract the first half (lower 16 bits)
    half first_half = (half)(packed_value & 0xFFFF);  // Mask to get the lower 16 bits

    // Extract the second half (upper 16 bits)
    half second_half = (half)((packed_value >> 16) & 0xFFFF);  // Shift right and mask to get the upper 16 bits

    // Print the two half values
    printf("First half: %f\n", __half2float(first_half));  // Convert half to float for readable output
    printf("Second half: %f\n", __half2float(second_half));
}

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
                                  int          Split_K)
{
    static int SHMEM_SZ = max((TilingConfig::TILE_M * TILE_K + TilingConfig::TILE_N * TILE_K) * sizeof(half) * 2,
                              (TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C) * TilingConfig::TILE_N * sizeof(float));
    cudaFuncSetAttribute(
        Coruscant_STC_SpMM_Kernel<TilingConfig, SparseKernelConfig>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    // printf("Max shared memory size: %d B\n", SHMEM_SZ);
    int dimN =
        max(N_Global / TilingConfig::TILE_N, 1);  // max(N_Global/TilingConfig::TILE_N,1) used when N=8, TILE_N=16
    int  dimM = M_Global * Split_K / TilingConfig::TILE_M;
    dim3 GridDim(dimN, dimM, 1);  // Grid Size is increased due to SplitK for higher SM occupancy
        //each M tiled row handled by SplitK TBs.
    dim3 BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS, 1, 1);

    //std::cout << "----SpMM_SplitK_Kernel_Ex(): Shared Memory Size: " << SHMEM_SZ << " Bytes" << std::endl;
    //std::cout << "----SpMM_SplitK_Kernel_Ex(): GridDim: " << dimN << "x" << dimM << " BlockDim: " << WARP_SIZE * TilingConfig::BLOCK_WARPS << "x1x1" << std::endl;
        // GridDim: 1x196: (7168/256) * 7(Split_K)
    // stream is just the GPU job_ID.
    Coruscant_STC_SpMM_Kernel<TilingConfig, SparseKernelConfig><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(
        A, bmp, NZ, idx,//Compressed_A, TileOffsets, 
        B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);
}



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
                            int          Split_K)
{
#ifdef DEBUG_MODE
    printf("--- SpMM_API.cu/SpMM_SplitK_API(): Entering SpMM_SplitK_API----\n");
    printf(
        "SpMM_API.cu->SpMM_SplitK_API():  M: %d, N: %d, K: %d, SplitK: %d \n", M_Global, N_Global, K_Global, Split_K);
    assert(K_Global % TILE_K == 0);
    assert(M_Global % 256 == 0);
#endif
    half* SpMM_SplitK_OutputPTR;
    if (Split_K == 1)
        SpMM_SplitK_OutputPTR = C;
    else
        SpMM_SplitK_OutputPTR = Reduction_Workspace;
    // Batched SpMM
    //printf("Beginning of SpMM_SplitK_Kernel_Ex, N_Global is %d\n", N_Global); donghyeon: it's just the input.
    switch (N_Global) {
        case 1:
            Coruscant_STC_SpMM_SplitK_Kernel_Ex<TilingConfig<4, 1, 1, 1>, SparseKernelConfig<64>>(
                stream, A, bmp, NZ, idx,//Compressed_A, TileOffsets, 
                B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        //gotta pad case 1. 
        case 4:
            Coruscant_STC_SpMM_SplitK_Kernel_Ex<TilingConfig<4, 1, 1, 1>, SparseKernelConfig<64>>(
                stream, A, bmp, NZ, idx,//Compressed_A, TileOffsets, 
                B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        case 8:
            Coruscant_STC_SpMM_SplitK_Kernel_Ex<TilingConfig<4, 1, 1, 1>, SparseKernelConfig<64>>(
                stream, A, bmp, NZ, idx,//Compressed_A, TileOffsets, 
                B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        
        case 16:
            Coruscant_STC_SpMM_SplitK_Kernel_Ex<TilingConfig<4, 1, 1>, SparseKernelConfig<64>>(
                stream, A, bmp, NZ, idx,//Compressed_A, TileOffsets, 
                B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        
        case 32:
            Coruscant_STC_SpMM_SplitK_Kernel_Ex<TilingConfig<4, 1, 2>, SparseKernelConfig<64>>(
                stream, A, bmp, NZ, idx,//Compressed_A, TileOffsets, 
                B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;

        
        case 64:
            Coruscant_STC_SpMM_SplitK_Kernel_Ex< TilingConfig<4, 1, 4>, SparseKernelConfig<64>>(
            //SpMM_SplitK_Kernel_Ex<TilingConfig<2, 2, 2>, SparseKernelConfig<32>>(
                stream, A, bmp, NZ, idx,//Compressed_A, TileOffsets, 
                B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        
        
        case 96:
            Coruscant_STC_SpMM_SplitK_Kernel_Ex<TilingConfig<4, 1, 2>, SparseKernelConfig<64>>(
                stream, A, bmp, NZ, idx,//Compressed_A, TileOffsets, 
                B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;



        case 128:
            Coruscant_STC_SpMM_SplitK_Kernel_Ex< TilingConfig<4, 1, 4>, SparseKernelConfig<64>>(
            //SpMM_SplitK_Kernel_Ex<TilingConfig<2, 2, 2>, SparseKernelConfig<32>>(
                stream, A, bmp, NZ, idx,//Compressed_A, TileOffsets, 
                B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        
        
        default:
            if (N_Global % 64 == 0) {
            Coruscant_STC_SpMM_SplitK_Kernel_Ex<TilingConfig<4, 1, 4>, SparseKernelConfig<64>>(
                stream, A, bmp, NZ, idx,//Compressed_A, TileOffsets, 
                B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
            }
            else if (N_Global % 32 == 0 || N_Global <= 8) {
            Coruscant_STC_SpMM_SplitK_Kernel_Ex<TilingConfig<4, 1, 1>, SparseKernelConfig<64>>(
                stream, A, bmp, NZ, idx,//Compressed_A, TileOffsets, 
                B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
            }
            else {
                printf("MM_Sparse_API Error: Unsupported N dimension %d!\n", N_Global);
                return cudaErrorUnknown;
            }

    }
    //
    cudaError_t Error = cudaGetLastError();
    if (Error != cudaSuccess)
        return Error;

    if (Split_K == 1)
        return Error;
    dim3 GridDim((M_Global * N_Global) / 256, 1, 1);
    dim3 BlockDim(WARP_SIZE, 1, 1);
    Coruscant_STC_SplitK_Reduction<<<GridDim, BlockDim, 0, stream>>>(C, Reduction_Workspace, M_Global, N_Global, Split_K);
    return cudaGetLastError();
}

__host__ int InitSparseMatrixA_API_bmp_real(half *matrix, int rows, int cols, 
                     uint64_t **bitmaps, uint32_t **packed_nonzeros, uint32_t **num_nonzeros) {
    int tile_size = 64;
    int num_tiles_row = cols;  // Number of tiles per row
    int num_tiles_col = rows / tile_size;  // Number of tiles per column
    int total_tiles = num_tiles_row * num_tiles_col;         // Total number of tiles
    printf("num_tiles_per_row : %d\n", num_tiles_row);
    printf("num_tiles_per_col : %d\n", num_tiles_col);

    int nnz_capacity = rows * cols;  // Max capacity for packed nonzeros (before padding)

    // If pointers are NULL, allocate memory dynamically
    if (*bitmaps == NULL) {
        *bitmaps = (uint64_t*)malloc(total_tiles * sizeof(uint64_t));
        if (*bitmaps == NULL) {
            fprintf(stderr, "Failed to allocate memory for bitmaps.\n");
            exit(EXIT_FAILURE);
        }
    }
    if (*packed_nonzeros == NULL) {
        *packed_nonzeros = (uint32_t*)malloc(nnz_capacity * sizeof(uint32_t));
        if (*packed_nonzeros == NULL) {
            fprintf(stderr, "Failed to allocate memory for packed nonzeros.\n");
            exit(EXIT_FAILURE);
        }
    }
    memset(*packed_nonzeros, 0, nnz_capacity * sizeof(uint32_t));
    if (*num_nonzeros == NULL) {
        *num_nonzeros = (uint32_t*)malloc((total_tiles+1) * sizeof(uint32_t)); //then discard the last element.
        if (*num_nonzeros == NULL) {
            fprintf(stderr, "Failed to allocate memory for num_nonzeros.\n");
            exit(EXIT_FAILURE);
        }
    }

    int tile_count = 0;  // To track the current tile being processed
    int nnz_count = 0;   // To track the current number of packed nonzeros (uint32_t)

    (*num_nonzeros)[0] = 0;

    for (int tile_col = 0; tile_col < num_tiles_col; ++tile_col) {
        for (int tile_row = 0; tile_row < num_tiles_row; ++tile_row) { //number of tiles per row. for 256x256, 256
         //number of tiles per column //for 256x256, 4
            uint64_t bitmap = 0;   // Bitmap for the current 8x8 tile
            int non_zero_count = 0; // Counter for nonzeros in the current tile
            uint32_t packed_value = 0;  // To store two half values packed into one uint32_t
            int half_count = 0;         // To count how many halfs are packed

            //looping over 64x1 tiles. -> 1x64 tiles. 
            for (int i = 0; i < tile_size; ++i) {    
                    int row = tile_col * tile_size + i;
                    int col = tile_row;
                    int pos = i;  // Position in the 64-bit bitmap

                    // Ensure we don't go out of bounds
                    if (row < rows && col < cols) {
                        half value = matrix[row * cols + col];
                        if (__half2float(value) != 0.0f) {
                            //bitmap |= (1ULL << pos);   // Set bit if non-zero
                            bitmap |= (1ULL << (63 - pos));
                            uint16_t raw_half_value = *(uint16_t *)&value;
                            
                            if (half_count == 0) {
                                // Pack the first half into the lower 16 bits
                                packed_value = (uint32_t)raw_half_value;  // Store the first half
                                //printf("Packing value to first half: 0x%x\n", raw_half_value);
                                (half_count)++;
                            } else {
                                // Pack the second half into the upper 16 bits
                                packed_value |= ((uint32_t)raw_half_value << 16);  // Shift and store the second half
                                //printf("Packing value to second half: 0x%x\n", raw_half_value);
                                (*packed_nonzeros)[nnz_count++] = packed_value;  // Store packed value
                                half_count = 0;  // Reset for the next pair of half values
                            }
                            non_zero_count++;
                        }
                    }
                }
            if (half_count == 1) {
                (*packed_nonzeros)[nnz_count++] = packed_value;  // Pad with one half
                non_zero_count++;
            }

            // Pad non-zeros to a multiple of 8 (our vector size is uint4)
            int padding_needed = (non_zero_count % 8 == 0) ? 0 : (8 - (non_zero_count % 8));
            for (int pad = 0; pad < padding_needed; pad += 2) {
                (*packed_nonzeros)[nnz_count++] = 0;  // Add padding zeros (two halfs packed into one uint32_t)
            }

            (*bitmaps)[tile_count] = bitmap;           // Store bitmap for this tile
            (*num_nonzeros)[tile_count + 1] = (*num_nonzeros)[tile_count] + (non_zero_count + padding_needed)/2;
            //printf("bitmap 0x%016llx\n", bitmap);
            tile_count++;
        }
    }

   *packed_nonzeros = (uint32_t*)realloc(*packed_nonzeros, ((*num_nonzeros)[tile_count]) * sizeof(uint32_t));
   printf("Tile Count within bmp transformation: %d\n", tile_count);
return (*num_nonzeros)[tile_count]; //we already regard num_nonzeros as the number of packed uint32_t.
}

