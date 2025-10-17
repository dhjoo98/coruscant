//Code refined from FlashLLM (https://github.com/AlibabaResearch/flash-llm)

#include "MatMulUtilities.cuh"
#include <vector>

#define DEBUG 0
#define DEBUG2 0



template<typename TilingConfig, typename SparseKernelConfig>

__device__ __forceinline__ void Coruscant_STC_SpMM_CopyFromGlobalToReg(//uint32_t* Registers_nz,
                                                         uint32_t    Registers_nz[64],
                                                         uint64_t*    Registers_bmp,
                                                         uint32_t*    Registers_nnz,
                                                         //const uint32_t* GlobalPTR_nz,
                                                         const uint4* GlobalPTR_nz,
                                                         const uint64_t* GlobalPTR_bmp,
                                                         const uint32_t* GlobalPTR_nnz, 
                                                         uint32_t* nnz_tile0, 
                                                         uint32_t* nnz_tile1,
                                                         int startTileIdx) 
{

constexpr int MAX_NZ_PER_BMP_div_2_4 = 8; //first divide by 2 for half, then divide by 4 for uint4. : 64 / 8 = 8
   
    // Each thread handles 2 bitmaps (each of a column)

    #if DEBUG2
        if (blockIdx.x == 0 && blockIdx.y == 383 && threadIdx.x == 127) { //[7168, 7168, 8]  //383
            printf("------Check inside Reg load...\n");
            printf("StartTileIdx: %d\n", startTileIdx);
            printf("bmp0: %u\n", GlobalPTR_bmp[startTileIdx]);
            printf("nnz0: %u\n", GlobalPTR_nnz[startTileIdx]);
        }
    #endif
#pragma unroll     
    for (int i = 0; i < 2; i++) {
        int globalTileIdx = startTileIdx + i;
        // Load bitmap
        Registers_bmp[i] = GlobalPTR_bmp[globalTileIdx];
        Registers_nnz[i] = GlobalPTR_nnz[globalTileIdx]; 

        // Load non-zero values into the register
        uint32_t num_nz_per_bitmap = __popcll(Registers_bmp[i]);
        if (i){
            *nnz_tile1 = num_nz_per_bitmap; //This is the number of halfs
        }
        else{
            *nnz_tile0 = num_nz_per_bitmap;
        }

        // Load non-zero elements (half precision) into the register
#pragma unroll 
        for (int j = 0; j < MAX_NZ_PER_BMP_div_2_4 ; j++) { //8 iterations to copy the 4 x packed two fp16s.
            //loading Vectors 
            if (j <= num_nz_per_bitmap / 8 ) {
            //if (j < num_nz_per_bitmap / 8 ) {
                //**Registers_nnz is in 'uint32' units. 
                Registers_nz[i * 32 + j * 4 + 0] = GlobalPTR_nz[Registers_nnz[i] / 4 + j].x; // load nz
                Registers_nz[i * 32 + j * 4 + 1] = GlobalPTR_nz[Registers_nnz[i] / 4 + j].y; // load nz
                Registers_nz[i * 32 + j * 4 + 2] = GlobalPTR_nz[Registers_nnz[i] / 4 + j].z; // load nz
                Registers_nz[i * 32 + j * 4 + 3] = GlobalPTR_nz[Registers_nnz[i] / 4 + j].w; // load nz
                //Registers_nz[i * 32 + j * 4 + 0] = GlobalPTR_nz[Registers_nnz[i] / 8 + j].x; // load nz
                //Registers_nz[i * 32 + j * 4 + 1] = GlobalPTR_nz[Registers_nnz[i] / 8 + j].y; // load nz
                //Registers_nz[i * 32 + j * 4 + 2] = GlobalPTR_nz[Registers_nnz[i] / 8 + j].z; // load nz
                //Registers_nz[i * 32 + j * 4 + 3] = GlobalPTR_nz[Registers_nnz[i] / 8 + j].w; // load nz
            }
        }
    }
}

// Init Shared Memory to 0
template<typename TilingConfig>
__device__ __forceinline__ void Coruscant_STC_SpMM_InitSharedMemory(half* __restrict__ SharedPTR)
{
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    //
    static_assert(TilingConfig::TILE_M % TilingConfig::BLOCK_WARPS == 0,
                  "TILE_M must be an integer multiple to BLOCK_WARPS");
    constexpr int RowsPerWarp = TilingConfig::TILE_M / TilingConfig::BLOCK_WARPS;
    //
    static_assert(TILE_K == 64, "For now, TILE_K is assumed to be 64.\n");
    const int StartRowNum         = warp_id * RowsPerWarp;
    half*     SharedPTR_PerThread = SharedPTR + StartRowNum * TILE_K + HALF_PER_128B * lane_id;
    //
    static_assert(RowsPerWarp % (WARP_SIZE * HALF_PER_128B / TILE_K) == 0,
                  "RowsPerWarp%(WARP_SIZE*HALF_PER_128B/TILE_K) should be 0\n");
    constexpr int ITERATIONS_PER_THREAD = RowsPerWarp / (WARP_SIZE * HALF_PER_128B / TILE_K);
#pragma unroll
    for (int i = 0; i < ITERATIONS_PER_THREAD; i++) {
        cp_async_ignore_src<16>(SharedPTR_PerThread, (half*)NULL);
        SharedPTR_PerThread += WARP_SIZE * HALF_PER_128B;
    }
}

template<typename TilingConfig, typename SparseKernelConfig>
__device__ __forceinline__ void Coruscant_STC_SpMM_DecompressFromRegisterToShared(half* __restrict__ SharedPTR,
                                                                    uint32_t Registers_nz[64],
                                                                    uint64_t* Registers_bmp,
                                                                    uint32_t* nnz_tile0, 
                                                                    uint32_t* nnz_tile1,
                                                                    int TB_ROW, 
                                                                    int TB_COL,
                                                                    float* c)
                                                                    //int tileIdx)
{

    //int access = threadIdx.x / 16;
#pragma unroll
    for (int i = 0; i < 8; i++){
        c[i] += *reinterpret_cast<float*>(&Registers_nz[8*i]);
        }
}



template<typename TilingConfig, typename SparseKernelConfig>
__global__ void 
//__maxnreg__(255)
Coruscant_STC_SpMM_Kernel(const half*  A,
                            const uint64_t* bmp, 
                            //const uint32_t* NZ,
                            const uint4* NZ, 
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


    //
    const int BatchID     = blockIdx.y / (M_Global / TilingConfig::TILE_M); //M_Global / TILE_M: tiling the M dimension of Matrix A.
    const int IsLastBatch = (BatchID == (Split_K - 1));
    const int x           = blockIdx.x; //block DimX is 1 for skinny matrices (see SpMM_API/line 42)
    const int y           = blockIdx.y % (M_Global / TilingConfig::TILE_M);  //blockIdx.y % (num M Tile rows): wrap around num_tile_rows
        //i.e., TB0, TB(num M tile rows), TB(2*num M tile rows) .. handle the first M tile row
    //
    const int NumKBlock        = K_Global / TILE_K;  // assert (K_Global%TILE_K==0);
    const int AverageNumKBlock = (NumKBlock - 1) / Split_K + 1;
    const int RoundedKBlock    = AverageNumKBlock * Split_K;
    const int PaddingKBlock    = RoundedKBlock - NumKBlock;
    int       NumIter          = 0;
    if (IsLastBatch)
        NumIter = AverageNumKBlock - PaddingKBlock;
    else
        NumIter = AverageNumKBlock;


    //the following will reside in SMSP regfile
    uint64_t Registers_bmp[2];  //4 regs
    uint32_t Registers_nnz[2];  //2 regs
    uint32_t Registers_nz[64];  //64 regs // Enough to hold non-zero values for 2 tiles 
    uint32_t nnz_tile0;
    uint32_t nnz_tile1;

    extern __shared__ __align__(128) half smem[];  // at least be 128 Bytes aligned 

    // Warp and lane identification.
    const unsigned int warpId       = threadIdx.x / WARP_SIZE;
    const int          Tile_Start_M = y * TilingConfig::TILE_M;
    const int          Tile_Start_N = x * TilingConfig::TILE_N;
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Compute a grid of C matrix tiles in each warp.
    int Warp_i         = warpId / TilingConfig::BLOCK_COL_WARPS;
    int Warp_j         = warpId % TilingConfig::BLOCK_COL_WARPS;
    int warp_start_row = WARP_ROW_TENSORS * MMA_M * Warp_i;
    int warp_start_col = TilingConfig::WARP_COL_TENSORS * MMA_N * Warp_j;
    uint32_t __restrict__ a[WARP_ROW_TENSORS * 2][4];//[8][4] = 32 uint32 
    uint32_t __restrict__ b[TilingConfig::WARP_COL_TENSORS * 2][4]; //[8][4] = 32 uint32
    // copying B tile from GlobalMemory to SharedMemory
    const half* BTileGlobalPTR =
        B + Tile_Start_N * K_Global
        + BatchID * AverageNumKBlock * TILE_K;  // Address for matrix B, taking SplitK into consideration
    //
    
    int BaseTileIdx = y * (4 * K_Global) + BatchID * K_Global / Split_K;
    int tid = threadIdx.x;
    int TB_Row = tid / 32;
    int TB_Col = tid % 32;
    int StartTileIdx = BaseTileIdx + TB_Row * K_Global + TB_Col * 2;
    
    
    
    Coruscant_STC_SpMM_CopyFromGlobalToReg<TilingConfig, SparseKernelConfig>(Registers_nz,
                                                                Registers_bmp,
                                                                Registers_nnz,
                                                                NZ, 
                                                                bmp, 
                                                                idx,
                                                                &nnz_tile0, 
                                                                &nnz_tile1,
                                                                StartTileIdx); 
    
    Coruscant_STC_SpMM_InitSharedMemory<TilingConfig>(smem); //rst_smem
    cp_async_group_commit();
    Coruscant_STC_CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>( 
        smem + TilingConfig::TILE_M * TILE_K, BTileGlobalPTR, K_Global); //ld_dense: this is async, defined in MatMulUtilies.cuh
    cp_async_group_commit();
    
    // Initilazing C Matrix to Zeros
    float c[WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS][REG_PER_C_TENSOR_16_16]; // [4*4][8 in TilingConfig] = 64 floats
    for (int i = 0; i < WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS; i++)
        for (int j = 0; j < REG_PER_C_TENSOR_16_16; j++)
            c[i][j] = 0.0f;
    //
    cp_async_wait_group<1>();
    __syncthreads();
    Coruscant_STC_SpMM_DecompressFromRegisterToShared<TilingConfig, SparseKernelConfig>(
                                                                    //SharedPTR,
                                                                    smem,
                                                                    Registers_nz,
                                                                    Registers_bmp,
                                                                    &nnz_tile0, 
                                                                    &nnz_tile1,
                                                                    TB_Row, 
                                                                    TB_Col,
                                                                    &c[0][0]);
                                                                    //tileIdx); //make sure to keep this tid * 2 
    //
    cp_async_wait_group<0>();
    __syncthreads();
    StartTileIdx +=64;

    
#pragma unroll(1) //unroll exactly once.
    for (int tile_id_k = 0; tile_id_k < NumIter-1; tile_id_k++) {
        
        BTileGlobalPTR = B + Tile_Start_N * K_Global + BatchID * AverageNumKBlock * TILE_K + ((tile_id_k + 1) * TILE_K);
        // double buffer
        half* __restrict__ smem_write_PTR = smem;
        half* __restrict__ smem_read_PTR  = smem;
        smem_write_PTR = smem + ((tile_id_k + 1) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N); //place for 256x64 A and 64x16 B
        smem_read_PTR  = smem + ((tile_id_k) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N);
        //
        bool GlobalCopy = (tile_id_k + 1) < NumIter;

        Coruscant_STC_SpMM_InitSharedMemory<TilingConfig>(smem_write_PTR); //rst_smem
        cp_async_group_commit();
        Coruscant_STC_SpMM_CopyFromGlobalToReg<TilingConfig, SparseKernelConfig>(Registers_nz,
                                                                Registers_bmp,
                                                                Registers_nnz,
                                                                NZ, 
                                                                bmp,
                                                                idx,
                                                                &nnz_tile0,
                                                                &nnz_tile1, 
                                                                StartTileIdx); 

        // Copying B Tile
        Coruscant_STC_CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(
            smem_write_PTR + TilingConfig::TILE_M * TILE_K, BTileGlobalPTR, K_Global, GlobalCopy);  //ld_dense
        cp_async_group_commit();


        Coruscant_STC_PipelinedCoreComputations<TilingConfig>(c, a, b, smem_read_PTR, warp_start_row, warp_start_col); //compute
        //

        cp_async_wait_group<1>();
        __syncthreads();  // Sync to ensure the completion of stage 2, but the asyncopy of Tile_B may not finished yet
        Coruscant_STC_SpMM_DecompressFromRegisterToShared<TilingConfig, SparseKernelConfig>(
                                                                    smem_write_PTR,
                                                                    Registers_nz,
                                                                    Registers_bmp,
                                                                    &nnz_tile0,
                                                                    &nnz_tile1,
                                                                    TB_Row, 
                                                                    TB_Col,
                                                                    &c[0][0]);
                                                                    //tileIdx); //make sure to keep this tid * 2
            
        cp_async_wait_group<0>();  // Sync to ensure the completion of Loading B to shared memory
        __syncthreads();
        //StartTileIdx += 8; //for the next 246x64 tile (8 8x8 block apart row-wise)
        StartTileIdx += 64;

        
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Store the C fragments to shared memory.
    
     //add epliogue
    half* __restrict__ smem_read_PTR  = smem;
    smem_read_PTR  = smem + ((NumIter-1) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N);
    Coruscant_STC_PipelinedCoreComputations<TilingConfig>(c, a, b, smem_read_PTR, warp_start_row, warp_start_col); //compute
    __syncthreads();
    
    float(*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C] =
        reinterpret_cast<float(*)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C]>(smem);
    Coruscant_STC_StoreToSharedMemoryFromRegister<TilingConfig>(smem_CFrag, c);
    __syncthreads();
    // Now that shared memory contains all the D tiles, stream them to global memory.
    half* BlockGlobalPTR =
        Reduction_Workspace + BatchID * (M_Global * N_Global) + Tile_Start_M + Tile_Start_N * M_Global;
#pragma unroll
    for (int i = warpId; i < TilingConfig::TILE_N2; i += TilingConfig::BLOCK_WARPS)  // i-th column
#pragma unroll
        for (int j = threadIdx.x % WARP_SIZE; j < TilingConfig::TILE_M; j += WARP_SIZE)  // j-th row
            BlockGlobalPTR[j + i * M_Global] = __float2half_rn((*(smem_CFrag + i))[j]);
}

