#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;
// note: mma requires every thread within warp to participate
// otherwise it is undefined behaviour

#define WARP_SIZE 32

// pass in value LOAD/STORE 128 BITS
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

// by default cp.async instructions go to the current async group buffer
// calling cp.async.commit_group closes the curent group (push it into the queue) and starts a new one
// https://docs.nvidia.com/cuda/parallel-thread-execution/
// 9.7.9.25.3.3. Data Movement and Conversion Instructions: cp.async.wait_group
// cp.async.wait_group instruction will cause executing thread to wait 
// till only N or fewer of the most recent cp.async-groups are pending 
// and all the prior cp.async-groups committed by the executing threads are complete.
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

// ca(cache all, L1 + L2), bytes = 4, 8, 16
#define CP_ASYNC_CA(dst, src, bytes) asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
// cg(cache global, L2): bytes = 16
#define CP_ASYNC_CG(dst, src, bytes) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
// we advise L2 to prefetch the whole 128-byte line

template<
  const int WMMA_SIZE_M=16,
  const int WMMA_SIZE_N=16,
  const int WMMA_SIZE_K=16,
  // m16n16k16 MMA 
  const int WARPS_PER_BLOCK_M=4,
  const int WARPS_PER_BLOCK_N=4,
  // 4x4 WARP tiles
  const int WMMA_PER_WARP_M=4,
  const int WMMA_PER_WARP_N=4,
  // 4x4 WMMA tiles
  const int PAD_A=0,
  const int PAD_B=0,
  const int N_STAGE=2
>
__global__ __launch_bounds__(512)
void hgemm_m16n16k16mma4x4_wp4x4_stages(
  half* A, half* B, half* C,
  int M, int N, int K        
){
  // 512 threads, 16 warps per block
  //int b_x = blockIdx.x;
  //int b_y = blockIdx.y;
  // C tile: 256 x 256
  constexpr int BM = WMMA_SIZE_M * WMMA_PER_WARP_M * WARPS_PER_BLOCK_M;
  constexpr int BN = WMMA_SIZE_N * WMMA_PER_WARP_N * WARPS_PER_BLOCK_N;
  constexpr int BK = WMMA_SIZE_K;

  
  constexpr int GROUP_SIZE_M = 8;
  int b_id = blockIdx.y * gridDim.x + blockIdx.x;
  int num_blocks_m = (M + BM - 1) / BM;
  int num_blocks_n = (N + BN - 1) / BN;
  // Group-based remapping
  int num_pid_in_group = GROUP_SIZE_M * num_blocks_n;
  int group_id = b_id / num_pid_in_group;
  int first_pid_m = group_id * GROUP_SIZE_M;
  int group_size_m = min(num_blocks_m - first_pid_m, GROUP_SIZE_M);
  // Column-major ordering within group for L2 cache optimization
  int b_y = first_pid_m + ((b_id % num_pid_in_group) % group_size_m);
  int b_x = (b_id % num_pid_in_group) / group_size_m;
  
  // declare dynamic shared memory, since we demand more than 48 KB
  extern __shared__ half s_mem[];
  half* s_A = s_mem;
  half* s_B = s_mem + N_STAGE * BM * (BK + PAD_A);
  constexpr int stage_size_A = BM * (BK + PAD_A);
  constexpr int stage_size_B = BK * (BN + PAD_B);

  // determine thread and warp
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  // determine warp tile idx
  int warp_m = warp_id / 4; // 0, 1, 2, 3
  int warp_n = warp_id % 4; // 0, 1, 2, 3

  // calculate load offsets in SRAM
  // 2 thread loads a row of A (256 x 16, 512 threads)
  int load_s_A_m = tid / 2;
  int load_s_A_k = (tid % 2 == 0) ? 0 : 8;
  int load_s_B_k = tid / 32;
  int load_s_B_n = (tid % 32) * 8;

  // for reading from smem (loading into fragments):
  // A: each T takes 8 halfs in a row (16x16/32=8), 4 banks
  // s_A: same as reading, 4 way conflicts.
  // s_B: each T takes 8 halfs in a column (2T/column) 
  // each column takes (1/2 banks) -> 32 way conflicts 
  // Padding s_B is going to help ALOT.

  // calculate load offsets in GRAM
  int load_g_A_m = b_y * BM + load_s_A_m;
  int load_g_B_n = b_x * BN + load_s_B_n;

  // initialize the array of C fragments
  wmma::fragment<
    wmma::accumulator,
    WMMA_SIZE_M, WMMA_SIZE_N, WMMA_SIZE_K,
    half> frag_C[WMMA_PER_WARP_M][WMMA_PER_WARP_N];
  
  #pragma unroll
  for (int i=0; i<WMMA_PER_WARP_M; i++){
    #pragma unroll
    for (int j=0; j<WMMA_PER_WARP_N; j++){
      wmma::fill_fragment(frag_C[i][j], __float2half(0.0f));
    }
  }

  // convert generic smem address into shared address for cp.async
  u_int32_t load_s_A_base_ptr = __cvta_generic_to_shared(s_A);
  u_int32_t load_s_B_base_ptr = __cvta_generic_to_shared(s_B);

  // first, issue async loads for first (n-1) stages
  // there should be more k-tiles than stages
  #pragma unroll
  for (int idx_k=0; idx_k<(N_STAGE-1); idx_k++){
    int load_g_A_k = idx_k * WMMA_SIZE_K + load_s_A_k;
    int load_g_B_k = idx_k * WMMA_SIZE_K + load_s_B_k;
    int load_g_A_off = load_g_A_m * K + load_g_A_k;
    int load_g_B_off = load_g_B_k * N + load_g_B_n;

    u_int32_t load_s_A_ptr = load_s_A_base_ptr + 
                            (idx_k * stage_size_A + 
                             load_s_A_m * (BK + PAD_A) + 
                             load_s_A_k) * sizeof(half);
    u_int32_t load_s_B_ptr = load_s_B_base_ptr + 
                            (idx_k * stage_size_B + 
                             load_s_B_k * (BN + PAD_B) + 
                             load_s_B_n) * sizeof(half);
    // issue cp.async
    CP_ASYNC_CG(load_s_A_ptr, &A[load_g_A_off], 16);
    CP_ASYNC_CG(load_s_B_ptr, &B[load_g_B_off], 16);
    CP_ASYNC_COMMIT_GROUP();
  }

  // wait till stage 0 is ready (when (NSTAGE-1)-1 stages are still pending)
  CP_ASYNC_WAIT_GROUP(N_STAGE-2);
  __syncthreads();

  #pragma unroll
  for (int idx_k=(N_STAGE-1); idx_k<(K/WMMA_SIZE_K); idx_k++){
    // compute (idx_k+1) % stage, load idx_k % stage
    int curr_buf = (idx_k + 1) % N_STAGE;
    int next_buf = idx_k % N_STAGE;
    
    // calculate global load offsets
    int load_g_A_k = idx_k * WMMA_SIZE_K + load_s_A_k;
    int load_g_B_k = idx_k * WMMA_SIZE_K + load_s_B_k;
    int load_g_A_off = load_g_A_m * K + load_g_A_k;
    int load_g_B_off = load_g_B_k * N + load_g_B_n;

    u_int32_t load_s_A_ptr = load_s_A_base_ptr + 
                            (next_buf * stage_size_A + 
                             load_s_A_m * (BK + PAD_A) + 
                             load_s_A_k) * sizeof(half);
    u_int32_t load_s_B_ptr = load_s_B_base_ptr + 
                            (next_buf * stage_size_B + 
                             load_s_B_k * (BN + PAD_B) + 
                             load_s_B_n) * sizeof(half);
    // issue cp.async
    CP_ASYNC_CG(load_s_A_ptr, &A[load_g_A_off], 16);
    CP_ASYNC_CG(load_s_B_ptr, &B[load_g_B_off], 16);
    CP_ASYNC_COMMIT_GROUP();

    // having issued cp.async for next_buf, 
    // proceed with computation for curr_buf

    // now initialize and load into fragments (registers) from SRAM
    wmma::fragment<
      wmma::matrix_a, 
      WMMA_SIZE_M, WMMA_SIZE_N, WMMA_SIZE_K,
      half,
      wmma::row_major> frag_A[WMMA_PER_WARP_M];
    wmma::fragment<
      wmma::matrix_b, 
      WMMA_SIZE_M, WMMA_SIZE_N, WMMA_SIZE_K,
      half,
      wmma::row_major> frag_B[WMMA_PER_WARP_N];
    
    // load in frag_A, frag_B
    #pragma unroll
    for (int i=0; i<WMMA_PER_WARP_M; i++){
      // for each MMA tile
      wmma::load_matrix_sync(frag_A[i],
                             // use generic pointer here
                             s_A +
                             curr_buf * stage_size_A + 
                             (warp_m * WMMA_SIZE_M * WMMA_PER_WARP_M +
                             i * WMMA_SIZE_M) * (BK + PAD_A), 
                             BK + PAD_A);
    }
    #pragma unroll
    for (int j=0; j<WMMA_PER_WARP_N; j++){
      wmma::load_matrix_sync(frag_B[j],
                             // use generic pointer here
                             s_B +
                             curr_buf * stage_size_B + 
                             (warp_n * WMMA_SIZE_N * WMMA_PER_WARP_N +
                             j * WMMA_SIZE_N), 
                             BN + PAD_B);
    }
    // mma.sync
    #pragma unroll
    for (int i=0; i<WMMA_PER_WARP_M; i++){
      #pragma unroll
      for (int j=0; j<WMMA_PER_WARP_N; j++){
        wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
      }
    }

    // finished computation on curr_buf
    // need to wait for last async load on next_buf
    // the load group for next_buf is NSTAGE-2 groups away
    /*
    L: (async) load; W: wait, C: compute
    | 0  | 1  | 2  | 3  | 4  | 5  | 6  | 7  |
    k ->
    |Lk0 |Lk1 |Lk2 |Lk3 |Lk4 |Lk5 |Lk6 |Lk7 |
    |-   |-   |-   |-   |-   |-   |-   |Ck0 |
    |-   |-   |-   |-   |-   |-   |Wk0 |Wk1 |
    k -> (let's say there are 12 ks)
    |Lk8 |Lk9 |Lk10|Lk11|-   |-   |-   |-   |
    |Ck1 |Ck2 |Ck3 |Ck4 |Ck5 |Ck6 |Ck7 |Ck8 |    
    |Wk2 |Wk3 |Wk4 |WALL|-   |-   |-   |-   |
    k ->
    |-   |-   |-   |
    |Ck9 |Ck10|Ck11|  
    |-   |-   |-   |
    after iterating all ks, we need to do extra NSTAGE-1 iterations
    (next_buf is Nstage-1 ahead of curr_buf)
    L(k%Nstage), WC((k+1)%Nstage) / pending 6 groups = 8-2 
    */
    CP_ASYNC_WAIT_GROUP(N_STAGE-2);
    __syncthreads();
  }

  // N_STAGE is constexpr so this should be optimized in compile time
  // this branch should NOT be executed on warp
  if ((N_STAGE-2) > 0){
    // for the last round, just wait for all groups
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
  }

  #pragma unroll
  for(int idx=0; idx<N_STAGE-1; idx++){
    // just continue on the previous loop for extra N_STAGE-1 iterations
    // without async loads/waits, computation only
    int curr_buf = ((K/WMMA_SIZE_K) + idx + 1) % N_STAGE;
    // initialize and load into fragments (registers) from SRAM
    wmma::fragment<
      wmma::matrix_a, 
      WMMA_SIZE_M, WMMA_SIZE_N, WMMA_SIZE_K,
      half,
      wmma::row_major> frag_A[WMMA_PER_WARP_M];
    wmma::fragment<
      wmma::matrix_b, 
      WMMA_SIZE_M, WMMA_SIZE_N, WMMA_SIZE_K,
      half,
      wmma::row_major> frag_B[WMMA_PER_WARP_N];
    
    // load in frag_A, frag_B
    #pragma unroll
    for (int i=0; i<WMMA_PER_WARP_M; i++){
      // for each MMA tile
      wmma::load_matrix_sync(frag_A[i],
                             // use generic pointer here
                             s_A +
                             curr_buf * stage_size_A + 
                             (warp_m * WMMA_SIZE_M * WMMA_PER_WARP_M +
                             i * WMMA_SIZE_M) * (BK + PAD_A), 
                             BK + PAD_A);
    }
    #pragma unroll
    for (int j=0; j<WMMA_PER_WARP_N; j++){
      wmma::load_matrix_sync(frag_B[j],
                             // use generic pointer here
                             s_B +
                             curr_buf * stage_size_B + 
                             (warp_n * WMMA_SIZE_N * WMMA_PER_WARP_N +
                             j * WMMA_SIZE_N), 
                             BN + PAD_B);
    }
    // mma.sync
    #pragma unroll
    for (int i=0; i<WMMA_PER_WARP_M; i++){
      #pragma unroll
      for (int j=0; j<WMMA_PER_WARP_N; j++){
        wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
      }
    }
  }

  // writing back
  #pragma unroll
  for (int i=0; i<WMMA_PER_WARP_M; i++){
    #pragma unroll
    for (int j=0; j<WMMA_PER_WARP_N; j++){
      int store_g_C_m = b_y * BM + warp_m * (WMMA_SIZE_M * WMMA_PER_WARP_M) + i * WMMA_SIZE_M;
      int store_g_C_n = b_x * BN + warp_n * (WMMA_SIZE_N * WMMA_PER_WARP_N) + j * WMMA_SIZE_N;
      wmma::store_matrix_sync(C + store_g_C_m * N + store_g_C_n, frag_C[i][j],
                              N, wmma::mem_row_major);
    }
  }

}

#ifndef NO_TORCH_BINDING


#include <torch/types.h>
#include <torch/extension.h>

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                 \
if(((T).options().dtype() != (th_type))) {                   \
  std::cout << "Tensor Info:" << (T).options() << std::endl; \
  throw std::runtime_error("values must be "#th_type);       \
}

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)           \
if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) { \
  throw std::runtime_error("Tensor size mismatch!");  \
}

void hgemm_m16n16k16mma4x4_wp4x4_stages(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  if (M % 128 != 0 || N % 128 != 0 || K % 16 != 0) {
    throw std::runtime_error("M and N must be divisible by 128. K must be divisible by 16.");
  }
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int WMMA_SIZE_M = 16;
  constexpr int WMMA_SIZE_N = 16;
  constexpr int WMMA_SIZE_K = 16;
  constexpr int WARPS_PER_BLOCK_M = 4;
  constexpr int WARPS_PER_BLOCK_N = 4; 
  constexpr int WMMA_PER_WARP_M = 4;
  constexpr int WMMA_PER_WARP_N = 4;
  constexpr int BN = WMMA_SIZE_N * WMMA_PER_WARP_N * WARPS_PER_BLOCK_N;
  constexpr int BM = WMMA_SIZE_M * WMMA_PER_WARP_M * WARPS_PER_BLOCK_M;
  constexpr int BK = WMMA_SIZE_K;
  constexpr int PAD_A = 8;
  constexpr int PAD_B = 8;
  constexpr int N_STAGES = 2;
  constexpr int SMEM_SIZE = (((N_STAGES) * BM * (BK + PAD_A)) + ((N_STAGES) * BK * (BN + PAD_B))) * sizeof(half);
  cudaFuncSetAttribute(
    hgemm_m16n16k16mma4x4_wp4x4_stages<
      WMMA_SIZE_M, WMMA_SIZE_N, WMMA_SIZE_K,
      WARPS_PER_BLOCK_M, WARPS_PER_BLOCK_N,
      WMMA_PER_WARP_M, WMMA_PER_WARP_N,
      PAD_A, PAD_B, N_STAGES
    >,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    SMEM_SIZE);
  constexpr int NUM_THREADS= (WARPS_PER_BLOCK_M * WARPS_PER_BLOCK_N * WARP_SIZE);
  dim3 block(NUM_THREADS);
  dim3 grid((N / (WMMA_SIZE_N * WMMA_PER_WARP_N * WARPS_PER_BLOCK_N)), 
            (M / (WMMA_SIZE_M * WMMA_PER_WARP_M * WARPS_PER_BLOCK_M)));
 
  hgemm_m16n16k16mma4x4_wp4x4_stages<
    WMMA_SIZE_M, WMMA_SIZE_N, WMMA_SIZE_K,
    WARPS_PER_BLOCK_M, WARPS_PER_BLOCK_N,
    WMMA_PER_WARP_M, WMMA_PER_WARP_N,
    PAD_A, PAD_B, N_STAGES
  ><<<grid, block, SMEM_SIZE>>>
  (
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}

#endif // NO_TORCH_BINDING