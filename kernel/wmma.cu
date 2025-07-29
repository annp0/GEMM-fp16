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
// wait group will wait for the first n commited groups. If n=0, it will wait for all groups.
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

// ca(cache all, L1 + L2), bytes = 4, 8, 16
#define CP_ASYNC_CA(dst, src, bytes) asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
// cg(cache global, L2): bytes = 16
#define CP_ASYNC_CG(dst, src, bytes) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
// we advise L2 to prefetch the whole 128-byte line

template<const int WMMA_SIZE_M=16,
         const int WMMA_SIZE_N=16,
         const int WMMA_SIZE_K=16,
         // m16n16k16 MMA 
         const int WARPS_PER_BLOCK_M=4,
         const int WARPS_PER_BLOCK_N=2,
         // 4x2 WARP tiles
         const int WMMA_PER_WARP_M=2,
         const int WMMA_PER_WARP_N=4
         // 2x4 WMMA tiles
        >
__global__ void hgemm_m16n16k16mma4x2_wp2x4(
  half* A, half* B, half* C,
  int M, int N, int K
){
  int b_x = blockIdx.x;
  int b_y = blockIdx.y;
  constexpr int BM = WMMA_SIZE_M * WMMA_PER_WARP_M * WARPS_PER_BLOCK_M;
  constexpr int BN = WMMA_SIZE_N * WMMA_PER_WARP_N * WARPS_PER_BLOCK_N;
  constexpr int BK = WMMA_SIZE_K;
  // we take slices of 16 each time
  __shared__ half s_A[BM][BK], s_B[BK][BN];

  // determine thread and warp
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  // determine warp tile idx
  // 256 threads / 8 warps
  int warp_m = warp_id / 2; // 0, 1, 2, 3
  int warp_n = warp_id % 2; // 0, 1

  // calculate load offsets in SRAM
  // 2 thread loads a row of A
  int load_s_A_m = tid / 2;
  int load_s_A_k = (tid % 2 == 0) ? 0 : 8;
  // 16 thread loads a row of B
  int load_s_B_k = tid / 16;
  int load_s_B_n = (tid % 16) * 8;

  // calculate load offsets in GRAM
  int load_g_A_m = b_y * BM + load_s_A_m;
  int load_g_B_n = b_x * BN + load_s_B_n;

  // initialize an array of C fragments
  wmma::fragment<wmma::accumulator,
                 WMMA_SIZE_M, WMMA_SIZE_N, WMMA_SIZE_K,
                 half> frag_C[WMMA_PER_WARP_M][WMMA_PER_WARP_N];
  #pragma unroll
  for (int i=0; i<WMMA_PER_WARP_M; i++){
    #pragma unroll
    for (int j=0; j<WMMA_PER_WARP_N; j++){
      wmma::fill_fragment(frag_C[i][j], 0.0);
    }
  }

  // now head into computation
  // sliding k right + down with step size 16
  #pragma unroll
  for (int idx_k=0; idx_k<(K/WMMA_SIZE_K); idx_k++){
    // first loading in A, B fragments from GRAM
    // calculate offsets on k
    int load_g_A_k = idx_k * WMMA_SIZE_K + load_s_A_k;
    int load_g_A_off = load_g_A_m * K + load_g_A_k;
    int load_g_B_k = idx_k * WMMA_SIZE_K + load_s_B_k;
    int load_g_B_off = load_g_B_k * N + load_g_B_n;
    LDST128BITS(s_A[load_s_A_m][load_s_A_k]) = LDST128BITS(A[load_g_A_off]);
    LDST128BITS(s_B[load_s_B_k][load_s_B_n]) = LDST128BITS(B[load_g_B_off]);
    __syncthreads();

    // now initialize and load into fragments (registers) from SRAM
    wmma::fragment<wmma::matrix_a, 
                   WMMA_SIZE_M, WMMA_SIZE_N, WMMA_SIZE_K,
                   half,
                   wmma::row_major> frag_A[WMMA_PER_WARP_M];
    wmma::fragment<wmma::matrix_b, 
                   WMMA_SIZE_M, WMMA_SIZE_N, WMMA_SIZE_K,
                   half,
                   wmma::row_major> frag_B[WMMA_PER_WARP_N];
    #pragma unroll
    for (int i=0; i<WMMA_PER_WARP_M; i++){
      // for each MMA tile
      wmma::load_matrix_sync(frag_A[i], &s_A[warp_m * (WMMA_PER_WARP_M * WMMA_SIZE_M) + i * WMMA_SIZE_M][0], BK);
    }
    #pragma unroll
    for (int j=0; j<WMMA_PER_WARP_N; j++){
      wmma::load_matrix_sync(frag_B[j], &s_B[0][warp_n * (WMMA_PER_WARP_N * WMMA_SIZE_N) + j * WMMA_SIZE_N], BN);
    }

    // mma.sync
    #pragma unroll
    for (int i=0; i<WMMA_PER_WARP_M; i++){
      #pragma unroll
      for (int j=0; j<WMMA_PER_WARP_N; j++){
        wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
      }
    }
    // before next round of loading into SRAM, sync threads blockwise
    __syncthreads();
  }
  // writing back
  #pragma unroll
  for (int i=0; i<WMMA_PER_WARP_M; i++){
    #pragma unroll
    for (int j=0; j<WMMA_PER_WARP_N; j++){
      int store_g_C_m = b_y * BM + warp_m * (WMMA_SIZE_M * WMMA_PER_WARP_M) + i * WMMA_SIZE_M;
      int store_g_C_n = b_x * BN + warp_n * (WMMA_SIZE_N * WMMA_PER_WARP_N) + j * WMMA_SIZE_N;
      wmma::store_matrix_sync(C + store_g_C_m * N + store_g_C_n, frag_C[i][j],
                              N, wmma::row_major);
    }
  }
}
        