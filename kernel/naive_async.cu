#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// pass in value
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

// the dimensions must be aligned
template<const int BM=128, 
         const int BN=128, 
         const int BK=32,
         const int TM=8,
         const int TN=8,
         const int PAD=0>
__global__ void hgemm_bk32_th8x8_async(
  half* A, half* B, half* C,
  int M, int N, int K
){
  int b_x = blockIdx.x;
  int b_y = blockIdx.y;
  int t_x = threadIdx.x;
  int t_y = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + t_x;
  // transposing s_A allows for vectorized loads into registers
  // sizeof(s_A) = 2 * 32 * 128 * 2 B = 16 KB
  // within each block we have 256 threads
  // so each thread should load 16 elements (for each buffer)
  __shared__ half s_A[2][BK][BM + PAD], s_B[2][BK][BN + PAD];

  // for vectorized loads DRAM -> SRAM
  half reg_load_A[16];
  // for vectorized loads SRAM -> REG
  half reg_comp_A[8];
  half reg_comp_B[8];
  half reg_out_C[8][8] = {__float2half(0.0f)};

  // 2 threads to load 32 elements (16x2 halfs)
  // from one row of A into SRAM
  int load_s_A_m = tid / 2; 
  int load_s_A_k = (tid % 2 == 0) ? 0 : 16;
  // 8 threads to load 128 elements (16×8 halfs) 
  // from one row of B into SRAM
  int load_s_B_k = tid / 8;
  int load_s_B_n = (tid % 8) * 16;
  // access to DRAM is coalesced - each thread is accessing coalesced memory

  // DRAM load tile indices
  int load_g_A_m = b_y * BM + load_s_A_m;
  int load_g_B_n = b_x * BN + load_s_B_n;

  // loading the first buffer
  int load_g_A_k = load_s_A_k; // col of A
  int load_g_B_k = load_s_B_k; // row of B
  int load_g_A_off = load_g_A_m * K + load_g_A_k;
  int load_g_B_off = load_g_B_k * N + load_g_B_n;
  // convert SRAM address to the one that is accept by cp.async
  // this is just an address, offsets are in bytes
  u_int32_t load_s_B_ptr = __cvta_generic_to_shared(&s_B[0][load_s_B_k][load_s_B_n]);

  // async load first s_B[0]
  CP_ASYNC_CA(load_s_B_ptr, &B[load_g_B_off], 16);
  CP_ASYNC_CA(load_s_B_ptr + 16, &B[load_g_B_off + 8], 16);
  CP_ASYNC_COMMIT_GROUP();

  // sync load first s_A[0] from global
  LDST128BITS(reg_load_A[0]) = LDST128BITS(A[load_g_A_off]);
  LDST128BITS(reg_load_A[8]) = LDST128BITS(A[load_g_A_off + 8]);

  // transpose store into shared
  #pragma unroll
  for (int i=0; i<16; i++){
    s_A[0][load_s_A_k + i][load_s_A_m] = reg_load_A[i];
  }

  CP_ASYNC_WAIT_GROUP(0);
  __syncthreads();
  // first buffer is now ready
  // idx_k is the id of the k-block-tile
  for(int idx_k = 1; idx_k < (K + BK - 1) / BK; idx_k ++){
    // double buffering flag
    // idx_k = 1 -> flag = 0, 2 -> 1, 3 -> 0, ...
    int current_buf = (idx_k - 1) & 1; 
    int next_buf = idx_k & 1;

    // offsets
    // ks move right / down by idx_k * BK
    int load_g_A_k = idx_k * BK + load_s_A_k;
    int load_g_B_k = idx_k * BK + load_s_B_k;
    // recalculate offset
    int load_g_A_off = load_g_A_m * K + load_g_A_k;
    int load_g_B_off = load_g_B_k * N + load_g_B_n;

    // issue async loads for the next buffer
    u_int32_t load_s_B_ptr = __cvta_generic_to_shared(&s_B[next_buf][load_s_B_k][load_s_B_n]);
    CP_ASYNC_CA(load_s_B_ptr, &B[load_g_B_off], 16);
    CP_ASYNC_CA(load_s_B_ptr + 16, &B[load_g_B_off + 8], 16);
    CP_ASYNC_COMMIT_GROUP();

    // can go ahead with computation now
    // each thread calculates TM x TN (8x8) output of C
    #pragma unroll
    for (int t_k=0; t_k<BK; t_k++){
      LDST128BITS(reg_comp_A[0]) = LDST128BITS(s_A[current_buf][t_k][t_y * TM]);
      LDST128BITS(reg_comp_B[0]) = LDST128BITS(s_B[current_buf][t_k][t_x * TN]);
      #pragma unroll
      for (int t_m=0; t_m<TM; t_m++){
        #pragma unroll
        for (int t_n=0; t_n<TN; t_n++){
          reg_out_C[t_m][t_n] = __hfma(reg_comp_A[t_m], reg_comp_B[t_n], reg_out_C[t_m][t_n]);
        }
      }
    }

    // proceed to load next s_A since it is needed for the next stage anyway
    // the k offsets have been updated
    LDST128BITS(reg_load_A[0]) = LDST128BITS(A[load_g_A_off]);
    LDST128BITS(reg_load_A[8]) = LDST128BITS(A[load_g_A_off + 8]);
    
    #pragma unroll
    for (int i=0; i<16; i++){
      s_A[next_buf][load_s_A_k + i][load_s_A_m] = reg_load_A[i];
    }

    // wait for previously committed group for async loading B
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
    // ready for next round of async load and computation
  }

  int last_buf = ((K-1) / BK) & 1;

  // after the loop, we still have the last round of compute
  #pragma unroll
  for (int t_k=0; t_k<BK; t_k++){
    LDST128BITS(reg_comp_A[0]) = LDST128BITS(s_A[last_buf][t_k][t_y * TM]);
    LDST128BITS(reg_comp_B[0]) = LDST128BITS(s_B[last_buf][t_k][t_x * TN]);
    #pragma unroll
    for (int t_m=0; t_m<TM; t_m++){
      #pragma unroll
      for (int t_n=0; t_n<TN; t_n++){
        reg_out_C[t_m][t_n] = __hfma(reg_comp_A[t_m], reg_comp_B[t_n], reg_out_C[t_m][t_n]);
      }
    }
  }

  // store the result back
  for (int i=0; i<TM; i++){
    int store_g_C_m = b_y * BM + t_y * TM + i;
    int store_g_C_n = b_x * BN + t_x * TN;
    int store_g_C_off = store_g_C_m * N + store_g_C_n;
    LDST128BITS(C[store_g_C_off]) = LDST128BITS(reg_out_C[i][0]);
  }

}

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

void hgemm_bk32_th8x8_async(
  torch::Tensor a, 
  torch::Tensor b, 
  torch::Tensor c
){
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 32; 
  constexpr int TM = 8;
  constexpr int TN = 8;

  if (M % BM != 0 || N % BN != 0 || K % BK != 0) {
    throw std::runtime_error("M, N, K must be divisible by 128, 128, 32 respectively.");
  }

  dim3 block(BN/TN, BM/TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  hgemm_bk32_th8x8_async<
    BM, BN, BK, TM, TN, 8><<<grid, block>>>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}