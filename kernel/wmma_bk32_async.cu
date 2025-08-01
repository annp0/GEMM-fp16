#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define WARP_SIZE 32

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

__global__ __launch_bounds__(256)
void hgemm_m16n16k16_bk32_async(
    half* A, half* B, half* C,
    const int M, const int N, const int K) {

    // block and warp tile sizes
    const int BM = 128;
    const int BN = 256;
    const int BK = 32;

    // block and thread indices
    int b_x = blockIdx.x;
    int b_y = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid >> 5; // tid / 32

    // shared memory padding
    const int PAD_A = 8;
    const int PAD_B = 8;

    // declare dynamic shared memory
    extern __shared__ half s_mem[];
    half *s_A = s_mem;
    half *s_B = s_mem + 2 * BM * (BK + PAD_A);
    int stage_size_A = BM * (BK + PAD_A);
    int stage_size_B = BK * (BN + PAD_B);

    // WMMA fragments for double buffering
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_C[4][4];

    // initialize accumulator fragments to zero
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_C[i][j], __float2half(0.0f));
        }
    }

    // calculate load offsets in shared memory
    // 4 threads load a row of A, each thread takes 2 rows (128 x 32, 256 threads)
    int load_s_A_m = (tid >> 2) << 1; // (tid / 4) * 2
    int load_s_A_k = (tid & 3) << 3;  // (tid % 4) * 8
    
    // 32 threads load a row of B, each thread takes 4 rows (32 x 256, 32 threads per row)
    int load_s_B_k = (tid >> 5) << 2; // (tid / 32) * 4
    int load_s_B_n = (tid & 31) << 3; // (tid % 32) * 8

    // convert generic addresses to shared memory addresses for async copy
    int s_A_base_addr = __cvta_generic_to_shared(s_A);
    int s_B_base_addr = __cvta_generic_to_shared(s_B);

    // calculate shared memory addresses for async loads
    int load_s_A_addr_0 = s_A_base_addr + OFFSET(load_s_A_m, load_s_A_k, BK + PAD_A) * sizeof(half);
    int load_s_A_addr_1 = load_s_A_addr_0 + (BK + PAD_A) * sizeof(half);
    int load_s_B_addr_0 = s_B_base_addr + OFFSET(load_s_B_k, load_s_B_n, BN + PAD_B) * sizeof(half);
    int load_s_B_addr_1 = load_s_B_addr_0 +     (BN + PAD_B) * sizeof(half);
    int load_s_B_addr_2 = load_s_B_addr_0 + 2 * (BN + PAD_B) * sizeof(half);
    int load_s_B_addr_3 = load_s_B_addr_0 + 3 * (BN + PAD_B) * sizeof(half);

    // calculate global memory load offsets
    int load_g_A_m = b_y * BM + load_s_A_m;
    int load_g_B_n = b_x * BN + load_s_B_n;

    int load_g_A_addr = OFFSET(load_g_A_m, load_s_A_k, K);
    int load_g_B_addr = OFFSET(load_s_B_k, load_g_B_n, N);

    // determine which fragment this warp computes
    int warp_m = warp_id & 1;  // 0 or 1
    int warp_n = warp_id >> 1; // 0, 1, 2, or 3

    // load first K-tile into shared memory using async copy
    {
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_s_A_addr_0), "l"(&A[load_g_A_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_s_A_addr_1), "l"(&A[load_g_A_addr +     K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_s_B_addr_0), "l"(&B[load_g_B_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_s_B_addr_1), "l"(&B[load_g_B_addr +     N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_s_B_addr_2), "l"(&B[load_g_B_addr + 2 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_s_B_addr_3), "l"(&B[load_g_B_addr + 3 * N]));

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    // main K-tile loop with double buffering
    for (int k_tile = 1; k_tile < K / BK; k_tile++) {

        // determine which buffer to use (double buffering)
        int curr_buf = (k_tile & 1) ^ 1;     // current buffer for computation
        int next_buf = ((k_tile - 1) & 1) ^ 1; // next buffer for loading

        // update global memory addresses for next K-tile
        load_g_A_addr += BK;
        load_g_B_addr += BK * N;

        // issue async loads for next K-tile
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_s_A_addr_0 + next_buf * stage_size_A * (int)sizeof(half)), "l"(&A[load_g_A_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_s_A_addr_1 + next_buf * stage_size_A * (int)sizeof(half)), "l"(&A[load_g_A_addr +     K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_s_B_addr_0 + next_buf * stage_size_B * (int)sizeof(half)), "l"(&B[load_g_B_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_s_B_addr_1 + next_buf * stage_size_B * (int)sizeof(half)), "l"(&B[load_g_B_addr +     N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_s_B_addr_2 + next_buf * stage_size_B * (int)sizeof(half)), "l"(&B[load_g_B_addr + 2 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_s_B_addr_3 + next_buf * stage_size_B * (int)sizeof(half)), "l"(&B[load_g_B_addr + 3 * N]));

        // load fragments from current buffer for first half of K-tile (k=0:15)
        wmma::load_matrix_sync(frag_A[0][0], &s_A[curr_buf * stage_size_A + (warp_m * 64     ) * (BK + PAD_A) +  0], BK + PAD_A);
        wmma::load_matrix_sync(frag_A[0][1], &s_A[curr_buf * stage_size_A + (warp_m * 64 + 16) * (BK + PAD_A) +  0], BK + PAD_A);
        wmma::load_matrix_sync(frag_A[0][2], &s_A[curr_buf * stage_size_A + (warp_m * 64 + 32) * (BK + PAD_A) +  0], BK + PAD_A);
        wmma::load_matrix_sync(frag_A[0][3], &s_A[curr_buf * stage_size_A + (warp_m * 64 + 48) * (BK + PAD_A) +  0], BK + PAD_A);
        
        // load fragments for second half of K-tile (k=16:31)
        wmma::load_matrix_sync(frag_A[1][0], &s_A[curr_buf * stage_size_A + (warp_m * 64     ) * (BK + PAD_A) + 16], BK + PAD_A);
        wmma::load_matrix_sync(frag_A[1][1], &s_A[curr_buf * stage_size_A + (warp_m * 64 + 16) * (BK + PAD_A) + 16], BK + PAD_A);
        wmma::load_matrix_sync(frag_A[1][2], &s_A[curr_buf * stage_size_A + (warp_m * 64 + 32) * (BK + PAD_A) + 16], BK + PAD_A);
        wmma::load_matrix_sync(frag_A[1][3], &s_A[curr_buf * stage_size_A + (warp_m * 64 + 48) * (BK + PAD_A) + 16], BK + PAD_A);

        // load B fragments for first half of K-tile (k=0:15)
        wmma::load_matrix_sync(frag_B[0][0], &s_B[curr_buf * stage_size_B +                    warp_n * 64     ], BN + PAD_B);
        wmma::load_matrix_sync(frag_B[0][1], &s_B[curr_buf * stage_size_B +                    warp_n * 64 + 16], BN + PAD_B);
        wmma::load_matrix_sync(frag_B[0][2], &s_B[curr_buf * stage_size_B +                    warp_n * 64 + 32], BN + PAD_B);
        wmma::load_matrix_sync(frag_B[0][3], &s_B[curr_buf * stage_size_B +                    warp_n * 64 + 48], BN + PAD_B);
        
        // load B fragments for second half of K-tile (k=16:31)
        wmma::load_matrix_sync(frag_B[1][0], &s_B[curr_buf * stage_size_B + 16 * (BN + PAD_B) + warp_n * 64     ], BN + PAD_B);
        wmma::load_matrix_sync(frag_B[1][1], &s_B[curr_buf * stage_size_B + 16 * (BN + PAD_B) + warp_n * 64 + 16], BN + PAD_B);
        wmma::load_matrix_sync(frag_B[1][2], &s_B[curr_buf * stage_size_B + 16 * (BN + PAD_B) + warp_n * 64 + 32], BN + PAD_B);
        wmma::load_matrix_sync(frag_B[1][3], &s_B[curr_buf * stage_size_B + 16 * (BN + PAD_B) + warp_n * 64 + 48], BN + PAD_B);

        // perform matrix multiplication for current K-tile
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                // first half of K-tile (k=0:15)
                wmma::mma_sync(frag_C[i][j], frag_A[0][i], frag_B[0][j], frag_C[i][j]);
                // second half of K-tile (k=16:31)
                wmma::mma_sync(frag_C[i][j], frag_A[1][i], frag_B[1][j], frag_C[i][j]);
            }
        }

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    // process the last K-tile (no more async loads needed)
    int curr_buf = ((K / BK) & 1) ^ 1;

    // load fragments for last K-tile computation
    wmma::load_matrix_sync(frag_A[0][0], &s_A[curr_buf * stage_size_A + (warp_m * 64     ) * (BK + PAD_A) +  0], BK + PAD_A);
    wmma::load_matrix_sync(frag_A[0][1], &s_A[curr_buf * stage_size_A + (warp_m * 64 + 16) * (BK + PAD_A) +  0], BK + PAD_A);
    wmma::load_matrix_sync(frag_A[0][2], &s_A[curr_buf * stage_size_A + (warp_m * 64 + 32) * (BK + PAD_A) +  0], BK + PAD_A);
    wmma::load_matrix_sync(frag_A[0][3], &s_A[curr_buf * stage_size_A + (warp_m * 64 + 48) * (BK + PAD_A) +  0], BK + PAD_A);
    wmma::load_matrix_sync(frag_A[1][0], &s_A[curr_buf * stage_size_A + (warp_m * 64     ) * (BK + PAD_A) + 16], BK + PAD_A);
    wmma::load_matrix_sync(frag_A[1][1], &s_A[curr_buf * stage_size_A + (warp_m * 64 + 16) * (BK + PAD_A) + 16], BK + PAD_A);
    wmma::load_matrix_sync(frag_A[1][2], &s_A[curr_buf * stage_size_A + (warp_m * 64 + 32) * (BK + PAD_A) + 16], BK + PAD_A);
    wmma::load_matrix_sync(frag_A[1][3], &s_A[curr_buf * stage_size_A + (warp_m * 64 + 48) * (BK + PAD_A) + 16], BK + PAD_A);

    wmma::load_matrix_sync(frag_B[0][0], &s_B[curr_buf * stage_size_B +                    warp_n * 64     ], BN + PAD_B);
    wmma::load_matrix_sync(frag_B[0][1], &s_B[curr_buf * stage_size_B +                    warp_n * 64 + 16], BN + PAD_B);
    wmma::load_matrix_sync(frag_B[0][2], &s_B[curr_buf * stage_size_B +                    warp_n * 64 + 32], BN + PAD_B);
    wmma::load_matrix_sync(frag_B[0][3], &s_B[curr_buf * stage_size_B +                    warp_n * 64 + 48], BN + PAD_B);
    wmma::load_matrix_sync(frag_B[1][0], &s_B[curr_buf * stage_size_B + 16 * (BN + PAD_B) + warp_n * 64     ], BN + PAD_B);
    wmma::load_matrix_sync(frag_B[1][1], &s_B[curr_buf * stage_size_B + 16 * (BN + PAD_B) + warp_n * 64 + 16], BN + PAD_B);
    wmma::load_matrix_sync(frag_B[1][2], &s_B[curr_buf * stage_size_B + 16 * (BN + PAD_B) + warp_n * 64 + 32], BN + PAD_B);
    wmma::load_matrix_sync(frag_B[1][3], &s_B[curr_buf * stage_size_B + 16 * (BN + PAD_B) + warp_n * 64 + 48], BN + PAD_B);

    // final matrix multiplication
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::mma_sync(frag_C[i][j], frag_A[0][i], frag_B[0][j], frag_C[i][j]);
            wmma::mma_sync(frag_C[i][j], frag_A[1][i], frag_B[1][j], frag_C[i][j]);
        }
    }

    // store results back to global memory
    int store_g_C_m = b_y * BM + warp_m * 64;
    int store_g_C_n = b_x * BN + warp_n * 64;
    int store_g_C_addr = OFFSET(store_g_C_m, store_g_C_n, N);
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(&C[store_g_C_addr + i * 16 * N + j * 16], frag_C[i][j], N, wmma::mem_row_major);
        }
    }
}