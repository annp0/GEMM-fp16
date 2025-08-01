#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <signal.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
using namespace nvcuda;

#define NO_TORCH_BINDING
#include "wmma_stages.cu"
#include "wmma_bk32_async.cu"
#include "../cublas/hgemm.cu"

// Signal handler for cleanup
void signal_handler(int sig) {
    printf("\nInterrupted, cleaning up cublas handle...\n");
    destroy_cublas_handle();
    exit(sig);
}

void test_correctness(int M, int N, int K) {
    printf("\n=== Correctness Test for M=N=K=%d ===\n", M);
    
    size_t size_a = M * K * sizeof(half);
    size_t size_b = K * N * sizeof(half);
    size_t size_c = M * N * sizeof(half);

    // Allocate host memory
    half *h_a, *h_b, *h_c_stages, *h_c_async, *h_c_cublas;
    h_a = (half *)malloc(size_a);
    h_b = (half *)malloc(size_b);
    h_c_stages = (half *)malloc(size_c);
    h_c_async = (half *)malloc(size_c);
    h_c_cublas = (half *)malloc(size_c);

    // Initialize random data with fixed seed for reproducible results
    srand(42);
    for (int i = 0; i < M * K; i++)
        h_a[i] = (half)(rand() / float(RAND_MAX));
    for (int i = 0; i < K * N; i++)
        h_b[i] = (half)(rand() / float(RAND_MAX));

    // Allocate device memory
    half *d_a, *d_b, *d_c;
    
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    // Copy data to device
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    // === Test WMMA_STAGES ===
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
    constexpr int N_STAGES = 3;
    constexpr int SMEM_SIZE_STAGES = (((N_STAGES) * BM * (BK + PAD_A)) + ((N_STAGES) * BK * (BN + PAD_B))) * sizeof(half);

    dim3 blockDim_stages(WARPS_PER_BLOCK_M * WARPS_PER_BLOCK_N * WARP_SIZE);
    dim3 gridDim_stages((N + BN - 1) / BN, (M + BM - 1) / BM);

    cudaMemset(d_c, 0, size_c);

    cudaFuncSetAttribute(
            hgemm_m16n16k16mma4x4_wp4x4_stages<
        WMMA_SIZE_M, WMMA_SIZE_N, WMMA_SIZE_K,
        WARPS_PER_BLOCK_M, WARPS_PER_BLOCK_N,
        WMMA_PER_WARP_M, WMMA_PER_WARP_N,
        PAD_A, PAD_B, N_STAGES
    >,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            SMEM_SIZE_STAGES);
    
    hgemm_m16n16k16mma4x4_wp4x4_stages<
        WMMA_SIZE_M, WMMA_SIZE_N, WMMA_SIZE_K,
        WARPS_PER_BLOCK_M, WARPS_PER_BLOCK_N,
        WMMA_PER_WARP_M, WMMA_PER_WARP_N,
        PAD_A, PAD_B, N_STAGES
    ><<<gridDim_stages, blockDim_stages, SMEM_SIZE_STAGES>>>(d_a, d_b, d_c, M, N, K);
    
    cudaDeviceSynchronize();
    cudaMemcpy(h_c_stages, d_c, size_c, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // === Test WMMA_BK32_ASYNC ===
    constexpr int BM_ASYNC = 128;
    constexpr int BN_ASYNC = 256;
    constexpr int BK_ASYNC = 32;
    constexpr int SMEM_SIZE_ASYNC = (2 * BM_ASYNC * (BK_ASYNC + 8) + 
                                    2 * BK_ASYNC * (BN_ASYNC + 8)) * sizeof(half);

    dim3 blockDim_async(256);
    dim3 gridDim_async((N + BN_ASYNC - 1) / BN_ASYNC, (M + BM_ASYNC - 1) / BM_ASYNC);

    cudaFuncSetAttribute(
            hgemm_m16n16k16_bk32_async,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            SMEM_SIZE_ASYNC);

    cudaMemset(d_c, 0, size_c);
    hgemm_m16n16k16_bk32_async<<<gridDim_async, blockDim_async, SMEM_SIZE_ASYNC>>>(d_a, d_b, d_c, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c_async, d_c, size_c, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // === Test cuBLAS ===
    cudaMemset(d_c, 0, size_c);
    cublas_tensor_op_nn(d_a, d_b, d_c, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c_cublas, d_c, size_c, cudaMemcpyDeviceToHost);

    // === Check maximum absolute error ===
    float max_abs_error_stages = 0.0f;
    float max_abs_error_async = 0.0f;

    for (int i = 0; i < M * N; i++) {
        float cublas_val = __half2float(h_c_cublas[i]);
        float stages_val = __half2float(h_c_stages[i]);
        float async_val = __half2float(h_c_async[i]);

        // Check stages kernel
        float abs_error_stages = fabsf(stages_val - cublas_val);
        max_abs_error_stages = fmaxf(max_abs_error_stages, abs_error_stages);

        // Check async kernel
        float abs_error_async = fabsf(async_val - cublas_val);
        max_abs_error_async = fmaxf(max_abs_error_async, abs_error_async);
    }

    printf("WMMA_STAGES: max_abs_error = %.6f\n", max_abs_error_stages);
    printf("WMMA_ASYNC:  max_abs_error = %.6f\n", max_abs_error_async);

    // Clean up
    free(h_a); free(h_b); free(h_c_stages); free(h_c_async); free(h_c_cublas);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}


int main() {
    // Register signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Run correctness test first
    test_correctness(4096, 4096, 4096);
    
    // Seed random number generator
    srand(time(0));
    
    const int warmup = 5;
    const int repeat = 10;

    // Buffer for CSV output
    char csv_buffer[10000] = "";
    strcat(csv_buffer, "Matrix_Size,WMMA_Stages_ms,WMMA_Stages_TFLOPS,WMMA_Async_ms,WMMA_Async_TFLOPS,cuBLAS_ms,cuBLAS_TFLOPS\n");

    printf("DIM:    M=N=K | stages=3 bk=16            | async bk=32               | cuBLAS                    |\n");
    printf("--------------------------------------------------------------------------------------------------\n");

    for (int MNK = 256; MNK <= 16384; MNK += 256) {
        int M = MNK, N = MNK, K = MNK;

        size_t size_a = M * K * sizeof(half);
        size_t size_b = K * N * sizeof(half);
        size_t size_c = M * N * sizeof(half);

        init_cublas_handle();

        // Allocate host memory
        half *h_a, *h_b;
        h_a = (half *)malloc(size_a);
        h_b = (half *)malloc(size_b);

        // Initialize random data
        for (int i = 0; i < M * K; i++)
            h_a[i] = (half)(rand() / float(RAND_MAX));
        for (int i = 0; i < K * N; i++)
            h_b[i] = (half)(rand() / float(RAND_MAX));

        // Allocate device memory
        half *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, size_a);
        cudaMalloc(&d_b, size_b);
        cudaMalloc(&d_c, size_c);

        // Copy data to device
        cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
        cudaMemset(d_c, 0, size_c);  // Initialize output to zero
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // ===== WMMA_BK32_ASYNC KERNEL =====
        constexpr int BM_ASYNC = 128;
        constexpr int BN_ASYNC = 256;
        constexpr int BK_ASYNC = 32;
        constexpr int PAD_A_ASYNC = 8;
        constexpr int PAD_B_ASYNC = 8;
        constexpr int SMEM_SIZE_ASYNC = (2 * BM_ASYNC * (BK_ASYNC + PAD_A_ASYNC) + 
                                        2 * BK_ASYNC * (BN_ASYNC + PAD_B_ASYNC)) * sizeof(half);
        constexpr int NUM_THREADS_ASYNC = 256;

        dim3 blockDim_async(NUM_THREADS_ASYNC);
        dim3 gridDim_async((N + BN_ASYNC - 1) / BN_ASYNC, (M + BM_ASYNC - 1) / BM_ASYNC);

        cudaFuncSetAttribute(
            hgemm_m16n16k16_bk32_async,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            SMEM_SIZE_ASYNC);

        // Warmup for wmma_bk32_async
        for (int i = 0; i < warmup; ++i) {
            hgemm_m16n16k16_bk32_async<<<gridDim_async, blockDim_async, SMEM_SIZE_ASYNC>>>(d_a, d_b, d_c, M, N, K);
        }
        cudaDeviceSynchronize();

        // Benchmark wmma_bk32_async
        cudaEventRecord(start);
        for (int i = 0; i < repeat; ++i) {
            hgemm_m16n16k16_bk32_async<<<gridDim_async, blockDim_async, SMEM_SIZE_ASYNC>>>(d_a, d_b, d_c, M, N, K);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float msec_async = 0;
        cudaEventElapsedTime(&msec_async, start, stop);
        float avg_msec_async = msec_async / repeat;
        double tflops_async = (double)M * N * K * 2 / 1e12 / (avg_msec_async / 1000.0);

        cudaMemset(d_c, 0, size_c);  // Initialize output to zero

        // ===== CUBLAS KERNEL =====
        // Warmup for cuBLAS
        for (int i = 0; i < warmup; ++i) {
            cublas_tensor_op_nn(d_a, d_b, d_c, M, N, K);
        }
        cudaDeviceSynchronize();

        // Benchmark cuBLAS
        cudaEventRecord(start);
        for (int i = 0; i < repeat; ++i) {
            cublas_tensor_op_nn(d_a, d_b, d_c, M, N, K);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float msec_cublas = 0;
        cudaEventElapsedTime(&msec_cublas, start, stop);
        float avg_msec_cublas = msec_cublas / repeat;
        double tflops_cublas = (double)M * N * K * 2 / 1e12 / (avg_msec_cublas / 1000.0);

        cudaMemset(d_c, 0, size_c);  // Initialize output to zero

        // ===== WMMA_STAGES KERNEL =====
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
        constexpr int N_STAGES = 3;
        constexpr int SMEM_SIZE_STAGES = (((N_STAGES) * BM * (BK + PAD_A)) + ((N_STAGES) * BK * (BN + PAD_B))) * sizeof(half);
        constexpr int NUM_THREADS_STAGES = (WARPS_PER_BLOCK_M * WARPS_PER_BLOCK_N * WARP_SIZE);

        dim3 blockDim_stages(NUM_THREADS_STAGES);
        dim3 gridDim_stages((N + BN - 1) / BN, (M + BM - 1) / BM);

        cudaFuncSetAttribute(
            hgemm_m16n16k16mma4x4_wp4x4_stages<
                WMMA_SIZE_M, WMMA_SIZE_N, WMMA_SIZE_K,
                WARPS_PER_BLOCK_M, WARPS_PER_BLOCK_N,
                WMMA_PER_WARP_M, WMMA_PER_WARP_N,
                PAD_A, PAD_B, N_STAGES
            >,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            SMEM_SIZE_STAGES);

        // Warmup for wmma_stages
        for (int i = 0; i < warmup; ++i) {
            hgemm_m16n16k16mma4x4_wp4x4_stages<
                WMMA_SIZE_M, WMMA_SIZE_N, WMMA_SIZE_K,
                WARPS_PER_BLOCK_M, WARPS_PER_BLOCK_N,
                WMMA_PER_WARP_M, WMMA_PER_WARP_N,
                PAD_A, PAD_B, N_STAGES
            ><<<gridDim_stages, blockDim_stages, SMEM_SIZE_STAGES>>>(d_a, d_b, d_c, M, N, K);
        }
        cudaDeviceSynchronize();

        // Benchmark wmma_stages
        cudaEventRecord(start);
        for (int i = 0; i < repeat; ++i) {
            hgemm_m16n16k16mma4x4_wp4x4_stages<
                WMMA_SIZE_M, WMMA_SIZE_N, WMMA_SIZE_K,
                WARPS_PER_BLOCK_M, WARPS_PER_BLOCK_N,
                WMMA_PER_WARP_M, WMMA_PER_WARP_N,
                PAD_A, PAD_B, N_STAGES
            ><<<gridDim_stages, blockDim_stages, SMEM_SIZE_STAGES>>>(d_a, d_b, d_c, M, N, K);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float msec_stages = 0;
        cudaEventElapsedTime(&msec_stages, start, stop);
        float avg_msec_stages = msec_stages / repeat;
        double tflops_stages = (double)M * N * K * 2 / 1e12 / (avg_msec_stages / 1000.0);

        printf("M=N=K = %5d | %6.3f ms | %6.2f TFLOPS | %6.3f ms | %6.2f TFLOPS | %6.3f ms | %6.2f TFLOPS |\n", 
               MNK, avg_msec_stages, tflops_stages, avg_msec_async, tflops_async, avg_msec_cublas, tflops_cublas);

        // Add to CSV buffer
        char csv_line[200];
        sprintf(csv_line, "%d,%.3f,%.2f,%.3f,%.2f,%.3f,%.2f\n", 
                MNK, avg_msec_stages, tflops_stages, avg_msec_async, tflops_async, avg_msec_cublas, tflops_cublas);
        strcat(csv_buffer, csv_line);

        // Clean up host memory
        free(h_a);
        free(h_b);
        
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }
    
    // Write CSV to file
    FILE *csv_file = fopen("benchmark_results.csv", "w");
    if (csv_file) {
        fprintf(csv_file, "%s", csv_buffer);
        fclose(csv_file);
        printf("\nResults exported to benchmark_results.csv\n");
    }
    
    destroy_cublas_handle();
    return 0;
}