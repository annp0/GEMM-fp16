#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

static cublasHandle_t g_handle = nullptr;

void init_cublas_handle() {
  if (g_handle == nullptr) {
    cublasStatus_t status = cublasCreate(&g_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      printf("Failed to create cuBLAS handle: %d", status);
      exit(EXIT_FAILURE);
    }
    // enable tensor code operations
    status = cublasSetMathMode(g_handle, CUBLAS_TENSOR_OP_MATH);
    if (status != CUBLAS_STATUS_SUCCESS) {
      printf("Failed to set cuBLAS Math Mode: %d", status);
      exit(EXIT_FAILURE);
    }
  }
}

void destroy_cublas_handle() {
  if (g_handle != nullptr) {
    cublasStatus_t status = cublasDestroy(g_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      printf("Failed to destroy cuBLAS handle: %d", status);
    }
    g_handle = nullptr;
  }
}

// NN: A/B/C All row major
void cublas_tensor_op_nn(
  half *A, half *B, half *C, 
  size_t M, size_t N, size_t K
) {
  static half alpha = __float2half(1.0);
  static half beta = __float2half(0.0);

  if (g_handle == nullptr) {
    init_cublas_handle();
  }

  cublasGemmEx(g_handle, 
               CUBLAS_OP_N, 
               CUBLAS_OP_N, 
               N, M, K, 
               &alpha, 
               B, CUDA_R_16F, N, 
               A, CUDA_R_16F, K, 
               &beta,  
               C, CUDA_R_16F, N, 
               CUBLAS_COMPUTE_16F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
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

// NN: A/B/C All row major
void hgemm_cublas(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  cublas_tensor_op_nn(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}

#endif
