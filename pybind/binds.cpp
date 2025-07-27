#include <torch/types.h>
#include <torch/extension.h>

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) m.def(STRINGFY(func), &func, STRINGFY(func));

void init_cublas_handle();
void destroy_cublas_handle();
void hgemm_cublas(torch::Tensor a, torch::Tensor b, torch::Tensor c);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    TORCH_BINDING_COMMON_EXTENSION(init_cublas_handle)
    TORCH_BINDING_COMMON_EXTENSION(destroy_cublas_handle)
    TORCH_BINDING_COMMON_EXTENSION(hgemm_cublas)
}