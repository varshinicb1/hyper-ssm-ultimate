#include <torch/extension.h>

// CUDA forward declarations defining the block functions implemented in riemannian_kernel.cu
torch::Tensor lorentz_product_cuda_forward(torch::Tensor x, torch::Tensor y);
torch::Tensor project_to_tangent_cuda_forward(torch::Tensor x, torch::Tensor g);

// C++ Interface logic resolving device constraints and routing to CUDA implementations
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor lorentz_product_forward(torch::Tensor x, torch::Tensor y) {
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    return lorentz_product_cuda_forward(x, y);
}

torch::Tensor project_to_tangent_forward(torch::Tensor x, torch::Tensor g) {
    CHECK_INPUT(x);
    CHECK_INPUT(g);
    return project_to_tangent_cuda_forward(x, g);
}

// PyBind11 Module definition exposing the C++ implementations structurally to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lorentz_product", &lorentz_product_forward, "Hyper-SSM Minkowski Inner Product (CUDA)");
    m.def("project_to_tangent", &project_to_tangent_forward, "Hyper-SSM Riemannian Tangent Projection (CUDA)");
}
