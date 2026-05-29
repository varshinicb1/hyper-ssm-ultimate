#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>


// ------------------------------------------------------------------------
// Custom CUDA Kernel: Lorentz Product (Minkowski Inner Product)
// Computes effectively: -x0*y0 + x1*y1 + ... + xn*yn
// ------------------------------------------------------------------------
template <typename scalar_t>
__global__ void lorentz_product_kernel(const scalar_t *__restrict__ x,
                                       const scalar_t *__restrict__ y,
                                       scalar_t *__restrict__ out,
                                       const int batch_size, const int dim) {

  // Each thread handles one full vector inner-product inside a batch/sequence.
  // Memory coalesce mapping: x is structured [B, dim]
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < batch_size) {
    int offset = idx * dim;

    // Time component (-x0*y0)
    scalar_t dot = -x[offset] * y[offset];

    // Spatial components (x1*y1 + ... + xn*yn)
    for (int d = 1; d < dim; ++d) {
      dot += x[offset + d] * y[offset + d];
    }

    // Write out aggregated continuous geometry
    out[idx] = dot;
  }
}

// ------------------------------------------------------------------------
// Custom CUDA Kernel: Tangent Space Projection
// Computes effectively: out = g + lorentz_product(x, g) * x
// ------------------------------------------------------------------------
template <typename scalar_t>
__global__ void project_to_tangent_kernel(
    const scalar_t *__restrict__ x, const scalar_t *__restrict__ g,
    const scalar_t *__restrict__ lorentz_prod, scalar_t *__restrict__ out,
    const int total_elements, const int dim) {

  // Thread mapping per-element across entire flat matrix geometry
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elements) {
    // We need to know which vector this element belongs to in order to map the
    // correct lorentz scaler to the geometry
    int vec_idx = idx / dim;

    // Equation: u + <x, u>_L * x
    out[idx] = g[idx] + (lorentz_prod[vec_idx] * x[idx]);
  }
}

// ------------------------------------------------------------------------
// PyTorch Forward Dispatchers mapping C++ tensors into CUDA threads
// ------------------------------------------------------------------------
torch::Tensor lorentz_product_cuda_forward(torch::Tensor x, torch::Tensor y) {
  // x and y shape: [..., Dim]
  // Flatten leading dimensions
  auto origin_shape = x.sizes().vec();
  int dim = origin_shape.back();

  int batch_size = x.numel() / dim;

  // Setup matching output tensor [..., 1]
  origin_shape.back() = 1;
  auto out = torch::empty(origin_shape, x.options());

  // Blocks and threads
  const int threads = 256;
  const int blocks = (batch_size + threads - 1) / threads;

  // Dispatch execution across available CUDA cores
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      x.scalar_type(), "lorentz_product_kernel", ([&] {
        lorentz_product_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(), batch_size, dim);
      }));

  return out;
}

torch::Tensor project_to_tangent_cuda_forward(torch::Tensor x,
                                              torch::Tensor g) {
  // Both x and g: [..., Dim]
  int dim = x.size(-1);
  int total_elements = x.numel();
  int batch_size = total_elements / dim;

  // Output matches input topology perfectly
  auto out = torch::empty_like(x);

  // 1. Calculate inner Minkowski product
  auto lorentz_prod = lorentz_product_cuda_forward(
      x, g); // [..., 1] (Flattened size is batch_size)

  // 2. Perform element-wise projection scaling
  const int threads = 256;
  const int blocks = (total_elements + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      x.scalar_type(), "project_to_tangent_kernel", ([&] {
        project_to_tangent_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(), g.data_ptr<scalar_t>(),
            lorentz_prod.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
            total_elements, dim);
      }));

  return out;
}
