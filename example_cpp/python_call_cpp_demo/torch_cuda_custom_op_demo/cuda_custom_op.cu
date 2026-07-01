#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace {

__global__ void add_kernel(const float* a, const float* b, float* out,
                           int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
  TORCH_CHECK(a.scalar_type() == torch::kFloat32, "a must be float32");
  TORCH_CHECK(b.scalar_type() == torch::kFloat32, "b must be float32");
  TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
  TORCH_CHECK(b.is_contiguous(), "b must be contiguous");

  auto out = torch::empty_like(a);
  int64_t n = a.numel();
  constexpr int threads = 256;
  int blocks = static_cast<int>((n + threads - 1) / threads);

  add_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(),
                                  out.data_ptr<float>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return out;
}

}  // namespace

TORCH_LIBRARY(my_cuda_ops, m) {
  m.def("add_cuda(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_cuda_ops, CUDA, m) {
  m.impl("add_cuda", &add_cuda);
}
