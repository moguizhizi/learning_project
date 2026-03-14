#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <iostream>

// CUDA kernel
__global__ void kernel(float *ptr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        ptr[i] *= 2.0f;
    }
}

void launch(const at::Tensor &input) {
    at::cuda::OptionalCUDAGuard device_guard(input.device());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    float *ptr = input.data_ptr<float>();

    int n = input.numel();

    int block = 256;
    int grid = (n + block - 1) / block;

    kernel<<<grid, block, 0, stream>>>(ptr, n);
}

int main() {
    // 显式创建 Device（避免 IntelliSense 误判）
    at::Device device(at::kCUDA, 1);

    // 创建 GPU tensor
    at::Tensor input = torch::randn({10}, device);

    std::cout << "before:\n" << input << std::endl;

    launch(input);

    // 等待 kernel 完成
    cudaDeviceSynchronize();

    std::cout << "after:\n" << input << std::endl;

    return 0;
}