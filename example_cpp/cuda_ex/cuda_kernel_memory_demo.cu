#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

// 生成随机数 kernel
__global__ void rand_kernel(float *ptr, int n, unsigned int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        unsigned int x = seed ^ i;
        x = x * 1664525u + 1013904223u;
        ptr[i] = (x % 1000) / 1000.0f; // 简单伪随机
    }
}

// 乘2 kernel
__global__ void mul2_kernel(float *ptr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        ptr[i] *= 2.0f;
    }
}

int main() {
    int n = 10;
    int device = 1;

    // 1 设置 GPU1
    cudaSetDevice(device);

    // 2 GPU 分配显存
    float *d_ptr;
    cudaMalloc(&d_ptr, n * sizeof(float));

    int block = 256;
    int grid = (n + block - 1) / block;

    // 3 在 GPU 生成随机数
    rand_kernel<<<grid, block>>>(d_ptr, n, 1234);

    // 4 GPU kernel 乘2
    mul2_kernel<<<grid, block>>>(d_ptr, n);

    // 5 拷贝回 CPU
    std::vector<float> h(n);

    cudaMemcpy(h.data(), d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost);

    // 6 打印
    std::cout << "after:\n";

    for (auto v : h) std::cout << v << std::endl;

    // 7 释放显存
    cudaFree(d_ptr);

    return 0;
}