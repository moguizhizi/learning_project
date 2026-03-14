#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

// -----------------------------
// activation function
// -----------------------------
__device__ float silu(const float &x) {
    return x / (1.0f + expf(-x));
}

// packed activation (这里简单返回)
__device__ float silu_packed(const float &x) {
    return silu(x);
}

// -----------------------------
// compute
// -----------------------------
template <typename T, T (*ACT_FN)(const T &), bool act_first>
__device__ T compute(T x, T y) {
    if constexpr (act_first)
        return ACT_FN(x) * y;
    else
        return ACT_FN(x * y);
}

// -----------------------------
// Kernel（简化版）
// -----------------------------
template <typename scalar_t, typename packed_t, scalar_t (*ACT_FN)(const scalar_t &), packed_t (*PACKED_ACT_FN)(const packed_t &),
    bool act_first, bool use_vec>
__global__ void act_and_mul_kernel(scalar_t *__restrict__ out, const scalar_t *__restrict__ input, const int d) {
    const scalar_t *x_ptr = input + blockIdx.x * 2 * d;
    const scalar_t *y_ptr = x_ptr + d;
    scalar_t *out_ptr = out + blockIdx.x * d;

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        scalar_t x = x_ptr[i];
        scalar_t y = y_ptr[i];
        out_ptr[i] = compute<scalar_t, ACT_FN, act_first>(x, y);
    }
}

// -----------------------------
// host main
// -----------------------------
int main() {
    int N = 4;  // token
    int D = 16; // hidden

    int input_size = N * 2 * D;
    int output_size = N * D;

    std::vector<float> h_input(input_size);

    for (int i = 0; i < input_size; i++) h_input[i] = (rand() % 100) / 100.0f;

    float *d_input, *d_out;

    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_out, output_size * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);

    int block = 128;
    int grid = N;

    // -----------------------------
    // Kernel launch
    // -----------------------------
    act_and_mul_kernel<float, // scalar_t
        float,                // packed_t
        silu,                 // ACT_FN
        silu_packed,          // PACKED_ACT_FN
        true,                 // act_first
        false                 // use_vec
        ><<<grid, block>>>(d_out, d_input, D);

    cudaDeviceSynchronize();

    std::vector<float> h_out(output_size);

    cudaMemcpy(h_out.data(), d_out, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Output:\n";

    for (int i = 0; i < N; i++) {
        std::cout << "token " << i << " : ";
        for (int j = 0; j < D; j++) std::cout << h_out[i * D + j] << " ";
        std::cout << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_out);

    return 0;
}