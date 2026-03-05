#include <cuda_fp16.h>

#include <iostream>

__device__ __forceinline__ __half2 my_mul_f16x2(__half2 a, __half2 b) {
    __half2 val;
    asm("{ mul.f16x2 %0,%1,%2; }"
        : "=r"(*reinterpret_cast<unsigned int *>(&val))
        : "r"(*reinterpret_cast<unsigned int *>(&a)), "r"(*reinterpret_cast<unsigned int *>(&b)));
    return val;
}

__global__ void kernel(__half2 *out, __half2 *a, __half2 *b) {
    int idx = threadIdx.x;
    out[idx] = my_mul_f16x2(a[idx], b[idx]);
}

int main() {
    const int N = 4;

    __half2 h_a[N], h_b[N], h_out[N];

    for (int i = 0; i < N; i++) {
        h_a[i] = __halves2half2(__float2half(2.0f), __float2half(3.0f));
        h_b[i] = __halves2half2(__float2half(4.0f), __float2half(5.0f));
    }

    __half2 *d_a, *d_b, *d_out;

    cudaMalloc(&d_a, N * sizeof(__half2));
    cudaMalloc(&d_b, N * sizeof(__half2));
    cudaMalloc(&d_out, N * sizeof(__half2));

    cudaMemcpy(d_a, h_a, N * sizeof(__half2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(__half2), cudaMemcpyHostToDevice);

    kernel<<<1, N>>>(d_out, d_a, d_b);

    cudaMemcpy(h_out, d_out, N * sizeof(__half2), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        float x = __half2float(__low2half(h_out[i]));
        float y = __half2float(__high2half(h_out[i]));
        std::cout << x << ", " << y << std::endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return 0;
}