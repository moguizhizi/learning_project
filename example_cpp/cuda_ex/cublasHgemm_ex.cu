#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    const int m = 4, n = 3, k = 2;
    const half alpha = __float2half(1.0f), beta = __float2half(0.0f);

    half h_A[m * k], h_B[k * n], h_C[m * n];
    for (int i = 0; i < m * k; ++i)
        h_A[i] = __float2half(i + 1.f);
    for (int i = 0; i < k * n; ++i)
        h_B[i] = __float2half(i + 1.f);

    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(half) * m * k);
    cudaMalloc(&d_B, sizeof(half) * k * n);
    cudaMalloc(&d_C, sizeof(half) * m * n);
    cudaMemcpy(d_A, h_A, sizeof(half) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(half) * k * n, cudaMemcpyHostToDevice);

    cublasHandle_t h;
    cublasCreate(&h);
    cublasHgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_B, k, &beta, d_C, n);
    cudaMemcpy(h_C, d_C, sizeof(half) * m * n, cudaMemcpyDeviceToHost);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j)
            printf("%6.2f ", __half2float(h_C[i * n + j]));
        printf("\n");
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(h);
    return 0;
}