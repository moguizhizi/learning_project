#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    const int m = 4;
    const int n = 3;
    const int k = 2;

    // 主机内存
    float h_A[m * k] = {1, 2, 3, 4, 5, 6, 7, 8}; // m×k
    float h_B[k * n] = {1, 2, 3, 4, 5, 6};       // k×n
    float h_C[m * n] = {0};                      // m×n

    // 设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(float) * m * k);
    cudaMalloc(&d_B, sizeof(float) * k * n);
    cudaMalloc(&d_C, sizeof(float) * m * n);

    cudaMemcpy(d_A, h_A, sizeof(float) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * k * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sizeof(float) * m * n, cudaMemcpyHostToDevice);

    // CUBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 参数： C = 1·A·B + 0·C  → 纯乘法
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 行主序 → cublas 用列主序，所以把 A,B 视为转置
    // 调用 cublasSgemm:  C = α·op(A)·op(B) + β·C
    cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N, // 不转置
                n,
                m,
                k, // 列主序维度：m,n,k
                &alpha,
                d_B,
                n, // B 是 k×n，leading dim = n
                d_A,
                k, // A 是 m×k，leading dim = k
                &beta,
                d_C,
                n + 1); // C 是 m×n，leading dim = n

    cudaMemcpy(h_C, d_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result C (m×n):\n");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j)
            printf("%6.1f ", h_C[i * n + j]);
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    return 0;
}