#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    const int m = 3;
    const int n = 4;
    const int k = 2;
    const int batchCount = 5;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 主机内存
    float h_A[m * k * batchCount]; // m×k × batch
    float h_B[k * n * batchCount]; // k×n × batch
    float h_C[m * n * batchCount]; // m×n × batch
    for (int b = 0; b < batchCount; ++b) {
        for (int i = 0; i < m * k; ++i)
            h_A[b * m * k + i] = i + 1 + b * 10;
        for (int i = 0; i < k * n; ++i)
            h_B[b * k * n + i] = i + 1 + b * 10;
        for (int i = 0; i < m * n; ++i)
            h_C[b * m * n + i] = 0;
    }

    // 设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(float) * m * k * batchCount);
    cudaMalloc(&d_B, sizeof(float) * k * n * batchCount);
    cudaMalloc(&d_C, sizeof(float) * m * n * batchCount);
    cudaMemcpy(d_A, h_A, sizeof(float) * m * k * batchCount, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * k * n * batchCount, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sizeof(float) * m * n * batchCount, cudaMemcpyHostToDevice);

    // CUBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // strides: 相邻两个矩阵之间的元素间隔
    long long strideA = m * k;
    long long strideB = k * n;
    long long strideC = m * n;

    // 调用: C = α·A·B + β·C  (列主序)
    cublasSgemmStridedBatched(handle,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              n,
                              m,
                              k, // 列主序维度
                              &alpha,
                              d_B,
                              n,
                              strideB, // B: k×n
                              d_A,
                              k,
                              strideA, // A: m×k
                              &beta,
                              d_C,
                              n,
                              strideC, // C: m×n
                              batchCount);

    cudaMemcpy(h_C, d_C, sizeof(float) * m * n * batchCount, cudaMemcpyDeviceToHost);

    // 打印前 2 个 batch
    for (int b = 0; b < 2; ++b) {
        printf("Batch %d (m×n):\n", b);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j)
                printf("%6.1f ", h_C[b * strideC + i * n + j]);
            printf("\n");
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    return 0;
}