#include <cmath>
#include <cstdint>
#include <cstdio>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do {                                      \
  cudaError_t err = call;                                          \
  if (err != cudaSuccess) {                                        \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
           cudaGetErrorString(err));                               \
    return 1;                                                      \
  }                                                                \
} while (0)

__device__ __forceinline__ uint32_t pack_half2(__half x, __half y) {
  __half2 h2 = __halves2half2(x, y);
  return *reinterpret_cast<uint32_t *>(&h2);
}

__global__ void mma_m16n8k16_f32_f16_f16_f16_kernel(
    const __half *A,
    const __half *B,
    float *D) {
  int lane = threadIdx.x;  // 0~31

  uint32_t a[4];
  uint32_t b[2];
  uint32_t c[2];
  uint32_t d[2];

  int a_row_base = lane / 4;        // 0~7
  int a_col_pair = (lane % 4) * 2;  // 0,2,4,6

  a[0] = pack_half2(A[(a_row_base + 0) * 16 + (a_col_pair + 0)],
                    A[(a_row_base + 0) * 16 + (a_col_pair + 1)]);

  a[1] = pack_half2(A[(a_row_base + 8) * 16 + (a_col_pair + 0)],
                    A[(a_row_base + 8) * 16 + (a_col_pair + 1)]);

  a[2] = pack_half2(A[(a_row_base + 0) * 16 + (a_col_pair + 8)],
                    A[(a_row_base + 0) * 16 + (a_col_pair + 9)]);

  a[3] = pack_half2(A[(a_row_base + 8) * 16 + (a_col_pair + 8)],
                    A[(a_row_base + 8) * 16 + (a_col_pair + 9)]);

  int b_col = lane / 4;             // 0~7
  int b_row_pair = (lane % 4) * 2;  // 0,2,4,6

  b[0] = pack_half2(B[(b_row_pair + 0) * 8 + b_col],
                    B[(b_row_pair + 1) * 8 + b_col]);

  b[1] = pack_half2(B[(b_row_pair + 8) * 8 + b_col],
                    B[(b_row_pair + 9) * 8 + b_col]);

  // C 是 f16 accumulator fragment。
  // 逻辑上仍然是 c0,c1,c2,c3，
  // 但实际用两个 32-bit 寄存器保存：
  // c[0] = {c0,c1}
  // c[1] = {c2,c3}
  c[0] = pack_half2(__float2half(0.0f), __float2half(0.0f));
  c[1] = pack_half2(__float2half(0.0f), __float2half(0.0f));

  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0, %1}, "
      "{%2, %3, %4, %5}, "
      "{%6, %7}, "
      "{%8, %9};\n"
      : "=r"(d[0]), "=r"(d[1])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]),
        "r"(c[0]), "r"(c[1]));

  int group = lane / 4;
  int tid_in_group = lane % 4;
  int col = tid_in_group * 2;

  __half* d0 = reinterpret_cast<__half*>(&d[0]);
  __half* d1 = reinterpret_cast<__half*>(&d[1]);

  D[(group + 0) * 8 + col + 0] = __half2float(d0[0]);
  D[(group + 0) * 8 + col + 1] = __half2float(d0[1]);
  D[(group + 8) * 8 + col + 0] = __half2float(d1[0]);
  D[(group + 8) * 8 + col + 1] = __half2float(d1[1]);
}

int main() {
  constexpr int M = 16;
  constexpr int N = 8;
  constexpr int K = 16;

  __half h_A[M * K];
  __half h_B[K * N];
  float h_D[M * N];
  float ref[M * N];

  for (int i = 0; i < M * K; ++i) {
    h_A[i] = __float2half(float((i % 7) + 1));
  }

  for (int i = 0; i < K * N; ++i) {
    h_B[i] = __float2half(float((i % 5) + 1));
  }

  for (int i = 0; i < M * N; ++i) {
    h_D[i] = 0.0f;
    ref[i] = 0.0f;
  }

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += __half2float(h_A[m * K + k]) *
               __half2float(h_B[k * N + n]);
      }
      ref[m * N + n] = sum;
    }
  }

  __half *d_A;
  __half *d_B;
  float *d_D;

  CHECK_CUDA(cudaMalloc(&d_A, sizeof(h_A)));
  CHECK_CUDA(cudaMalloc(&d_B, sizeof(h_B)));
  CHECK_CUDA(cudaMalloc(&d_D, sizeof(h_D)));

  CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeof(h_A), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeof(h_B), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_D, h_D, sizeof(h_D), cudaMemcpyHostToDevice));

  mma_m16n8k16_f32_f16_f16_f16_kernel<<<1, 32>>>(d_A, d_B, d_D);

  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(h_D, d_D, sizeof(h_D), cudaMemcpyDeviceToHost));

  int errors = 0;
  for (int i = 0; i < M * N; ++i) {
    if (fabs(h_D[i] - ref[i]) > 1e-3f) {
      if (errors < 10) {
        printf("Mismatch at %d: got %f, expected %f\n",
               i, h_D[i], ref[i]);
      }
      errors++;
    }
  }

  if (errors == 0) {
    printf("PASS\n");
  } else {
    printf("FAIL: %d mismatches\n", errors);
  }

  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_D));

  return 0;
}