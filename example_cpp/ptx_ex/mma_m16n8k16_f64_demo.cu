#include <cmath>
#include <cstdio>

#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                      \
  do {                                                          \
    cudaError_t err = call;                                    \
    if (err != cudaSuccess) {                                  \
      printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
             cudaGetErrorString(err));                         \
      return 1;                                                \
    }                                                          \
  } while (0)

__global__ void mma_m16n8k16_f64_kernel(double *D) {
  int lane = threadIdx.x;  // 0~31

  double a[8];
  double b[4];
  double c[4];
  double d[4];

  // 这里用全 1，重点是验证 f64 MMA 指令能跑。
  // A fragment: 8 个 f64
  // B fragment: 4 个 f64
  // C/D fragment: 4 个 f64
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    a[i] = 1.0;
  }

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    b[i] = 1.0;
    c[i] = 0.0;
    d[i] = 0.0;
  }

  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64.rn "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7, %8, %9, %10, %11}, "
      "{%12, %13, %14, %15}, "
      "{%16, %17, %18, %19};\n"
      : "=d"(d[0]), "=d"(d[1]), "=d"(d[2]), "=d"(d[3])
      : "d"(a[0]), "d"(a[1]), "d"(a[2]), "d"(a[3]),
        "d"(a[4]), "d"(a[5]), "d"(a[6]), "d"(a[7]),
        "d"(b[0]), "d"(b[1]), "d"(b[2]), "d"(b[3]),
        "d"(c[0]), "d"(c[1]), "d"(c[2]), "d"(c[3]));

  // D 是 16x8。这里沿用 m16n8 的常见 D fragment 写回方式。
  int group = lane / 4;        // 0~7
  int tid_in_group = lane % 4;  // 0~3
  int col = tid_in_group * 2;

  D[(group + 0) * 8 + col + 0] = d[0];
  D[(group + 0) * 8 + col + 1] = d[1];
  D[(group + 8) * 8 + col + 0] = d[2];
  D[(group + 8) * 8 + col + 1] = d[3];
}

int main() {
  constexpr int M = 16;
  constexpr int N = 8;
  constexpr int size = M * N;

  double h_D[size];
  double *d_D;

  for (int i = 0; i < size; ++i) {
    h_D[i] = 0.0;
  }

  CHECK_CUDA(cudaMalloc(&d_D, sizeof(h_D)));
  CHECK_CUDA(cudaMemcpy(d_D, h_D, sizeof(h_D), cudaMemcpyHostToDevice));

  mma_m16n8k16_f64_kernel<<<1, 32>>>(d_D);

  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(h_D, d_D, sizeof(h_D), cudaMemcpyDeviceToHost));

  int errors = 0;
  for (int i = 0; i < size; ++i) {
    if (fabs(h_D[i] - 16.0) > 1e-9) {
      if (errors < 10) {
        printf("Mismatch at %d: got %.17g, expected 16\n", i, h_D[i]);
      }
      errors++;
    }
  }

  if (errors == 0) {
    printf("PASS\n");
  } else {
    printf("FAIL: %d mismatches\n", errors);
  }

  CHECK_CUDA(cudaFree(d_D));
  return 0;
}