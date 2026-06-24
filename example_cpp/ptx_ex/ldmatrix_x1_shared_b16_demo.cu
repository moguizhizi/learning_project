#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

__global__ void ldmatrix_x1_demo() {
  __shared__ uint16_t tile[8 * 8];

  int lane = threadIdx.x & 31;

  if (lane == 0) {
    for (int i = 0; i < 8 * 8; ++i) {
      tile[i] = i;
    }
  }
  __syncthreads();

  // ldmatrix 用的是 shared memory address
  uint32_t smem_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(tile + (lane % 8) * 8));

  uint32_t reg;

  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
      : "=r"(reg)
      : "r"(smem_addr));

  if (lane < 8) {
    printf("lane=%d reg=0x%08x\n", lane, reg);
  }
}

int main() {
  ldmatrix_x1_demo<<<1, 32>>>();
  cudaDeviceSynchronize();
  return 0;
}