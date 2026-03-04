#include <cuda_fp16.h>

__global__ void test_kernel(__half2 *out, const __half2 *a, const __half2 *b) {
    int idx = threadIdx.x;
    out[idx] = __hmul2(a[idx], b[idx]);
}