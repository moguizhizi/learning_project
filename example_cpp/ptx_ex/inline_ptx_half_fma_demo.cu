#include <cuda_fp16.h>
#include <stdio.h>

__device__ __forceinline__ __half hfma_ptx(__half a, __half b, __half c) {
    __half val;

    asm("{fma.rn.f16 %0,%1,%2,%3;}"
        : "=h"(*reinterpret_cast<unsigned short *>(&val))
        : "h"(*reinterpret_cast<unsigned short *>(&a)), "h"(*reinterpret_cast<unsigned short *>(&b)),
        "h"(*reinterpret_cast<unsigned short *>(&c)));

    return val;
}

__global__ void test_kernel() {
    __half a = __float2half(2.0f);
    __half b = __float2half(3.0f);
    __half c = __float2half(4.0f);

    __half r = hfma_ptx(a, b, c);

    printf("result = %f\n", __half2float(r));
}

int main() {
    test_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
