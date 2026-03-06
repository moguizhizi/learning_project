#include <cuda_fp16.h>
#include <stdio.h>

__device__ __forceinline__ __half2 hfma2_ptx(__half2 a, __half2 b, __half2 c) {
    __half2 val;

    asm("{fma.rn.f16x2 %0,%1,%2,%3;}"
        : "=r"(*reinterpret_cast<unsigned int *>(&val))
        : "r"(*reinterpret_cast<unsigned int *>(&a)), "r"(*reinterpret_cast<unsigned int *>(&b)), "r"(*reinterpret_cast<unsigned int *>(&c)));

    return val;
}

__global__ void test_kernel() {
    // 构造 half2
    __half2 a = __halves2half2(__float2half(2.0f), __float2half(5.0f));
    __half2 b = __halves2half2(__float2half(3.0f), __float2half(6.0f));
    __half2 c = __halves2half2(__float2half(4.0f), __float2half(7.0f));

    __half2 r = hfma2_ptx(a, b, c);

    // 拆成两个 half
    float x = __half2float(__low2half(r));
    float y = __half2float(__high2half(r));

    printf("result.x = %f\n", x);
    printf("result.y = %f\n", y);
}

int main() {
    test_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}