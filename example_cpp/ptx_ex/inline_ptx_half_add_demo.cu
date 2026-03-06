#include <cuda_fp16.h>
#include <stdio.h>

__device__ __forceinline__ __half my_hadd(__half a, __half b) {
    __half val;

    asm("{ add.f16 %0,%1,%2; }"
        : "=h"(*reinterpret_cast<unsigned short *>(&val))
        : "h"(*reinterpret_cast<unsigned short *>(&a)), "h"(*reinterpret_cast<unsigned short *>(&b)));

    return val;
}

__global__ void test_kernel(float *out) {
    __half a = __float2half(2.0f);
    __half b = __float2half(3.0f);

    __half c = my_hadd(a, b);

    out[0] = __half2float(c); // 转回 float 方便打印
}

int main() {
    float h_out = 0.0f;
    float *d_out;

    cudaMalloc(&d_out, sizeof(float));

    test_kernel<<<1, 1>>>(d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("2 + 3 = %f\n", h_out);

    cudaFree(d_out);
    return 0;
}