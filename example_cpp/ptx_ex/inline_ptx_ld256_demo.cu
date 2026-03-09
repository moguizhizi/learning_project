#include <cuda_runtime.h>
#include <stdio.h>

struct u32x8_t {
    unsigned int u0, u1, u2, u3, u4, u5, u6, u7;
};

__device__ __forceinline__ void ld256(u32x8_t &val, u32x8_t *data) {
    asm volatile("ld.global.nc.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8]"
        : "=r"(val.u0), "=r"(val.u1), "=r"(val.u2), "=r"(val.u3), "=r"(val.u4), "=r"(val.u5), "=r"(val.u6), "=r"(val.u7)
        : "l"(data));
}

__global__ void kernel(u32x8_t *data) {
    u32x8_t val;

    ld256(val, data);

    printf("%u %u %u %u %u %u %u %u\n", val.u0, val.u1, val.u2, val.u3, val.u4, val.u5, val.u6, val.u7);
}

int main() {
    u32x8_t h;

    h.u0 = 1;
    h.u1 = 2;
    h.u2 = 3;
    h.u3 = 4;
    h.u4 = 5;
    h.u5 = 6;
    h.u6 = 7;
    h.u7 = 8;

    u32x8_t *d;

    cudaMalloc(&d, sizeof(u32x8_t));
    cudaMemcpy(d, &h, sizeof(u32x8_t), cudaMemcpyHostToDevice);

    kernel<<<1, 1>>>(d);
    cudaDeviceSynchronize();

    cudaFree(d);
}