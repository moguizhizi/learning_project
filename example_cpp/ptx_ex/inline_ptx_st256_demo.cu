#include <cuda_runtime.h>
#include <stdio.h>

struct u32x8_t {
    unsigned int u0, u1, u2, u3, u4, u5, u6, u7;
};

__device__ __forceinline__ void st256(u32x8_t &val, u32x8_t *ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && defined(CUDA_VERSION) && CUDA_VERSION >= 12090
    asm volatile("st.global.v8.u32 [%0],{%1,%2,%3,%4,%5,%6,%7};"
        :
        : "l"(ptr), "r"(val.u0), "r"(val.u1), "r"(val.u2), "r"(val.u3), "r"(val.u4), "r"(val.u5), "r"(val.u6), "r"(val.u7)
        : "memory");
#else
    uint4 *uint4_ptr = reinterpret_cast<uint4 *>(ptr);
    uint4_ptr[0] = make_uint4(val.u0, val.u1, val.u2, val.u3);
    uint4_ptr[1] = make_uint4(val.u4, val.u5, val.u6, val.u7);
#endif
}

__global__ void kernel(u32x8_t *out) {
    u32x8_t val;

    val.u0 = 1;
    val.u1 = 2;
    val.u2 = 3;
    val.u3 = 4;
    val.u4 = 5;
    val.u5 = 6;
    val.u6 = 7;
    val.u7 = 8;

    st256(val, out);
}

int main() {
    u32x8_t *d_out;
    u32x8_t h_out;

    cudaMalloc(&d_out, sizeof(u32x8_t));

    kernel<<<1, 1>>>(d_out);

    cudaMemcpy(&h_out, d_out, sizeof(u32x8_t), cudaMemcpyDeviceToHost);

    printf("%u %u %u %u %u %u %u %u\n", h_out.u0, h_out.u1, h_out.u2, h_out.u3, h_out.u4, h_out.u5, h_out.u6, h_out.u7);

    cudaFree(d_out);
}