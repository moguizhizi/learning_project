#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <type_traits>

template <typename packed_t>
__device__ __forceinline__ packed_t packed_mul(const packed_t &a, const packed_t &b) {
    if constexpr (std::is_same_v<packed_t, __half2> || std::is_same_v<packed_t, __nv_bfloat162>) {
        return __hmul2(a, b);
    } else if constexpr (std::is_same_v<packed_t, float2>) {
        return make_float2(a.x * b.x, a.y * b.y);
    }
}

__global__ void kernel_half2() {
    __half2 a = __halves2half2(__float2half(2.0f), __float2half(3.0f));
    __half2 b = __halves2half2(__float2half(4.0f), __float2half(5.0f));

    __half2 c = packed_mul(a, b);

    float2 result = __half22float2(c);

    printf("half2 result: %f %f\n", result.x, result.y);
}

__global__ void kernel_float2() {
    float2 a = make_float2(2.0f, 3.0f);
    float2 b = make_float2(4.0f, 5.0f);

    float2 c = packed_mul(a, b);

    printf("float2 result: %f %f\n", c.x, c.y);
}

int main() {
    kernel_half2<<<1, 1>>>();
    kernel_float2<<<1, 1>>>();

    cudaDeviceSynchronize();
}