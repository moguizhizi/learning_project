#include <cstdint>
#include <cstdio>

bool use_vec(uint32_t num_tokens, uint32_t elementSize, uint32_t num_elements) {
    if (elementSize == 0) {
        return false;
    }

    int device = -1;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        printf("cudaGetDevice failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    cudaDeviceProp prop;
    auto error = cudaGetDeviceProperties(&prop, device);
    if (error != cudaSuccess) {
        printf("cudaGetDeviceProperties returned %d\n-> %s\n", (int)error, cudaGetErrorString(error));
        return false;
    }

    int cc_major = prop.major;
    int support_vec = (cc_major >= 10 && num_tokens > 128) ? 32 : 16;
    int vec_size = support_vec / elementSize;
    bool use_vec = (num_elements % vec_size == 0);

    return use_vec;
}

int main() {
    uint32_t num_tokens = 256;
    uint32_t elementSize = sizeof(float);
    uint32_t num_elements = 1024;

    bool result = use_vec(num_tokens, elementSize, num_elements);

    printf("use_vec = %s\n", result ? "true" : "false");

    return 0;
}