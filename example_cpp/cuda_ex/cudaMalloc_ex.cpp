#include <cuda_runtime.h>
#include <iostream>

int main() {
    int *darray;
    size_t size = 100 * sizeof(int);
    cudaError state = cudaMalloc((void **)&darray, size);
    if (state == cudaSuccess) {
        std::cout << "GPU 內存分配成功， 地址：" << darray << std::endl;
    } else {
        std::cout << "GPU 內存分配失敗" << std::endl;
    }

    cudaFree(darray);
    return 0;
}