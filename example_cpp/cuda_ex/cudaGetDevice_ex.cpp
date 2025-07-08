#include <cuda_runtime.h>
#include <iostream>

int main() {
    int id = -1;
    cudaError state = cudaGetDevice(&id);
    if(state == cudaSuccess){
        std::cout << "当前使用的 GPU 设备 ID 是: " << id << std::endl;
    }
    else{
        std::cout << "获取 GPU ID 失败 : " << cudaGetErrorString(state) << id << std::endl;
    }
}