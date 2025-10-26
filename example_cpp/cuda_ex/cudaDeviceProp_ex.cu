#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; i++) {

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "***********device:" << i << "*************" << std::endl;
        std::cout << "name:" << prop.name << std::endl;
        std::cout << "SM Count:" << prop.multiProcessorCount << std::endl;
        std::cout << "Global memory:" << prop.totalGlobalMem / 1e9 << "G" << std::endl;
        std::cout << "sharedMem Per Block:" << prop.sharedMemPerBlock / 1024 << "KB" << std::endl;
        std::cout << "Register Count Per Block:" << prop.regsPerBlock << std::endl;
        std::cout << "warpSize:" << prop.warpSize << std::endl;
        std::cout << "maxThreadsPerBlock:" << prop.maxThreadsPerBlock << std::endl;
        std::cout << "maxThreadsDim:" << prop.maxThreadsDim[0] << "-" << prop.maxThreadsDim[1] << "-" << prop.maxThreadsDim[2] << std::endl;
        std::cout << "maxGridSize:" << prop.maxGridSize[0] << "-" << prop.maxGridSize[1] << "-" << prop.maxGridSize[2] << std::endl;
        std::cout << "clockRate:" << prop.clockRate << std::endl;
        std::cout << "Capability:" << prop.major << "-" << prop.minor << std::endl;
        std::cout << "concurrentKernels:" << prop.concurrentKernels << std::endl;
    }

    return 0;
}
