#include <cuda_runtime.h>
#include <iostream>

int main() {
    cudaError state = cudaSuccess;
    int canAccess;

    state = cudaDeviceCanAccessPeer(&canAccess, 2, 1);
    if (state == cudaSuccess) {
        if (canAccess) {
            std::cout << "3,1 can Access" << std::endl;
        } else {
            std::cout << "3,1 can not Access" << std::endl;
        }
    }
}