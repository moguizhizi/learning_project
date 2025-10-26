#include <cuda_runtime.h>
#include <iostream>

int main() {
    int h_height = 10;
    int h_weight = 112;
    int element_size = sizeof(float);

    float h_data[h_height * h_weight];

    for (int i = 0; i < h_height; i++) {
        for (int j = 0; j < h_weight; j++) {
            h_data[i * h_weight + j] = i + j;
        }
    }

    float *d_data;
    size_t d_pitch;

    cudaMallocPitch(&d_data, &d_pitch, h_weight * element_size, h_height);

    cudaMemcpy2D(d_data, d_pitch, h_data, h_weight * element_size, h_weight * element_size, h_height, cudaMemcpyKind::cudaMemcpyHostToDevice);

    std::cout << d_pitch << std::endl;

    cudaFree(d_data);

    return 0;
}