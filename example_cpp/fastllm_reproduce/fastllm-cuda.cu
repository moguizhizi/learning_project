#include "fastllm-cuda.cuh"

#define checkCudaErrors(message, val) showError(val, message, __FILE__, __LINE__)

void showError(cudaError_t result, char const *const message, const char *const file, int const line) {
    if (cudaSuccess != result) {
        printf("%s\n  CUDA error = %d, %s at %s:%d\n  '%s'\n", message, result, cudaGetErrorName(result), file, line, cudaGetErrorString(result));
    }
}

void *FastllmCudaMalloc(size_t size) {
    int id = -1;
    cudaError state = cudaGetDevice(&id);
    checkCudaErrors("Error: CUDA error when find device!", state);

    if (size > 1024 * 1024) {
        auto &bigBuffers = bigBuffersMap[id];
        int setId = -1;
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (bigBuffers[i].size >= size && !bigBuffers[i].busy && (bigBuffers[i].size - size < 1024 * 1024)) {
                if (setId == -1 || bigBuffers[setId].size > bigBuffers[i].size) {
                    setId = i;
                }
            }
        }

        if (setId != -1) {
            bigBuffers[setId].busy = true;
            return bigBuffers[setId].data;
        }

        void *ret;

        cudaError state = cudaMalloc(&ret, size);
        if (state != cudaSuccess) {
            printf("Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory left on device.", size >> 20);
            checkCudaErrors("", state);
            return nullptr;
        }
        bigBuffers.push_back(CudaMemoryBuffer(ret, size, true));
        return ret;
    }

    auto &cudaBuffers = cudaBuffersMap[id];
    for (int i = cudaBuffersMinId[id]; i < cudaBuffers.size(); i++) {
        if (cudaBuffers[i].size > size && !cudaBuffers[i].busy) {
            cudaBuffers[i].busy = true;
            noBusyCnt[id] -= cudaBuffers[i].size;
            while (cudaBuffersMinId[id] < cudaBuffers.size() && cudaBuffers[cudaBuffersMinId[id]].busy) {
                cudaBuffersMinId[id]++;
            }
            return cudaBuffers[i].data;
        }
    }

    void *ret;

    cudaError state = cudaMalloc(&ret, size);
    if (state != cudaSuccess) {
        printf("Error: CUDA error when allocating %lu KB memory! maybe there's no enough memory left on device.", size >> 10);
        checkCudaErrors("", state);
        return nullptr;
    }
    cudaBuffers.push_back(CudaMemoryBuffer(ret, size, true));
    return ret;
}

void FastllmCudaCopyFromHostToDevice(void *dst, void *src, size_t size) {
    cudaError state = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when copy from memory to GPU!", state);
}