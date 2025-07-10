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

void FastllmCudaFree(void *ret) {
    if (ret == nullptr) {
        return;
    }
    if (cudaBuffersMap.empty()) {
        return;
    }

    cudaError state = cudaSuccess;

    for (auto &it : cudaBuffersMap) {
        if (noBusyCnt[it.first] > 1024 * 1024 * 1024) {
            auto &cudaBuffers = it.second;
            std::vector<CudaMemoryBuffer> temp;
            for (int i = 0; i < cudaBuffers.size(); i++) {
                if (!cudaBuffers[i].busy) {
                    state = cudaSetDevice(it.first);
                    state = cudaFree(cudaBuffers[i].data);
                    if (state != cudaSuccess) {
                        printf("Error: CUDA error when release memory on device %d!", it.first);
                    }
                    checkCudaErrors("", state);
                } else {
                    temp.push_back(cudaBuffers[i]);
                }
            }
            cudaBuffers.clear();
            it.second = temp;
            noBusyCnt[it.first] = 0;
        }
    }

    for (auto &it : cudaBuffersMap) {
        auto &cudaBuffers = it.second;
        for (int i = 0; i < cudaBuffers.size(); i++) {
            if (cudaBuffers[i].data == ret) {
                noBusyCnt[it.first] += cudaBuffers[i].size;
                cudaBuffers[i].busy = false;
                cudaBuffersMinId[it.first] = std::min(cudaBuffersMinId[it.first], i);
                return;
            }
        }
        auto &bigBuffers = bigBuffersMap[it.first];
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (bigBuffers[i].data == ret) {
                bigBuffers[i].busy = false;
                return;
            }
        }
    }

    state = cudaFree(ret);
    checkCudaErrors("CUDA error when release memory!", state);
}

void FastllmCudaCopyFromHostToDevice(void *dst, void *src, size_t size) {
    cudaError state = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when copy from memory to GPU!", state);
}

void FastllmCudaCopyFromDeviceToHost(void *dst, void *src, size_t size) {
    cudaError state = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    checkCudaErrors("Error: CUDA error when copy from GPU to memory!", state);
}

void FastllmCudaCopyFromDeviceToDevice(void *dst, void *src, size_t size) {
    cudaError state = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    checkCudaErrors("Error: CUDA error when copy on GPU!", state);
}

void FastllmCudaMemcpyBetweenDevices(int dstId, void *dst, int srcId, void *src, size_t size) {
    int canAccess = 0;
    cudaError state = cudaDeviceCanAccessPeer(&canAccess, dstId, srcId);
    if (state == cudaSuccess && canAccess) {
        cudaMemcpyPeer(dst, dstId, src, srcId, size);
    } else {
        uint8_t *cpudata = new uint8_t[size];
        cudaSetDevice(srcId);
        cudaMemcpy(cpudata, src, size, cudaMemcpyDeviceToHost);
        cudaSetDevice(dstId);
        cudaMemcpy(dst, cpudata, size, cudaMemcpyHostToDevice);
        delete[] cpudata;
    }
    checkCudaErrors("Error: CUDA error when copy Between GPUs!", state);
    DeviceSync();
}

void DeviceSync() {
    // cudaDeviceSynchronize();
}