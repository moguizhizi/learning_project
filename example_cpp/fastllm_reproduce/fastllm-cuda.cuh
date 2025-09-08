#pragma once

#include "struct_space.hpp"
#include <cstdio>
#include <cuda_runtime.h>
#include <map>
#include <vector>

extern std::map<int, std::vector<CudaMemoryBuffer>> cudaBuffersMap;
extern std::map<int, int> cudaBuffersMinId;
extern std::map<int, size_t> noBusyCnt;
extern std::map<int, std::vector<CudaMemoryBuffer>> bigBuffersMap;

void *FastllmCudaMalloc(size_t);
void showError(cudaError_t result, char const *const message, const char *const file, int const line);
void FastllmCudaCopyFromHostToDevice(void *dst, void *src, size_t size);
void FastllmCudaCopyFromDeviceToHost(void *dst, void *src, size_t size);
void FastllmCudaCopyFromDeviceToDevice(void *dst, void *src, size_t size);
void FastllmCudaMemcpyBetweenDevices(int dstId, void *dst, int srcId, void *src, size_t size);
void FastllmCudaFree(void *ret);
void FastllmCudaSetDevice(int gpu_id);
int FastllmCudaGetDevice();
void DeviceSync();
void FastllmCudaClearBigBuffer();
void FastllmCudaMemcpy2DDeviceToDevice(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height);
void *FastllmCudaDirectMalloc(size_t size);
void FastllmCudaMemset0(void *ret, size_t size);
void *FastllmCudaPrepareInput(const Data &input);
void FastllmCudaFinishInput(const Data &input, void *data);
void *FastllmCudaPrepareOutput(Data &output);