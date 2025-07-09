#pragma once

#include "struct_space.hpp"
#include <map>
#include <vector>

std::map<int, std::vector<CudaMemoryBuffer>> cudaBuffersMap;
std::map<int, int> cudaBuffersMinId;
std::map<int, size_t> noBusyCnt;
std::map<int, std::vector<CudaMemoryBuffer>> bigBuffersMap;

void *FastllmCudaMalloc(size_t);
void showError(cudaError_t result, char const *const message, const char *const file, int const line);
void FastllmCudaCopyFromHostToDevice(void *dst, void *src, size_t size);
void FastllmCudaCopyFromDeviceToHost(void *dst, void *src, size_t size);
void FastllmCudaCopyFromDeviceToDevice(void *dst, void *src, size_t size);