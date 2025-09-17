#pragma once

#include "struct_space.hpp"
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <map>
#include <vector>

extern std::map<int, std::vector<CudaMemoryBuffer>> cudaBuffersMap;
extern std::map<int, int> cudaBuffersMinId;
extern std::map<int, size_t> noBusyCnt;
extern std::map<int, std::vector<CudaMemoryBuffer>> bigBuffersMap;

struct TopKFunctor {
    float *cudaInput;  // 指向原始输入数据的设备指针
    float *cudaOutput; // 指向输出数据的设备指针
    int channels;
    int topk;

    // 构造函数
    TopKFunctor(float *cudaInput, float *cudaOutput, int channels, int topk);

    __device__ __host__ void operator()(int i) const;
};

__global__ void FastllmGeluKernel(half *a, half *b, int len);
__global__ void FastllmGeluKernel(float *a, float *b, int len);
__global__ void FastllmGeluNewKernel(float *a, float *b, int len);
__global__ void FastllmSiluKernel(float *a, float *b, int len);
__global__ void FastllmSiluKernel(half *a, half *b, int len);
__global__ void FastllmAddKernel(float *a, float *b, float v, int len);
__global__ void FastllmAddKernel(half *a, half *b, half v, int len);
__global__ void FastllmMulKernel(float *a, float *b, float v, int len);
__global__ void FastllmMulKernel(half *a, half *b, half v, int len);
__global__ void FastllmAddToKernel(float *a, float *b, float alpha, int len);
__global__ void FastllmAddToKernel(half *a, half *b, half alpha, int len);
__global__ void FastllmMulToKernel(float *a, float *b, float alpha, int len);
__global__ void FastllmMulToKernel(half *a, half *b, float alpha, int len);
__global__ void FastllmCudaFloat2HalfKernel(float *a, half *b, int len);
__global__ void FastllmCudaHalf2FloatKernel(half *a, float *b, int len);
__global__ void FastllmCudaBF162FloatKernel(uint16_t *a, float *b, int len);

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
void FastllmCudaFinishOutput(Data &output, void *data);
bool FastllmCudaGelu(const Data &input, Data &output);
bool FastllmCudaGeluNew(const Data &input, Data &output);
bool FastllmFloatToHalf(void *a, void *b, int len);
bool FastllmHalfToFloat(void *a, void *b, int len);
bool FastllmBF16ToFloat(void *a, void *b, int len);
bool FastllmCudaEmbedding(const Data &input, const Data &weight, Data &output);
bool FastllmCudaRMSNorm(const Data &input, Data &weight, Data &output, float eps);
bool FastllmCudaLayerNorm(const Data &input, Data &gamma, Data &beta, Data &output, int axis);
bool FastllmCudaSoftmax(const Data &input, Data &output, int axis);
bool FastllmCudaAddTo(Data &input0, const Data &input1, float alpha);
bool FastllmCudaMulTo(Data &input0, const Data &input1, float alpha);
bool FastllmCudaMul(const Data &input, float v, Data &output);
bool FastllmCudaSoftmaxBatch(Data **inputs, Data **outputs, int axis, int batch);
bool FastllmCudaTopK(const Data &input, Data &output, int topk);
bool FastllmCudaPermute(Data &input, const std::vector<int> &axis);
int GetPointerDeviceId(void *ptr);
int FastllmCudaGetDeviceCount();
cublasHandle_t getFastllmCublasHandle();
