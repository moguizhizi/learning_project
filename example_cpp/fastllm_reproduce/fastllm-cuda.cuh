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

struct CudaInfos {
    int cudaArch;
    bool hasTensorCore;

    CudaInfos();
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
__global__ void GetCudaInfoKernel(int *infos);
__global__ void InitBlockAtten(float *sum0, float *max0, float *sum1, float *max1, int len);
__global__ void FastllmRotatePosition2DKernel(float *data,
                                              float *positionIds,
                                              float *sin,
                                              float *cos,
                                              int len,
                                              int bs,
                                              int spatial,
                                              int n,
                                              int m,
                                              int partStride,
                                              int sinCosStride,
                                              int rotateDim);
__global__ void FastllmNearlyRotatePosition2DKernel(float *data,
                                                    float *positionIds,
                                                    float *sin,
                                                    float *cos,
                                                    int len,
                                                    int bs,
                                                    int spatial,
                                                    int n,
                                                    int m,
                                                    int partStride,
                                                    int sinCosStride,
                                                    int rotateDim);
__global__ void FastllmNearlyRotatePosition2DKernel(half *data,
                                                    float *positionIds,
                                                    float *sin,
                                                    float *cos,
                                                    int len,
                                                    int bs,
                                                    int spatial,
                                                    int n,
                                                    int m,
                                                    int partStride,
                                                    int sinCosStride,
                                                    int rotateDim);
__global__ void FastllmLlamaRotatePosition2DKernel(float *data,
                                                   float *positionIds,
                                                   float *sin,
                                                   float *cos,
                                                   int len,
                                                   int bs,
                                                   int spatial,
                                                   int n,
                                                   int m,
                                                   int partStride,
                                                   int sinCosStride,
                                                   int rotateDim);
__global__ void FastllmLlamaRotatePosition2DKernel(half *data,
                                                   float *positionIds,
                                                   float *sin,
                                                   float *cos,
                                                   int len,
                                                   int bs,
                                                   int spatial,
                                                   int n,
                                                   int m,
                                                   int partStride,
                                                   int sinCosStride,
                                                   int rotateDim);
__global__ void FastllmCudaBiasKernel(half *a, half *bias, int k);
__global__ void FastllmCudaBiasKernel(float *a, float *bias, int k);
__global__ void FastllmCudaInt82HalfKernel(uint8_t *a, float *scales, uint8_t *zeros, half *b, int len, int per);

CudaInfos *getCudaInfos();
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
bool FastllmCudaMLA(const Data &qNope, const Data &qPe, const Data &kvCache, const Data &peCache, Data &ss, Data &output, float softmaxScale);
bool FastllmCudaAttention(const Data &q, const Data &k, const Data &v, const Data &mask, const Data &output, int group, float scale, int maskType);
void GpuQK(half *q, half *k, half *qk, int qlen, int klen, int dim, float scale, int base);
bool FastllmCudaHalfAttention(
    const Data &q, const Data &k, const Data &v, const Data &mask, const Data &output, int group, float scale, int maskType);
bool FastllmCudaBatchMatMul(const Data &input0,
                            const Data &input1,
                            Data &output,
                            int input0Spatial,
                            int input1Spatial,
                            int outputSpatial,
                            int input0Stride,
                            int input1Stride,
                            int batch,
                            int n,
                            int m,
                            int k,
                            float alpha);
bool FastllmCudaBatchMatMulTransB(const Data &input0,
                                  const Data &input1,
                                  Data &output,
                                  int input0Spatial,
                                  int input1Spatial,
                                  int outputSpatial,
                                  int input0Stride,
                                  int input1Stride,
                                  int batch,
                                  int n,
                                  int m,
                                  int k,
                                  float alpha);
bool FastllmCudaRotatePosition2D(Data &data, const Data &positionIds, const Data &sinData, const Data &cosData, int rotaryDim);
bool FastllmCudaNearlyRotatePosition2D(Data &data, const Data &positionIds, const Data &sinData, const Data &cosData, int rotaryDim);
bool FastllmCudaLlamaRotatePosition2D(Data &data, const Data &positionIds, const Data &sinData, const Data &cosData, int rotaryDim);
bool FastllmCudaApplyLognAttn(Data &input, Data &lognAttn, Data &positionIds);
bool FastllmCudaRepeatPenalty(Data &input, Data &penalty, Data &penaltyScale);
bool FastllmCudaBatchMatMulBatch(
    void **i0s, void **i1s, void **os, int *ns, int *ms, int *ks, int *i0Strides, int *i1Strides, float alpha, int batch);
bool FastllmCudaAttentionBatch(Data **q, Data **k, Data **v, Data **mask, Data **output, int group, float scale, int batch);
bool FastllmCudaSplitBatch(Data &input, Data **outputs, int axis);
bool FastllmCudaCatBatch(Data **inputs, Data &output, int axis);
bool FastllmCudaMulBatch(Data **inputs, float v, int batch, Data **outputs);
bool FastllmCudaBatchMatMulTransBBatch(
    void **i0s, void **i1s, void **os, int *ns, int *ms, int *ks, int *i0Strides, int *i1Strides, float alpha, int batch);
void LaunchFastllmGemmFp16Fp16(half *input, half *weight, half *output, half *bias, int n, int m, int k);
bool FastllmCudaHalfMatMulFloat16(const Data &input, Data &weight, const Data &bias, Data &output, int n, int m, int k);
void LaunchFastllmGemmFp16Int8(half *input, uint8_t *weight, half *output, half *bias, float *scales, uint8_t *zeros, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatInt8(const Data &input, Data &weight, const Data &bias, Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloat32(const Data &input, Data &weight, const Data &bias, Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloat32(const Data &input, Data &weight, const Data &bias, Data &output, int n, int m, int k);
void LaunchFastllmGemmFp32Fp16(float *input, half *weight, float *output, float *bias, int n, int m, int k);
void LaunchFastllmGemmFp32Int8(float *input, uint8_t *weight, float *output, float *bias, float *scales, uint8_t *zeros, int n, int m, int k);
bool FastllmCudaMatMulFloatInt8(const Data &input, Data &weight, const Data &bias, Data &output, int n, int m, int k);
void LaunchFastllmGemvInt4Kernel2(float *input, uint8_t *weight, float *output, float *bias, float *scales, uint8_t *zeros, int n, int m, int k);
cublasHandle_t getFastllmCublasHandle();
