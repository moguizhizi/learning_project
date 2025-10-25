#include "fastllm-cuda.cuh"
#include <cuda_fp16.h>
#include <mma.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

using namespace nvcuda;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
#define CUDA_NO_TENSOR_CORE
#endif

typedef union __align__(16) {
    uint2 in;
    uint8_t out[8];
} union_char8;

typedef union __align__(16) {
    uint32_t in;
    uint8_t out[4];
} union_char4;

typedef union __align__(16) _union_half_4 {
    uint2 in;
    half out[4];
    half2 out2[2];
    __device__ _union_half_4() {
        // Do nothing
    }
} union_half4;

typedef union __align__(16) _union_half_8 {
    uint4 in;
    half out[8];
    half2 out2[4];
    __device__ _union_half_8() {
        // Do nothing
    }
} union_half8;

const size_t ST128_FP16_COUNT = 8;

std::map<int, std::vector<CudaMemoryBuffer>> cudaBuffersMap;
std::map<int, int> cudaBuffersMinId;
std::map<int, size_t> noBusyCnt;
std::map<int, std::vector<CudaMemoryBuffer>> bigBuffersMap;

#define checkCudaErrors(message, val) showError(val, message, __FILE__, __LINE__)

void showError(cudaError_t result, char const *const message, const char *const file, int const line) {
    if (cudaSuccess != result) {
        printf("%s\n  CUDA error = %d, %s at %s:%d\n  '%s'\n", message, result, cudaGetErrorName(result), file, line, cudaGetErrorString(result));
    }
}

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2 *>(&(pointer))[0])

TopKFunctor::TopKFunctor(float *cudaInput, float *cudaOutput, int channels, int topk) {
    this->cudaInput = cudaInput;
    this->cudaOutput = cudaOutput;
    this->channels = channels;
    this->topk = topk;
}

CudaInfos::CudaInfos() {
    int infoLen = 10;
    int *infos;
    cudaMalloc(&infos, infoLen * sizeof(int));
    GetCudaInfoKernel<<<1, 1>>>(infos);
    int *infosInCpu = new int[infoLen];
    cudaMemcpy(infosInCpu, infos, infoLen * sizeof(int), cudaMemcpyDeviceToHost);

    cudaArch = infosInCpu[0];
    hasTensorCore = cudaArch >= 700;

    cudaFree(infos);
    delete[] infosInCpu;

    printf("CUDA_ARCH: %d\n", cudaArch);
    printf("USE_TENSOR_CORE: %d\n", hasTensorCore);
}

__device__ __host__ void TopKFunctor::operator()(int i) const {
    thrust::device_ptr<float> d_input(this->cudaInput);
    thrust::device_ptr<float> d_output(this->cudaOutput);

    thrust::device_ptr<float> row_start = d_input + i * this->channels;
    thrust::device_ptr<float> row_end = row_start + this->channels;

    thrust::device_vector<int> indices(this->channels);
    thrust::sequence(indices.begin(), indices.end());

    auto begin = thrust::make_zip_iterator(thrust::make_tuple(row_start, indices.begin()));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(row_end, indices.end()));

    // 按值降序排序
    thrust::sort(begin, end, thrust::greater<thrust::tuple<float, int>>());

    // 复制前topk个结果到输出
    for (int k = 0; k < topk; ++k) {
        d_output[i * topk * 2 + k * 2] = indices[k];       // 索引
        d_output[i * topk * 2 + k * 2 + 1] = row_start[k]; // 值
    }
}

__global__ void FastllmGeluKernel(half *a, half *b, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        float x = __half2float(a[idx]);
        b[idx] = __float2half(x * 0.5f * (1.0f + erff(x / 1.41421)));
    }
}

__global__ void FastllmGeluKernel(float *a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        float x = a[idx];
        b[idx] = x * 0.5f * (1.0f + erff(x / 1.41421));
    }
}

__global__ void FastllmGeluNewKernel(float *a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        float x = a[idx];
        b[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
    }
}

__global__ void FastllmSiluKernel(float *a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        float x = a[idx];
        b[idx] = x / (1.0 + expf(-x));
    }
}

__global__ void FastllmSiluKernel(half *a, half *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
#ifdef CUDA_NO_TENSOR_CORE
        float x = __half2float(a[idx]);
        b[idx] = __float2half((x / (1.0 + expf(-x))));
#else
        half x = a[idx];
        b[idx] = __hdiv(x, __hadd(__float2half(1.0), hexp(-x)));
#endif
    }
}

__global__ void FastllmAddKernel(float *a, float *b, float v, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = a[idx] + v;
    }
}

__global__ void FastllmAddKernel(half *a, half *b, half v, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
#ifdef CUDA_NO_TENSOR_CORE
        b[idx] = __float2half(__half2float(a[idx]) + __half2float(v));
#else
        b[idx] = __hadd(a[idx], v);
#endif
    }
}

__global__ void FastllmMulKernel(float *a, float *b, float v, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = a[idx] * v;
    }
}

__global__ void FastllmMulKernel(half *a, half *b, half v, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
#ifdef CUDA_NO_TENSOR_CORE
        b[idx] = __float2half(__half2float(a[idx]) * __half2float(v));
#else
        b[idx] = __hmul(a[idx], v);
#endif
    }
}

__global__ void FastllmAddToKernel(float *a, float *b, float alpha, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        a[idx] += b[idx] * alpha;
    }
}

__global__ void FastllmAddToKernel(half *a, half *b, half alpha, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
#ifdef CUDA_NO_TENSOR_CORE
        a[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]) * __half2float(alpha));
#else
        a[idx] = __hadd(a[idx], __hmul(b[idx], alpha));
#endif
    }
}

__global__ void FastllmMulToKernel(float *a, float *b, float alpha, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        a[idx] *= b[idx] * alpha;
    }
}

__global__ void FastllmMulToKernel(half *a, half *b, float alpha, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
#ifdef CUDA_NO_TENSOR_CORE
        a[idx] = __float2half(__half2float(b[idx]) * alpha * __half2float(a[idx]));
#else
        a[idx] *= (half)((float)b[idx] * alpha);
#endif
    }
}

__global__ void FastllmCudaFloat2HalfKernel(float *a, half *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = __float2half_rz(a[idx]);
    }
}

__global__ void FastllmCudaHalf2FloatKernel(half *a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = __half2float(a[idx]);
    }
}

__global__ void FastllmCudaBF162FloatKernel(uint16_t *a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        ((uint32_t *)b)[idx] = a[idx] << 16;
    }
}

__global__ void GetCudaInfoKernel(int *infos) {
#if defined(__CUDA_ARCH__)
    infos[0] = __CUDA_ARCH__;
#else
    infos[0] = 0; // cuda arch
#endif
}

__global__ void InitBlockAtten(float *sum0, float *max0, float *sum1, float *max1, int len) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len) {
        sum0[i] = sum1[i] = 0.0f;
        max0[i] = max1[i] = -10000.0f;
    }
}

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
                                              int rotateDim) {
    int o = (blockIdx.x / n) / 2;
    int l = o / bs;
    int b = o % bs;
    int part = (blockIdx.x / n) % 2;
    int index = (int)positionIds[(b * 2 + part) * partStride + l];
    int j = threadIdx.x;

    float curSin = sin[index * sinCosStride + j];
    float curCos = cos[index * sinCosStride + j];

    int i = blockIdx.x % n;

    int id = o * spatial + i * m + part * m / 2 + j;

    float va = data[id], vb = data[id + m / 4];

    data[id] = va * curCos - vb * curSin;
    data[id + m / 4] = vb * curCos + va * curSin;
}

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
                                                    int rotateDim) {
    int o = blockIdx.x / n;
    int l = o / bs;
    int b = o % bs;
    int part = 0;
    int index = (int)positionIds[(b * 2 + 0) * partStride + l];
    int j = threadIdx.x;

    float curSin = sin[index * sinCosStride + j];
    float curCos = cos[index * sinCosStride + j];

    int i = blockIdx.x % n;

    int id = o * spatial + i * m + j * 2;

    float va = data[id], vb = data[id + 1];

    data[id] = va * curCos - vb * curSin;
    data[id + 1] = vb * curCos + va * curSin;
}

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
                                                    int rotateDim) {
    int o = (blockIdx.x / n);
    int l = o / bs;
    int b = o % bs;
    int part = 0;
    int j = threadIdx.x;
    int index = (int)(positionIds[(b * 2 + part) * partStride + l]);

    float curSin = sin[index * sinCosStride + j];
    float curCos = cos[index * sinCosStride + j];

    int i = blockIdx.x % n;
    int id = o * spatial + i * m + j * 2;

    half *d = (half *)data;
    float va = __half2float(d[id]), vb = __half2float(d[id + 1]);

    d[id] = __float2half(va * curCos - vb * curSin);
    d[id + 1] = __float2half(va * curSin + vb * curCos);
}

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
                                                   int rotateDim) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int b = o / len;
    int j = threadIdx.x;
    int index = (int)(positionIds[b * partStride + l]);

    float curSin = sin[index * sinCosStride + j];
    float curCos = cos[index * sinCosStride + j];
    int i = blockIdx.x % n;
    int id = o * spatial + i * m + j;
    float *d = (float *)data;

    float va = d[id], vb = d[id + m / 2];
    d[id] = va * curCos - vb * curSin;
    d[id + m / 2] = va * curSin + vb * curCos;
}

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
                                                   int rotateDim) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int b = o / len;
    int j = threadIdx.x;
    int index = (int)(positionIds[b * partStride + l]);

    float curSin = sin[index * sinCosStride + j];
    float curCos = cos[index * sinCosStride + j];

    int i = blockIdx.x % n;
    int id = o * spatial + i * m + j;

    half *d = (half *)data;
    float va = __half2float(d[id]), vb = __half2float(d[id + m / 2]);
    d[id] = __float2half(va * curCos - vb * curSin);
    d[id + m / 2] = __float2half(va * curSin + vb * curCos);
}

__global__ void FastllmCudaBiasKernel(half *a, half *bias, int k) {
    int stride = blockDim.x;
    for (int i = threadIdx.x; i < k; i += stride) {
#ifdef CUDA_NO_TENSOR_CORE
        a[blockIdx.x * k + i] = __float2half(__half2float(a[blockIdx.x * k + i]) + __half2float(bias[i]));
#else
        a[blockIdx.x * k + i] = __hadd(a[blockIdx.x * k + i], bias[i]);
#endif
    }
}

__global__ void FastllmCudaBiasKernel(float *a, float *bias, int k) {
    float *now = a + blockIdx.x * k;
    int stride = blockDim.x;
    for (int i = threadIdx.x; i < k; i += stride) {
        now[i] += bias[i];
    }
}

__global__ void FastllmCudaInt82HalfKernel(uint8_t *a, float *scales, uint8_t *zeros, half *b, int len, int per) {
#ifdef CUDA_NO_TENSOR_CORE
    float scalesBuffer[2];
    uint8_t zerosBuffer[2];
    int threshold = ST128_FP16_COUNT;
    int index = (threadIdx.x + blockIdx.x * blockDim.x) * ST128_FP16_COUNT;
    for (int idx = index; idx < len; idx += (gridDim.x * blockDim.x) * ST128_FP16_COUNT) {
        int startIdx = idx / per;
        int endIdx = (idx + ST128_FP16_COUNT - 1) / per;
        scalesBuffer[1] = scalesBuffer[0] = scales[startIdx];
        zerosBuffer[1] = zerosBuffer[0] = zeros[startIdx];
        if (endIdx > startIdx) {
            threshold = (idx + ST128_FP16_COUNT - 1) % per;
            scalesBuffer[1] = scales[endIdx];
            zerosBuffer[1] = zeros[endIdx];
        }
        // 读取
        union_char8 aBuffer[2];
        half bBuffer[ST128_FP16_COUNT];
        aBuffer[0].in = *reinterpret_cast<const uint2 *>(a + idx);
        // 处理
        for (int i = 0; i < ST128_FP16_COUNT; i++) {
            if (idx + i < len) {
                int scaleIdx = i < threshold ? 0 : 1;
                bBuffer[i] = __float2half(scalesBuffer[scaleIdx] * ((float)aBuffer[0].out[i] - zerosBuffer[scaleIdx]));
            }
        }
        reinterpret_cast<uint4 *>(b)[idx / ST128_FP16_COUNT] = *reinterpret_cast<uint4 *>(bBuffer);
    }
#else
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = __float2half(scales[idx / per] * ((float)a[idx] - zeros[idx / per]));
    }
#endif
}

__global__ void FastllmCudaInt42HalfKernel(uint8_t *a, float *scales, float *mins, half *b, int len, int per) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int index = idx * ST128_FP16_COUNT; index < len; index += (gridDim.x * blockDim.x) * ST128_FP16_COUNT) {
        int startId = index / per;
        int endId = (index + ST128_FP16_COUNT - 1) / per;

        float2 scaleBuffer;
        float2 minsBuffer;

        scaleBuffer.x = __ldg(scales + startId);
        scaleBuffer.y = __ldg(scales + endId);

        minsBuffer.x = __ldg(mins + startId);
        minsBuffer.y = __ldg(mins + endId);

        uint8_t *baseA = a + index / 2;

        union_char4 aBuffer;
        union_half8 bBuffer;

        aBuffer.in = *(reinterpret_cast<const uint32_t *>(baseA));

        float scale, min;
        for (int i = 0; i < ST128_FP16_COUNT / 2; i++) {
            if (index + 2 * i + 1 < len) {
                if ((index + 2 * i) / per == startId) {
                    scale = scaleBuffer.x;
                    min = minsBuffer.x;
                }

                if ((index + 2 * i) / per == endId) {
                    scale = scaleBuffer.y;
                    min = minsBuffer.y;
                }

                if ((index + 2 * i + 1) / per == startId) {
                    scale = scaleBuffer.x;
                    min = minsBuffer.x;
                }

                if ((index + 2 * i + 1) / per == endId) {
                    scale = scaleBuffer.y;
                    min = minsBuffer.y;
                }

                bBuffer.out[2 * i] = __float2half((aBuffer.out[i] >> 4) * scale + min);
                bBuffer.out[2 * i + 1] = __float2half((aBuffer.out[i] >> 0xF) * scale + min);
            }
        }

        (reinterpret_cast<uint4 *>(b))[index] = bBuffer.in;
    }
}

template <int THREAD_PER_BLOCK, typename T> __global__ void FastllmCudaFloatEmbeddingKernel(float *input, T *weight, T *output, int embSize) {
    input += blockIdx.x;
    output += blockIdx.x * embSize;
    int token = (int)(input[0] + 1e-5);
    weight += token * embSize;
    for (int i = threadIdx.x; i < embSize; i += THREAD_PER_BLOCK) {
        output[i] = weight[i];
    }
}

template <int THREADS_PER_BLOCK, typename T> __global__ void CausalMask(T *a, T maskValue, int q, int k, int base) {
    a += blockIdx.x * k;
    for (int i = base + blockIdx.x + threadIdx.x + 1; i < k; i += THREADS_PER_BLOCK) {
        a[i] = maskValue;
    }
}

template <int THREADS_PER_BLOCK> __global__ void SimpleMask(float *a, float *b, float maskValue, int spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < spatial) {
        if (b[idx] > 0.99) {
            a[idx] = maskValue;
        }
    }
}

template <int THREAD_PER_BLOCK> __global__ void SimpleMask(half *a, half *b, half maskValue, int spatial) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < spatial) {
        if (__half2float(b[i]) > 0.99) {
            a[i] = maskValue;
        }
    }
}

template <int THREAD_PER_BLOCK> __device__ void FastllmSoftmaxKernelInner1Func(float *input, float *output, int channels, float *maxp, float *sump) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float maxV;

    unsigned int tid = threadIdx.x;
    float maxValue = -1e100;
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        maxValue = max(maxValue, input[i]);
    }
    sdata[tid] = maxValue;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s])
        }
        __syncthreads();
    }

    if (tid == 0) {
        maxV = sdata[0];
        if (maxp != nullptr) {
            maxp[0] = sdata[0];
        }
    }

    __syncthreads();

    float sum = 0;
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        sum += expf(input[i] - maxV);
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (fabs(sdata[0]) < 1e-6) {
            sdata[0] = 0.0001;
        }
        if (sump != nullptr) {
            sump[0] = sdata[0];
        }
    }

    __syncthreads();

    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        output[i] = expf(input[i] - maxV) / sdata[0];
    }
}

template <int THREAD_PER_BLOCK> __device__ void FastllmSoftmaxKernelInner1Func(half *input, half *output, int channels, float *maxp, float *sump) {
    __shared__ float sdata[THREAD_PER_BLOCK];

    // 1. 每个线程计算一部分
    unsigned int tid = threadIdx.x;
    float maxValue = -1e10;
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        maxValue = max(maxValue, (float)input[i]);
    }
    sdata[tid] = maxValue;
    __syncthreads();

    // 2. 求max
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // 3. 记录max
    if (tid == 0) {
        if (maxp != nullptr) {
            sdata[0] = max(maxp[0], sdata[0]);
        }
    }
    __syncthreads();
    float maxV = sdata[0];
    __syncthreads();

    // 4. 求和
    float sum = 0;
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        sum = sum + exp((float)input[i] - maxV);
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        if (fabs(sdata[0]) < 1e-6) {
            sdata[0] = 0.0001;
        }
        if (sump != nullptr) {
            sump[0] = sump[0] * exp(maxp[0] - maxV) + sdata[0];
            sdata[0] = sump[0];
            maxp[0] = maxV;
        }
    }
    __syncthreads();

    float scale = 1.0 / sdata[0];
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        output[i] = (half)(exp((float)input[i] - maxV) * scale);
    }
}

template <int THREAD_PER_BLOCK> __global__ void FastllmSoftmaxKernelInner1(float *input, float *output, int outer, int channels) {
    int o = blockIdx.x;
    FastllmSoftmaxKernelInner1Func<THREAD_PER_BLOCK>(input + o * channels, output + o * channels, channels, nullptr, nullptr);
}

template <int THREAD_PER_BLOCK> __global__ void FastllmSoftmaxKernelInner1(half *input, half *output, int outer, int channels) {
    int o = blockIdx.x;
    FastllmSoftmaxKernelInner1Func<THREAD_PER_BLOCK>(input + o * channels, output + o * channels, channels, nullptr, nullptr);
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmSoftmaxKernelInner1(half *input, half *output, int outer, int channels, float *maxp, float *sump) {
    int o = blockIdx.x;
    FastllmSoftmaxKernelInner1Func<THREAD_PER_BLOCK>(input + o * channels, output + o * channels, channels, maxp + o, sump + o);
}

template <int THREAD_PER_BLOCK, typename T>
__global__ void FastllmSoftmaxKernelInner1WithCausalMask(T *input, T *output, int outer, int channels, int base) {
    int o = blockIdx.x;
    FastllmSoftmaxKernelInner1Func<THREAD_PER_BLOCK>(input + o * channels, output + o * channels, min(channels, o + base + 1), nullptr, nullptr);
}

template <int THREAD_PER_BLOCK, typename T>
__global__ void FastllmSoftmaxKernelInner1WithCausalMask(T *input, T *output, int outer, int channels, int base, float *maxp, float *sump) {
    int o = blockIdx.x;
    FastllmSoftmaxKernelInner1Func<THREAD_PER_BLOCK>(input + o * channels, output + o * channels, min(channels, o + base + 1), maxp + o, sump + o);
}

template <typename T, int THREAD_PER_BLOCK> __global__ void FastllmSoftmaxKernelBatchInner1(uint8_t **pointer) {
    int o = blockIdx.x;
    FastllmSoftmaxKernelInner1Func<THREAD_PER_BLOCK>(
        (T *)pointer[o * 3], (T *)pointer[o * 3 + 1], (int)((size_t)pointer[o * 3 + 2]), nullptr, nullptr);
}

template <typename T, int THREAD_PER_BLOCK> __global__ void FastllmSoftmaxKernelBatchInner1(uint8_t **pointer, int outer) {
    int o = blockIdx.x;
    int channels = (int)((size_t)pointer[o / outer * 2 + 1]);
    FastllmSoftmaxKernelInner1Func<THREAD_PER_BLOCK>(
        (T *)pointer[o / outer * 2] + (o % outer) * channels, (T *)pointer[o / outer * 2] + (o % outer) * channels, channels, nullptr, nullptr);
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmRMSNormKernelInner1(float *input, float *weight, float *output, int outer, int channels, float eps) {
    int o = blockIdx.x;
    input = input + o * channels;
    output = output + o * channels;

    __shared__ float sdata2[THREAD_PER_BLOCK];
    __shared__ float scale;

    // 1. 每个线程计算一部分
    unsigned int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        float x = input[i];
        sum += x * x;
    }
    sdata2[tid] = sum;
    __syncthreads();

    // 2. 求和
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata2[tid] = sdata2[tid] + sdata2[tid + s]
        }
        __syncthreads();
    }

    // 3. 计算参数
    if (tid == 0) {
        scale = 1.0 / sqrt(sdata2[0] / channels + eps);
    }

    __syncthreads();

    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        float x = input[i];
        output[i] = x * scale * weight[i]
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmRMSNormKernelInner1(half *input, float *weight, half *output, int outer, int channels, float eps) {
    int o = blockIdx.x;
    input = input + o * channels;
    output = output + o * channels;

    __shared__ float sdata2[THREAD_PER_BLOCK];
    __shared__ float scale;

    // 1. 每个线程计算一部分
    unsigned int tid = threadIdx.x;
    float sum2 = 0.0;
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        float x = __half2float(input[i]);
        sum2 += x * x;
    }
    sdata2[tid] = sum2;
    __syncthreads();

    // 2. 求和
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata2[tid] = sdata2[tid] + sdata2[tid + s]
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile float *now = sdata2;
        now[tid] += now[tid + 32];
        now[tid] += now[tid + 16];
        now[tid] += now[tid + 8];
        now[tid] += now[tid + 4];
        now[tid] += now[tid + 2];
        now[tid] += now[tid + 1];
    }

    __syncthreads();

    // 3. 计算参数
    if (tid == 0) {
        scale = 1.0 / sqrt(sdata2[0] / channels + eps);
    }
    __syncthreads();

    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        output[i] = __float2half(__half2float(input[i]) * scale * weight[i]);
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmLayerNormKernelInner1(float *input, float *gamma, float *beta, float *output, int outer, int channels) {
    int o = blockIdx.x;
    input = input + o * channels;
    output = output + o * channels;

    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float sdata2[THREAD_PER_BLOCK];
    __shared__ float mean;
    __shared__ float var;

    // 1. 每个线程计算一部分
    unsigned int tid = threadIdx.x;
    float sum = 0.0, sum2 = 0.0;
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        float x = input[i];
        sum += x;
        sum2 += x * x;
    }
    sdata[tid] = sum;
    sdata2[tid] = sum2;
    __syncthreads();

    // 2. 求和
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata2[tid] += sdata2[tid + s];
        }
        __syncthreads();
    }

    // 3. 计算参数
    if (tid == 0) {
        mean = sdata[0] / channels;
        var = sdata2[0] + mean * mean * channels - 2 * mean * channels * mean;
        var = sqrt(var / channels + 1e-10);
    }
    __syncthreads();

    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        output[i] = (input[i] - mean) / var * gamma[i] + beta[i];
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmLayerNormKernelInner1(half *input, float *gamma, float *beta, half *output, int outer, int channels) {
    int o = blockIdx.x;
    input = input + o * channels;
    output = output + o * channels;

    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float sdata2[THREAD_PER_BLOCK];
    __shared__ float mean;
    __shared__ float var;

    // 1. 每个线程计算一部分
    unsigned int tid = threadIdx.x;
    float sum = 0.0, sum2 = 0.0;
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        float x = __half2float(input[i]);
        sum += x;
        sum2 += x * x;
    }
    sdata[tid] = sum;
    sdata2[tid] = sum2;
    __syncthreads();

    // 2. 求和
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata2[tid] += sdata2[tid + s];
        }
        __syncthreads();
    }

    // 3. 计算参数
    if (tid == 0) {
        mean = sdata[0] / channels;
        var = sdata2[0] + mean * mean * channels - 2 * mean * channels * mean;
        var = sqrt(var / channels + 1e-10);
    }
    __syncthreads();

    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        output[i] = __float2half((__half2float(input[i]) - mean) / var * gamma[i] + beta[i]);
    }
}

template <int THREAD_PER_BLOCK> __global__ void FastllmLayerNormKernelTop1(float *input, float *output, int channels) {
    __shared__ float idData[THREAD_PER_BLOCK];
    __shared__ float maxData[THREAD_PER_BLOCK];
    float *inputData = input + blockIdx.x * channels;
    float *outputData = output + blockIdx.x * 2;
    int tid = threadIdx.x;
    idData[tid] = tid;
    maxData[tid] = -1e100;
    for (int j = tid; j < channels; j += THREAD_PER_BLOCK) {
        if (inputData[j] > maxData[tid]) {
            maxData[tid] = inputData[j];
            idData[tid] = j;
        }
    }
    __syncthreads();

    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (maxData[tid] < maxData[tid + s]) {
                maxData[tid] = maxData[tid + s];
                idData[tid] = idData[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        outputData[0] = idData[0];
        outputData[1] = maxData[0];
    }
}

template <int THREAD_PER_BLOCK, int MAXK> __global__ void FastllmLayerNormKernelTopK(float *input, float *output, int K, int channels) {
    __shared__ float idData[THREAD_PER_BLOCK][MAXK];
    __shared__ float maxData[THREAD_PER_BLOCK][MAXK];

    float *inputData = input + blockIdx.x * channels;
    float *outputData = output + blockIdx.x * K;

    unsigned int tid = threadIdx.x;
    for (int i = 0; i < K; i++) {
        maxData[tid][i] = -1e100;
    }

    for (int i = 0; i < channels; i += THREAD_PER_BLOCK) {
        float cur = inputData[i];
        for (int l = 0; l < K; l++) {
            if (cur > maxData[tid][l]) {
                int x = K - 1;
                while (x > l) {
                    maxData[tid][x] = maxData[tid][x - 1];
                    idData[tid][x] = idData[tid][x - 1];
                    x--;
                }
                maxData[tid][l] = cur;
                idData[tid][l] = i;
                break;
            }
        }
    }

    __syncthreads();

    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            int pos0 = 0, pos1 = 0;
            while (pos0 + pos1 < K) {
                if (maxData[tid][pos0] < maxData[tid + s][pos1]) {
                    pos1++;
                } else {
                    pos0++;
                }
            }
            pos0--;
            pos1--;

            int pos = K - 1;
            while (pos >= 0) {
                if ((pos1 < 0) || (pos0 >= 0 && maxData[tid][pos0] < maxData[tid + s][pos1])) {
                    maxData[tid][pos] = maxData[tid][pos0];
                    idData[tid][pos] = idData[tid][pos0];
                    pos0--;
                } else {
                    maxData[tid][pos] = maxData[tid + s][pos1];
                    idData[tid][pos] = idData[tid + s][pos1];
                    pos1--;
                }
                pos--;
            }

            __syncthreads();
        }
    }

    if (tid == 0) {
        for (int i = 0; i < k; i++) {
            output[i * 2] = idData[tid][i];
            output[i * 2 + 1] = maxData[tid][i];
        }
    }
}

template <int THREAD_PER_BLOCK> __global__ void FastllmTransposeByRowKernel(uint8_t *dst, uint8_t *ori, int n, int m, int k) {
    int row = blockIdx.x / m;
    int col = blockIdx.x % m;

    uint8_t *curInput = ori + (row * m + col) * k;
    uint8_t *curOutput = dst + (col * n + row) * k;
    for (int i = threadIdx.x, i < k; i += THREAD_PER_BLOCK) {
        curOutput[i] = curInput[i];
    }
}

template <typename T>
__global__ void FastllmPermuteKernel(T *dst,
                                     const T *src,
                                     const int *axis,            // 新 -> 旧 维度映射
                                     const uint64_t *stride_old, // 原始 stride
                                     const uint64_t *stride_new, // 新 stride
                                     int axisLen,
                                     int totalLen) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= totalLen)
        return;
    int old = 0;
    for (int i = 0; i < axisLen; i++) {
        int coord = idx / stride_new[i];
        idx = idx % stride_new[i];
        int old_dim = axis[i];
        old += coord * stride_old[old_dim];
    }

    dst[idx] = src[old];
}

template <int THREAD_PER_BLOCK> __global__ void FastllmAttentionMaskKernel(float *a, float *b, float maskValue, int n, int m, int spatial) {
    int on = blockIdx.x / m;
    int om = blockIdx.x % m;
    int o = on * m + om;

    int id = threadIdx.x;
    for (i = id; i < spatial; i += THREAD_PER_BLOCK) {
        if (b[on * spatial + i] > 0.99) {
            a[o * spatial + i] = maskValue;
        }
    }
}

template <int THREAD_PER_BLOCK> __global__ void FastllmAttentionMaskKernel(half *a, half *b, half maskValue, int n, int m, int spatial) {
    int on = blockIdx.x / m;
    int om = blockIdx.x % m;
    int o = on * m + om;
    int idx = threadIdx.x;
    for (int i = idx; i < spatial; i += THREAD_PER_BLOCK) {
        if (__half2float(b[on * spatial + i]) > 0.99) {
            a[o * spatial + i] = maskValue;
        }
    }
}

template <int BN, int BM, int BK>
__global__ void
HalfFC(half *__restrict__ a, half *__restrict__ b, half *__restrict__ c, const int N, const int M, const int K, half scale, const int base) {
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int wrap0 = wid >> 1;
    int wrap1 = wid & 1;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int stN = bx * BN;
    int stK = by * BK;

    if (base + stN + BN < stK) {
        return;
    }

    __shared__ half cur[BN][BK];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[4][8];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b[4][8];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], (half)0.0);
        }
    }

    __syncthreads();

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            wmma::load_matrix_sync(frag_a[i][j], &a[(stN + wrap0 * 64 + i * 16) * M + j * 16], M);
        }
    }

    __syncthreads();

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            wmma::load_matrix_sync(frag_b[i][j], &b[(stK + wrap1 * 64 + i * 16) * M + j * 16], M);
        }
    }

    __syncthreads();

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 8; k++) {
                wmma::mma_sync(frag_c[i][j], frag_a[i][k], frag_b[j][k], frag_c[i][j]);
            }
        }
    }

    __syncthreads();

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(cur[wrap0 * 64 + i * 16][wrap1 * 64 + j * 16], frag_c[i][j], BK, wmma::mem_row_major);
        }
    }

    __syncthreads();

    for (int i = 0; i < BN; i++) {
        if (base + stN + i < stK + tid) {
            cur[i][tid] = (half)0.0;
        }
    }

    for (int i = 0; i < BN; i++) {
        c[(stN + i) * K + stK + tid] = __hmul(cur[i][tid], scale);
    }
}

template <int THREAD_PER_BLOCK>
__global__ void AttnBlockUpdate(half *data, int n, int m, float *lastMax, float *lastSum, float *curMax, float *curSum) {
    __shared__ float scale;

    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;

    if (tid == 0) {
        float oldSum = lastSum[bid] * exp(lastMax[bid] - curMaxp[bid]);
        scale = oldSum / curSum[bid];
        lastMax[bid] = curMax[bid];
        lastSum[bid] = curSum[bid];
    }

    __syncthreads();

    for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
        data[bid * m + tid] = (half)((float)data[bid * m + tid] * scale);
    }
}

template <int THREAD_PER_BLOCK> __global__ void FastllmApplyLognAttnKernel(float *input, float *logn, float *pos, int b, int s, int spatial) {
    int bs = blockIdx.x / s;
    int seq = blockIdx.x % s;

    int offset = (bs * s + seq) * spatial;
    float currentInput = intput + offset;

    int tid = threadIdx.x;

    float v = logn[(int)pos[0] + seq];

    for (int i = tid; i < spatial; i += THREAD_PER_BLOCK) {
        currentInput[i] = currentInput[i] * v;
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmRepeatPenaltyKernel(float *input, float *penalty, float *penaltyScaleData, int tokens, int vocabs) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int input_offset = bid * vocabs;
    int penalty_offset = bid * tokens;
    float scale = penaltyScaleData[bid];

    input = input + input_offset;
    penalty = penalty + penalty_offset;

    for (int i = tid; i < tokens; i += THREAD_PER_BLOCK) {
        int token = (int)(penalty[i] + 1e-6);
        if (token >= 0) {
            input[token] = input[token] < 0 ? input[token] * scale : input[token] / scale;
        }
    }
}

template <int THREAD_PER_BLOCK> __global__ void FastllmMatMulTransBBatchKernel(uint8_t **pointer, float alpha) {
    int bid = blockIdx.x;
    float *input0 = (float *)pointer[bid * 8 + 0];
    float *input1 = (float *)pointer[bid * 8 + 1];
    float *output = (float *)pointer[bid * 8 + 2];
    int n = (int)(size_t)pointer[bid * 8 + 3];
    int m = (int)(size_t)pointer[bid * 8 + 4];
    int k = (int)(size_t)pointer[bid * 8 + 5];
    int input0stride = (int)(size_t)pointer[bid * 8 + 6];
    int input1stride = (int)(size_t)pointer[bid * 8 + 7];

    int pera = 4, perb = 4;
    int cnta = (n - 1) / pera + 1;
    int cntb = (k - 1) / perb + 1;

    float cura[4][4], curb[4][4], curc[4][4];
    for (int taskid = threadIdx.x; taskid < cnta * cntb; taskid += THREAD_PER_BLOCK) {
        int taska = taskid / pera;
        int taskb = taskid % perb;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                cura[i][j] = 0.0;
                curb[i][j] = 0.0;
                curc[i][j] = 0.0;
            }
        }

        for (int l = 0; l < m; l += 4) {
            for (int a = taska * pera; a < (taska + 1) * pera && a < n; a++) {
#pragma unroll
                for (int x = 0; x < 4; x++) {
                    if (l + x < m) {
                        cura[a - taska * pera][x] = input0[a * input0stride + l + x];
                    }
                }
            }

            for (int b = taskb * perb; b < (taskb + 1) * perb && b < k; b++) {
#pragma unroll
                for (int x = 0; x < 4; x++) {
                    if (l + x < m) {
                        curb[b - taskb * perb][x] = input1[b * input1stride + l + x];
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        curc[i][j] = cura[i][k] * curb[j][k] + cur[i][j];
                    }
                }
            }
        }

        if ((taska + 1) * pera <= n && (taskb + 1) * perb <= k) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    output[(taska * pera + i) * k + taskb * perb + j] = curc[i][j] * alpha;
                }
            }
        } else {
            for (int i = 0; taska * pera + i < n && i < pera; i++) {
                for (int j = 0; taskb * perb + j < k && j < perb; j++) {
                    output[(taska * pera + i) * k + taskb * perb + j] = curc[i][j] * alpha;
                }
            }
        }
    }
}

template <int THREAD_PER_BLOCK> __global__ void FastllmHalfMatMulTransBBatchKernel(uint8_t **pointer, float alpha) {
    int id = blockIdx.x;
    half *input0 = (half *)pointer[id * 8 + 0];
    half *input1 = (half *)pointer[id * 8 + 1];
    half *output = (half *)pointer[id * 8 + 2];
    int n = (int)((size_t)pointer[id * 8 + 3]);
    int m = (int)((size_t)pointer[id * 8 + 4]);
    int k = (int)((size_t)pointer[id * 8 + 5]);
    int input0Stride = (int)((size_t)pointer[id * 8 + 6]);
    int input1Stride = (int)((size_t)pointer[id * 8 + 7]);

    int tid = threadIdx.x;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    if (m == 128) {
        int wid = threadIdx.x >> 5;
        int perN = 8, perK = 128;

        int BN = 8, BK = 128;

        half h_alpha = __float2half(alpha);

        __shared__ float cur[BN][BK];

        for (int stN = 0; stN < n; stN += perN) {
            int endN = std::min(n, stN + perN);
            for (int stK = 0; stK < k; stK += perK) {
                int endK = std::min(n, stK + perK);
                wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a[8];
                wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> frag_b[8];
                wmma::fragment<wmma::accumulator, 8, 32, 16, float> frag_c;

                wmma::fill_fragment(frag_c, 0.0);

#pragma unroll
                for (int j = 0; j < 8; j++) {
                    wmma::load_matrix_sync(frag_a[j], &input0[(stN)*input0Stride + j * 16], input0Stride);
                }

                __syncthreads();

#pragma unroll
                for (int j = 0; j < 8; j++) {
                    wmma::load_matrix_sync(frag_b[j], &input1[(stK + wid * 32) * input1Stride + j * 16], input1Stride);
                }

                __syncthreads();

#pragma unroll
                for (int j = 0; j < 8; j++) {
                    wmma::mma_sync(frag_c, frag_a[j], frag_b[j], frag_c);
                }

                __syncthreads();

                wmma::store_matrix_sync(&cur[0][wid * 32], frag_c, BK, wmma::mem_row_major);

                __syncthreads();

                if (stk + tid < endK) {
                    for (int i = 0; stN + i < endN; i++) {
                        output[(stN + i) * k + stK + tid] = (half)cur[i][tid] * h_alpha;
                    }
                }

                __syncthreads();
            }
        }
        return;
    }

#endif
    int pera = 4, perb = 4;
    half cura[4][4], curb[4][4];
    float curc[4][4];
    int cnta = (n - 1) / pera + 1, cntb = (k - 1) / perb + 1;
    for (int taskId = tid; taskId < cnta * cntb; taskId += THREAD_PER_BLOCK) {
        int taska = taskId / cntb, taskb = taskId % cntb;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                curc[i][j] = 0.0f;
            }
        }
        for (int l = 0; l < m; l += 4) {
            for (int a = taska * pera; a < (taska + 1) * pera && a < n; a++) {
                FETCH_FLOAT2(cura[a - taska * pera]) = FETCH_FLOAT2(input0[a * input0Stride + l]);
            }
            for (int b = taskb * perb; b < (taskb + 1) * perb && b < k; b++) {
                FETCH_FLOAT2(curb[b - taskb * perb]) = FETCH_FLOAT2(input1[b * input1Stride + l]);
            }

            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        curc[i][j] += (float)cura[i][k] * (float)curb[j][k];
                    }
                }
            }
        }

        if ((taska + 1) * pera <= n && (taskb + 1) * perb <= k) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = (half)(curc[i][j] * alpha);
                }
            }
        } else {
            for (int i = 0; i < pera && taska * pera + i < n; i++) {
                for (int j = 0; j < perb && taskb * perb + j < k; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = (half)(curc[i][j] * alpha);
                }
            }
        }
    }
}

template <int THREAD_PER_BLOCK> __global__ void FastllmMatMulKernel(uint8_t **pointer, float alpha) {
    int id = blockIdx.x;
    float *input0 = (float *)pointer[id * 8 + 0];
    float *input1 = (float *)pointer[id * 8 + 1];
    float *output = (float *)pointer[id * 8 + 2];
    int n = (int)((size_t)pointer[id * 8 + 3]);
    int m = (int)((size_t)pointer[id * 8 + 4]);
    int k = (int)((size_t)pointer[id * 8 + 5]);
    int input0Stride = (int)((size_t)pointer[id * 8 + 6]);
    int input1Stride = (int)((size_t)pointer[id * 8 + 7]);

    int tid = threadIdx.x;
    int pera = 4, perb = 4;
    float cura[4][4], curb[4][4], curc[4][4];
    int cnta = (n - 1) / pera + 1, cntb = (k - 1) / perb + 1;
    for (int taskId = tid; taskId < cnta * cntb; taskId += THREAD_PER_BLOCK) {
        int taska = taskId / cntb, taskb = taskId % cntb;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                cura[i][j] = 0;
                curb[i][j] = 0;
                curc[i][j] = 0;
            }
        }

        for (int l = 0; l < m; l += 4) {
            for (int a = taska * pera; a < (taska + 1) * pera && a < n; a++) {
#pragma unroll
                for (int x = 0; x < 4; x++) {
                    cura[a - taska * pera][x] = l + x < m ? input0[a * input0Stride + l + x] : 0;
                }
            }
            for (int b = taskb * perb; b < (taskb + 1) * perb && b < k; b++) {
#pragma unroll
                for (int x = 0; x < 4; x++) {
                    curb[b - taskb * perb][x] = l + x < m ? input1[(l + x) * input1Stride + b] : 0;
                }
            }

#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        curc[i][j] += cura[i][k] * curb[j][k];
                    }
                }
            }
        }

        if ((taska + 1) * pera <= n && (taskb + 1) * perb <= k) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = curc[i][j] * alpha;
                }
            }
        } else {
            for (int i = 0; i < pera && taska * pera + i < n; i++) {
                for (int j = 0; j < perb && taskb * perb + j < k; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = curc[i][j] * alpha;
                }
            }
        }
    }

    /*
        //int tid = threadIdx.x;
        for (int i = 0; i < n; i++) {
            float *curInput0 = input0 + i * input0Stride;
            for (int j = tid; j < k; j += THREAD_PER_BLOCK) {
                float *curInput1 = input1 + j;
                float sum = 0.0;
                for (int l = 0; l < m; l++) {
                    sum += curInput0[l] * curInput1[l * input1Stride];
                }
                output[i * k + j] = sum * alpha;
            }
        }
    */
}

template <int THREAD_PER_BLOCK> __global__ void FastllmHalfMatMulKernel(uint8_t **pointer, float alpha) {
    int id = blockIdx.x;
    half *input0 = (half *)pointer[id * 8 + 0];
    half *input1 = (half *)pointer[id * 8 + 1];
    half *output = (half *)pointer[id * 8 + 2];
    int n = (int)((size_t)pointer[id * 8 + 3]);
    int m = (int)((size_t)pointer[id * 8 + 4]);
    int k = (int)((size_t)pointer[id * 8 + 5]);
    int input0Stride = (int)((size_t)pointer[id * 8 + 6]);
    int input1Stride = (int)((size_t)pointer[id * 8 + 7]);
    int tid = threadIdx.x;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    if (k == 128) {
        int wid = tid >> 5;
        int perN = 8, perM = 128;
        for (int i = 0; i < n; i++) {
            output[i * k + tid] = (half)0;
        }

        __shared__ half curA[8][128];
        __shared__ float curC[8][128];

        for (int stN = 0; stN < n; stN += perN) {
            int endN = min(stN + perN, n);
            wmma::fragment<wmma::accumulator, 8, 32, 16, float> frag_c;
            wmma::fill_fragment(frag_c, 0.0);

            for (int stM = 0; stM < m; stM += perM) {
                int endM = min(stM + perM, m);
                if (stM + tid < m) {
                    for (int i = 0; stN + i < endN; i++) {
                        curA[i][tid] = input0[(stN + i) * input0Stride + stM + tid];
                    }
                } else {
                    for (int i = 0; stN + i < endN; i++) {
                        curA[i][tid] = (half)0.0;
                    }
                }

                wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a[8];
                wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_b[8];
                __syncthreads();

#pragma unroll
                for (int j = 0; j < 8; j++) {
                    wmma::load_matrix_sync(frag_a[j], &curA[0][16 * j], 128);
                }
                __syncthreads();

#pragma unroll
                for (int j = 0; j < 8; j++) {
                    wmma::load_matrix_sync(frag_b[j], &input1[(stM + 16 * j) * input1Stride + wid * 32], input1Stride);
                }
                __syncthreads();

#pragma unroll
                for (int j = 0; j < 8; j++) {
                    wmma::mma_sync(frag_c, frag_a[j], frag_b[j], frag_c);
                }
                __syncthreads();
            }
            wmma::store_matrix_sync(&curC[0][wid * 32], frag_c, 128, wmma::mem_row_major);
            __syncthreads();

            for (int i = 0; stN + i < endN; i++) {
                output[(stN + i) * k + tid] = (half)((float)output[(stN + i) * k + tid] + (float)curC[i][tid] * alpha);
            }
            __syncthreads();
        }
        return;
    }
#endif
    int pera = 4, perb = 4;
    float cura[4][4], curb[4][4], curc[4][4];
    int cnta = (n - 1) / pera + 1, cntb = (k - 1) / perb + 1;
    for (int taskId = tid; taskId < cnta * cntb; taskId += THREAD_PER_BLOCK) {
        int taska = taskId / cntb, taskb = taskId % cntb;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                cura[i][j] = 0;
                curb[i][j] = 0;
                curc[i][j] = 0;
            }
        }

        for (int l = 0; l < m; l += 4) {
            for (int a = taska * pera; a < (taska + 1) * pera && a < n; a++) {
#pragma unroll
                for (int x = 0; x < 4; x++) {
                    cura[a - taska * pera][x] = (l + x < m ? (float)input0[a * input0Stride + l + x] : 0.f);
                }
            }
            for (int b = taskb * perb; b < (taskb + 1) * perb && b < k; b++) {
#pragma unroll
                for (int x = 0; x < 4; x++) {
                    curb[b - taskb * perb][x] = (l + x < m ? (float)input1[(l + x) * input1Stride + b] : 0.f);
                }
            }

#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        curc[i][j] += cura[i][k] * curb[j][k];
                    }
                }
            }
        }

        if ((taska + 1) * pera <= n && (taskb + 1) * perb <= k) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = (half)(curc[i][j] * alpha);
                }
            }
        } else {
            for (int i = 0; i < pera && taska * pera + i < n; i++) {
                for (int j = 0; j < perb && taskb * perb + j < k; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = (half)(curc[i][j] * alpha);
                }
            }
        }
    }
    /*
        for (int i = 0; i < n; i++) {
            half *curInput0 = input0 + i * input0Stride;
            for (int j = tid; j < k; j += THREAD_PER_BLOCK) {
                half *curInput1 = input1 + j;
                float sum = 0.0;
                for (int l = 0; l < m; l++) {
                    sum += (float)curInput0[l] * (float)curInput1[l * input1Stride];
                }
                output[i * k + j] = (half)(sum * alpha);
            }
        }
    */
}

template <typename T> bool DoFastllmCudaAttentionBatch(Data **q, Data **k, Data **v, Data **mask, Data **output, int group, float scale, int batch) {
    int k0 = k[0]->dims[0];

    size_t memSum = 0;
    for (int b = 0; b < batch; b++) {
        memSum += q[b]->dims[0] * q[b]->dims[1] * k[b]->dims[1];
    }

    T *mem = (T *)FastllmCudaMalloc(memSum * sizeof(T));
    memSum = 0;
    T **qk = new T *[batch];
    for (int b = 0; b < batch; b++) {
        qk[b] = mem + memSum;
        memSum += q[b]->dims[0] * q[b]->dims[1] * k[b]->dims[1];
    }

    uint8_t **cpuPointers = new uint8_t *[batch * k0 * 8];
    uint8_t **pointers = (uint8_t **)FastllmCudaMalloc(batch * k0 * 8 * sizeof(uint8_t *));
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < k0; i++) {
            cpuPointers[b * k0 + i + 0] = (uint8_t *)(q[b]->cudaData + i * group * q[b]->dims[1] * q[b]->dims[2] * sizeof(T));
            cpuPointers[b * k0 + i + 1] = (uint8_t *)(k[b]->cudaData + i * k[b]->dims[1] * k[b]->dims[2] * sizeof(T));
            cpuPointers[b * k0 + i + 2] = (uint8_t *)(qk[b] + i * group * q[b]->dims[1] * k[b]->dims[1] * sizeof(T));
            cpuPointers[b * k0 + i + 3] = (uint8_t *)((size_t)group * q[b]->dims[1]);
            cpuPointers[b * k0 + i + 4] = (uint8_t *)((size_t)q[b]->dims[2]);
            cpuPointers[b * k0 + i + 5] = (uint8_t *)((size_t)k[b]->dims[1]);
            cpuPointers[b * k0 + i + 6] = (uint8_t *)((size_t)q[b]->strides[1]);
            cpuPointers[b * k0 + i + 7] = (uint8_t *)((size_t)k[b]->strides[1]);
        }
    }
    cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t *) * batch * k0 * 8, cudaMemcpyKind::cudaMemcpyHostToDevice);
    if (typeid(T) == typeid(half)) {
        FastllmHalfMatMulTransBBatchKernel<128><<<batch * k0, 128>>>(pointers, scale);
    } else {
        FastllmMatMulTransBBatchKernel<128><<<batch * k0, 128>>>(pointers, scale);
    }

    int channels = 0;
    int maxChannels = -1;
    int outer = q[0]->dims[0] * q[0]->dims[1];
    for (int b = 0; b < batch; b++) {
        channels = k[b]->dims[1];
        cpuPointers[b * 2 + 0] = (uint8_t *)qk[b];
        cpuPointers[b * 2 + 1] = (uint8_t *)((size_t)channels);
        maxChannels = std::max(maxChannels, channels)
    }

    cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t *) * batch * 2, cudaMemcpyKind::cudaMemcpyHostToDevice);
    if (maxChannels < 128) {
        FastllmSoftmaxKernelBatchInner1<T, 32><<<batch * outer, 32>>>(pointers, outer);
    } else if (maxChannels < 512) {
        FastllmSoftmaxKernelBatchInner1<T, 64><<<batch * outer, 64>>>(pointers, outer);
    } else {
        FastllmSoftmaxKernelBatchInner1<T, 128><<<batch * outer, 128>>>(pointers, outer);
    }

    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < k0; i++) {
            cpuPointers[(b * k0 + i) * 8 + 0] = (uint8_t *)qk[b] + i * group * q[b]->dims[1] * k[b]->dims[1] * sizeof(T);
            cpuPointers[(b * k0 + i) * 8 + 1] = (uint8_t *)v[b]->cudaData + i * v[b]->strides[0] * sizeof(T);
            cpuPointers[(b * k0 + i) * 8 + 2] = (uint8_t *)output[b]->cudaData + i * group * q[b]->dims[1] * v[b]->dims[2] * sizeof(T);
            cpuPointers[(b * k0 + i) * 8 + 3] = (uint8_t *)(size_t)(group * q[b]->dims[1]);
            cpuPointers[(b * k0 + i) * 8 + 4] = (uint8_t *)(size_t)k[b]->dims[1];
            cpuPointers[(b * k0 + i) * 8 + 5] = (uint8_t *)(size_t)v[b]->dims[2];
            cpuPointers[(b * k0 + i) * 8 + 6] = (uint8_t *)(size_t)k[b]->dims[1];
            cpuPointers[(b * k0 + i) * 8 + 7] = (uint8_t *)(size_t)v[b]->strides[1];
        }
    }
    cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t *) * batch * k0 * 8, cudaMemcpyHostToDevice);

    if (typeid(T) == typeid(half)) {
        FastllmHalfMatMulKernel<128><<<batch * k0, 128>>>(pointers, 1.0f);
    } else {
        FastllmMatMulKernel<128><<<batch * k0, 128>>>(pointers, 1.0f);
    }

    FastllmCudaFree(pointers);
    delete[] cpuPointers;

    FastllmCudaFree(mem);
    delete[] qk;

    DeviceSync();
    return true;
}

template <int THREAD_PER_BLOCK> __global__ void FastllmSplitBatchKernel(uint8_t *inputs, uint8_t **outputs, int outer, int channels, int inner) {
    int outerid = blockIdx.x / channels;
    int partid = blockIdx.x % channels;

    int input_offset = outerid * channels * inner + partid * inner;

    uint8_t *input = inputs + input_offset;
    uint8_t *output = outputs[partid] + outerid * inner;
    for (int i = threadIdx; i < inner; i += THREAD_PER_BLOCK) {
        output[i] = input[i];
    }
}

template <int THREAD_PER_BLOCK> __global__ void FastllmCatBatchKernel(uint8_t **inputs, uint8_t *output, int outer, int channels, int inner) {
    int oid = blockIdx.x / channels;
    int partid = blockIdx.x % channels;

    uint8_t *input = inputs[partid] + oid * inner;
    uint8_t *output = output + oid * channels * inner + partid * inner;

    for (int i = threadIdx.x; i < inner; i += THREAD_PER_BLOCK) {
        output[i] = input[i];
    }
}

template <int THREAD_PER_BLOCK> __global__ void FastllmMulBatchKernel(float **pointer, int batch, float v) {
    float *input = pointer[blockIdx.x];
    float *output = pointer[blockIdx.x + batch];
    int len = (int)((unsigned long long)pointer[blockIdx.x + batch * 2]);
    for (int i = threadIdx.x; i < len; i += THREAD_PER_BLOCK) {
        output[i] = input[i] * v;
    }
}

template <int THREAD_PER_BLOCK, int PART> __global__ void FastllmGemvFp16Fp16Kernel2MultiRow(half *A, half *B, half *C, half *bias, int m, int k) {
    int st = blockIdx.x;
    int p = st;
    int tid = threadIdx.x;
    __shared__ float sdata[PART][THREAD_PER_BLOCK];

#pragma unroll
    for (int x = 0; x < PART; x++) {
        sdata[x][tid] = 0.0f;
    }

    union_half8 regA;
    union_half8 regB;
    if (m % 8 == 0) {
        for (int i = tid * 8; i < m; i += THREAD_PER_BLOCK * 8) {
            for (int x = 0; x < PART; x++) {
                float sum = 0.0f;
                regA.in = *reinterpret_cast<const uint4 *> A[x * m + i];
                regB.in = *reinterpret_cast<const uint4 *> B[p * m + i];

                if (i < m) {
                    sum += __low2float(regA.out2[0]) * __low2float(regB.out2[0])
                }

                if (i + 1 < m) {
                    sum += __high2float(regA.out2[0]) * __high2float(regB.out2[0])
                }

                if (i + 2 < m) {
                    sum += __low2float(regA.out2[1]) * __low2float(regB.out2[1])
                }

                if (i + 3 < m) {
                    sum += __high2float(regA.out2[1]) * __high2float(regB.out2[1])
                }

                if (i + 4 < m) {
                    sum += __low2float(regA.out2[2]) * __low2float(regB.out2[2])
                }

                if (i + 5 < m) {
                    sum += __high2float(regA.out2[2]) * __high2float(regB.out2[2])
                }

                if (i + 6 < m) {
                    sum += __low2float(regA.out2[3]) * __low2float(regB.out2[3])
                }

                if (i + 7 < m) {
                    sum += __high2float(regA.out2[3]) * __high2float(regB.out2[3])
                }

                sdata[x][tid] += sum;
            }
        }
    } else {
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                sdata[x][tid] += __half2float(A[x * m + i]) * __half2float(B[p * m + i]);
            }
        }
    }

    __syncthreads();

    float diff = 0.0f;

    for (int s = THREAD_PER_BLOCK / 2; s >= 0; s >> 1) {
#pragma unroll
        for (int x = 0; i < PART; x++) {
            if (tid < s) {
                float other = sdata[x][s + tid] - diff;
                float sumTmp = sdata[x][tid] + other;
                diff = sumTmp - sdata[x][tid] - other;
                sdata[x][tid] = sumTmp;
            }
        }
    }

    __syncthreads();

    if (tid == 0) {
        if (bias != nullptr) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                C[x * k + p] = __float2half(sdata[x][tid] + __half2float(__ldg(bias + p)));
            }

        } else {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                C[x * k + p] = __float2half(sdata[x][tid]);
            }
        }
    }
    __syncthreads();
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFp16Int8Kernel2(half *A, uint8_t *B, half *C, half *bias, float *scales, uint8_t *zeros, int m, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];

    int st = blockIdx.x * PART;
    int end = st + PART;

    int tid = threadIdx.x;

    union_half8 regA;
    union_char8 regB;

    for (int p = st; p < end; p++) {
        sdata[tid] = 0;

        uint8_t *baseB = B + p * m;

        uint8_t zero = zeros[p];

        for (int i = tid * ST128_FP16_COUNT; i < m; i += THREAD_PER_BLOCK * ST128_FP16_COUNT) {
            regA.in = *reinterpret_cast<const uint4 *>(A + i);
            regB.in = *reinterpret_cast<const uint2 *>(baseB + i);

            float sum = 0.0;

            if (i < m) {
                sum += __low2float(regA.out2[0]) * (float)(regB.out[0] - zero);
            }

            if (i + 1 < m) {
                sum += __high2float(regA.out2[0]) * (float)(regB.out[1] - zero);
            }

            if (i + 2 < m) {
                sum += __low2float(regA.out2[1]) * (float)(regB.out[2] - zero);
            }

            if (i + 3 < m) {
                sum += __high2float(regA.out2[1]) * (float)(regB.out[3] - zero);
            }

            if (i + 4 < m) {
                sum += __low2float(regA.out2[2]) * (float)(regB.out[4] - zero);
            }

            if (i + 5 < m) {
                sum += __high2float(regA.out2[2]) * (float)(regB.out[5] - zero);
            }

            if (i + 6 < m) {
                sum += __low2float(regA.out2[3]) * (float)(regB.out[6] - zero);
            }

            if (i + 7 < m) {
                sum += __high2float(regA.out2[3]) * (float)(regB.out[7] - zero);
            }

            sdata[tid] += sum;
        }

        __syncthreads();

        float diff = 0.0f;

        for (int s = THREAD_PER_BLOCK / 2; s >= 0; s >> 1) {
            if (tid < s) {
                float other = sdata[s + tid] - diff;
                float sumTmp = sdata[tid] + other;
                diff = sumTmp - sdata[tid] - other;
                sdata[tid] = sumTmp;
            }
        }

        __syncthreads();

        if (tid == 0) {
            if (bias != nullptr) {
#pragma unroll
                C[p] = __float2half(sdata[tid] * __ldg(scales + p) + __half2float(__ldg(bias + p)));

            } else {
#pragma unroll
                C[p] = __float2half(sdata[tid] * __ldg(scales + p));
            }
        }
        __syncthreads();
    }
}

template <int THREAD_PER_BLOCK, int PART> __global__ void FastllmGemvFp32Fp32Kernel2(float *A, float *B, float *C, float *bias, int m, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
    for (int p = st; p < end; p++) {
        sdata[tid] = 0;
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
            sdata[tid] += A[i] * B[p * m + i];
        }
        __syncthreads();
        for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
            if ((tid & (2 * s - 1)) == 0) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            C[p] = sdata[0] + bias[p];
        }
        __syncthreads();
    }
}

template <int THREAD_PER_BLOCK, int PART> __global__ void FastllmGemvFp32Fp16Kernel2MultiRow(float *A, half *B, float *C, float *bias, int m, int k) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    const half zero = __float2half_rn(0.0);
    float4 regA;
    union_half4 regB;

    // 1. 计算
    int st = blockIdx.x;
    int p = st;
#pragma unroll
    for (int x = 0; x < PART; x++)
        sdata[x][tid] = 0;

    const half *baseB = B + p * m;
    if (m % 4 == 0) {
#pragma unroll
        for (int i = tid * 4; i + 3 < m; i += THREAD_PER_BLOCK * 4) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                regA = FETCH_FLOAT4(A[i + x * m]);
                regB.in = *reinterpret_cast<const uint2 *>(baseB + i);
                float sum = 0.0f;
                if (i < m)
                    sum += regA.x * __low2float(regB.out2[0]);
                if (i + 1 < m)
                    sum += regA.y * __high2float(regB.out2[0]);
                if (i + 2 < m)
                    sum += regA.z * __low2float(regB.out2[1]);
                if (i + 3 < m)
                    sum += regA.w * __high2float(regB.out2[1]);
                sdata[x][tid] += sum;
            }
        }
    } else {
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                sdata[x][tid] += A[i + x * m] * (float)baseB[i];
            }
        }
    }
    __syncthreads();
    float diff = 0.0f;
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                float other = sdata[x][tid + s] - diff;
                float sumTmp = sdata[x][tid] + other;
                diff = (sumTmp - sdata[x][tid]) - other;
                sdata[x][tid] = sumTmp;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (bias == nullptr) {
            for (int x = 0; x < PART; x++)
                C[p + k * x] = sdata[x][0];
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++)
                C[p + k * x] = sdata[x][0] + __ldg(bias + p);
        }
    }
    __syncthreads();
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvInt8Kernel2(float *A, uint8_t *B, float *C, float *bias, float *scales, uint8_t *zeros, int m, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 读入fdata
    /*for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
        fdata[i] = A[i];
    }
    __syncthreads();*/

    float4 regA;
    union_char4 regB;

    // 2. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
    for (int p = st; p < end; p++) {
        sdata[tid] = 0;
        uint8_t zero = zeros[p];
        const uint8_t *baseB = B + p * m;
#ifdef CUDA_NO_TENSOR_CORE
#pragma unroll
        for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
            regA = FETCH_FLOAT4(A[i]);
            regB.in = *reinterpret_cast<const uint32_t *>(baseB + i);
            float sum = 0.0f;
            if (i < m)
                sum += regA.x * (float)(regB.out[0] - zero);
            if (i + 1 < m)
                sum += regA.y * (float)(regB.out[1] - zero);
            if (i + 2 < m)
                sum += regA.z * (float)(regB.out[2] - zero);
            if (i + 3 < m)
                sum += regA.w * (float)(regB.out[3] - zero);
            sdata[tid] += sum;
        }
#else
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
            sdata[tid] += A[i] * (B[p * m + i] - zero);
        }
#endif
        __syncthreads();
        float diff = 0.0f;
        for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) {
                float other = sdata[tid + s] - diff;
                float sumTmp = sdata[tid] + other;
                diff = (sumTmp - sdata[tid]) - other;
                sdata[tid] = sumTmp;
            }
            __syncthreads();
        }

        if (tid == 0) {
            C[p] = sdata[0] * __ldg(scales + p) + __ldg(bias + p);
        }
        __syncthreads();
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvInt4Kernel2(float *A, uint8_t *B, float *C, float *bias, float *scales, uint8_t *zeros, int m, int k) {
    unsigned int tid = threadIdx.x;

    int st = blockIdx.x * PART;
    int end = st + PART;

    __shared__ float sdata[THREAD_PER_BLOCK];

    for (int p = st; p < end; p++) {
        sdata[tid] = 0;
        for (int i = tid; i < m / 2; i++) {
            uint8_t now = B[p * m / 2 + i];
            sdata[tid] += (A[2 * i] * (now >> 4 - __ldg(zeros + p)) + A[2 * i + 1] * (now & 0xF - __ldg(zeros + p)))
        }

        __syncthreads();

        float diff = 0.0f;
        for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) {
                float other = sdata[tid + s] - diff;
                float sumTmp = sdata[tid] + other;
                diff = (sumTmp - sdata[tid]) - other;
                sdata[tid] = sumTmp;
            }
            __syncthreads();
        }

        if (tid == 0) {
            C[p] = sdata[0] * __ldg(scales + p) + __ldg(bias + p);
        }
        __syncthreads();
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvInt4NoZeroKernel1MultiRow(float *A, uint8_t *B, float *C, float *bias, float *scales, float *mins, int m, int k) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];

    unsigned int tid = threadIdx.x;
    int p = blockIdx.x;

#pragma unroll
    for (int i = 0; i < PART; i++) {
        sdata[i][tid] = 0;
    }

    uint8_t *baseB = B + p * (m / 2);

    float4 regA;
    uint16_t regB;

    float minv = __ldg(mins + p) / __ldg(scales + p);

    for (int i = tid * 2; i < m; i += THREAD_PER_BLOCK * 2) {
#pragma unroll
        for (int x = 0; x < PART; x++) {
            regA = FETCH_FLOAT4(A[x * m + 2 * i]);
            regB = *reinterpret_cast<const uint16_t *>(baseB + i);

            sdata[x][tid] += regA.x * (minv + (regB >> 4) & 0xF) + regA.y * (minv + regB & 0xF) + regA.z * (minv + regB >> 12) +
                             regA.w * (minv + (regB >> 8) & 0xF);
        }
    }

    __syncthreads();

    float diff = 0.0f;
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                float other = sdata[x][tid + s] - diff;
                float sumTmp = sdata[x][tid] + other;
                diff = (sumTmp - sdata[x][tid]) - other;
                sdata[x][tid] = sumTmp;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (bias == nullptr) {
            for (int x = 0; x < PART; x++)
                C[p + k * x] = sdata[x][0] * scales[p];
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++)
                C[p + k * x] = sdata[x][0] * scales[p] + bias[p];
        }
    }
    __syncthreads();
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvInt4NoZeroKernel1(float *A, uint8_t *B, float *C, float *bias, float *scales, float *mins, int m, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
    for (int p = st; p < end; p++) {
        sdata[tid] = 0;
        const uint8_t *baseB = B + p * m / 2;
        float minv = __ldg(mins + p) / __ldg(scales + p);
        for (int i = tid * 2; i < m / 2; i += THREAD_PER_BLOCK * 2) {
            float4 aBuffer = FETCH_FLOAT4(A[i * 2]);
            uint16_t bBuffer = *reinterpret_cast<const uint16_t *>(baseB + i);
            sdata[tid] += aBuffer.x * (minv + ((bBuffer >> 4) & 15)) + aBuffer.y * (minv + (bBuffer & 15));
            sdata[tid] += aBuffer.z * (minv + (bBuffer >> 12)) + aBuffer.w * (minv + ((bBuffer >> 8) & 15));
        }
        __syncthreads();

        float diff = 0.0f;
        for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) {
                float other = sdata[tid + s] - diff;
                float sumTmp = sdata[tid] + other;
                diff = (sumTmp - sdata[tid]) - other;
                sdata[tid] = sumTmp;
            }
            __syncthreads();
        }
        // if (tid <= 32)
        // warpReduce(sdata, tid);
        if (tid == 0) {
            if (bias == nullptr) {
                C[p] = sdata[0] * scales[p];
            } else {
                C[p] = sdata[0] * scales[p] + bias[p];
            }
        }
        __syncthreads();
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFp16Int4NoZeroKernel1MultiRow(half *A, uint8_t *B, half *C, half *bias, float *scales, float *mins, int m, int k) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 计算
    int st = blockIdx.x;
    int p = st;
#pragma unroll
    for (int x = 0; x < PART; x++)
        sdata[x][tid] = 0;

    union_char4 bBuffer;
    float minv = __ldg(mins + p) / __ldg(scales + p);

    for (int i = tid; i < m / 8; i += THREAD_PER_BLOCK) {
        bBuffer.in = *reinterpret_cast<const uint32_t *>(B + st * m / 2 + i * 4);
        // uint8_t now0 = B[st * m / 2 + i * 4];
        // uint8_t now1 = B[st * m / 2 + i * 4 + 1];
        // uint8_t now2 = B[st * m / 2 + i * 4 + 2];
        // uint8_t now3 = B[st * m / 2 + i * 4 + 3];
        for (int x = 0; x < PART; x++) {
            union_half8 aBuffer;
            aBuffer.in = *reinterpret_cast<const uint4 *>(A + x * m + i * 8);
            sdata[x][tid] +=
                (__low2float(aBuffer.out2[0]) * (minv + (bBuffer.out[0] >> 4)) + __high2float(aBuffer.out2[0]) * (minv + (bBuffer.out[0] & 15)));
            sdata[x][tid] +=
                (__low2float(aBuffer.out2[1]) * (minv + (bBuffer.out[1] >> 4)) + __high2float(aBuffer.out2[1]) * (minv + (bBuffer.out[1] & 15)));
            sdata[x][tid] +=
                (__low2float(aBuffer.out2[2]) * (minv + (bBuffer.out[2] >> 4)) + __high2float(aBuffer.out2[2]) * (minv + (bBuffer.out[2] & 15)));
            sdata[x][tid] +=
                (__low2float(aBuffer.out2[3]) * (minv + (bBuffer.out[3] >> 4)) + __high2float(aBuffer.out2[3]) * (minv + (bBuffer.out[3] & 15)));
        }
    }
    __syncthreads();

    float diff = 0.0f;
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                float other = sdata[x][tid + s] - diff;
                float sumTmp = sdata[x][tid] + other;
                diff = (sumTmp - sdata[x][tid]) - other;
                sdata[x][tid] = sumTmp;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (bias == nullptr) {
#pragma unroll
            for (int x = 0; x < PART; x++)
                C[p + k * x] = (half)(sdata[x][0] * scales[p]);
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++)
                C[p + k * x] = (half)(sdata[x][0] * scales[p] + float(bias[p]));
        }
    }
    __syncthreads();
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFp16Int4NoZeroKernel2(half *A, uint8_t *B, half *C, half *bias, float *scales, float *mins, int m, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
    for (int p = st; p < end; p++) {
        sdata[tid] = 0;
        float minv = mins[p] / scales[p];
        for (int i = tid; i < m / 2; i += THREAD_PER_BLOCK) {
            uint8_t now = B[p * m / 2 + i];
            sdata[tid] += ((float)A[i * 2] * (minv + (now >> 4)) + (float)A[i * 2 + 1] * (minv + (now & 15)));
        }
        __syncthreads();
        for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
            if ((tid & (2 * s - 1)) == 0) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            if (bias == nullptr) {
                C[p] = (half)(sdata[0] * scales[p]);
            } else {
                C[p] = (half)(sdata[0] * scales[p] + (float)bias[p]);
            }
        }
        __syncthreads();
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void
FastllmGemvHalfInt4GroupKernelMultiRow(half *A, uint8_t *B, half *C, half *bias, half *scales, half *mins, int m, int k, int group, int groupCnt) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];

    unsigned int tid = threadIdx.x;
    int st = blockIdx.x;
    int baseGroup = st * group;

#pragma unroll
    for (int x = 0; x < PART; x++) {
        sdata[x][tid] = 0;
    }

    uint8_t *baseB = B + st * (m / 2);
    union_half8 aBuffer;
    union_char4 bBuffer;

    int g = 0;

    for (int i = tid; i < m / 8; i += THREAD_PER_BLOCK) {
        bBuffer.in = *(reinterpret_cast<const uint32_t *>(baseB + i * 4));
        g = baseGroup + (i * 8) / groupCnt;
        float curScale = __half2float(__ldg(scales + g));
        float curMin = __half2float(__ldg(mins + g));
#pragma unroll
        for (int x = 0; x < PART; x++) {
            aBuffer.in = *(reinterpret_cast<const uint4 *>(A + x * m + i * 8));

            sdata[x][tid] += __half2float(aBuffer.out[0]) * (curScale * bBuffer.out[0] >> 4 + curMin);
            sdata[x][tid] += __half2float(aBuffer.out[1]) * (curScale * bBuffer.out[0] & 0xF + curMin);
            sdata[x][tid] += __half2float(aBuffer.out[2]) * (curScale * bBuffer.out[1] >> 4 + curMin);
            sdata[x][tid] += __half2float(aBuffer.out[3]) * (curScale * bBuffer.out[1] & 0xF + curMin);
            sdata[x][tid] += __half2float(aBuffer.out[4]) * (curScale * bBuffer.out[2] >> 4 + curMin);
            sdata[x][tid] += __half2float(aBuffer.out[5]) * (curScale * bBuffer.out[2] & 0xF + curMin);
            sdata[x][tid] += __half2float(aBuffer.out[6]) * (curScale * bBuffer.out[3] >> 4 + curMin);
            sdata[x][tid] += __half2float(aBuffer.out[7]) * (curScale * bBuffer.out[3] & 0xF + curMin);
        }
    }

    for (int x = 0; x < PART; x++) {
        float val = sdata[x][tid];
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }

        if ((tid % warpSize) == 0) {
            sdata[x][tid / warpSize] = val;
        }

        __syncthreads(); // 等所有 warp 写完部分和
    }

    if (tid < warpSize) {
        for (x = 0; x < PART; x++) {
            float val = tid < (THREAD_PER_BLOCK / warpSize) ? sdata[x][tid] : 0.0;
            for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
                val += __shfl_down_sync(0xFFFFFFFF, val, offset);
            }

            if (tid == 0) {
                sdata[x][tid] = val;
            }

            __syncthreads();
        }
    }

    if (tid == 0) {
        if (bias != nullptr) {
#pragma unroll
            for (int x = 0; x < PART; x++)
                C[st + k * x] = __float2half(sdata[x][0] + __half2float(__ldg(bias + st)));
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++)
                C[st + k * x] = __float2half(sdata[x][0]);
        }
    }
    __syncthreads();
}

__global__ void FastllmCudaInt4Group2HalfKernel(uint8_t *a, half *scales, half *mins, half *b, int k, int m, int group, int groupCnt) {
    unsigned int tid = threadIdx.x;
    unsigned int st = blockIdx.x;
    half2 scalesBuffer;
    half2 minBuffer;
    int threshold = ST128_FP16_COUNT;
    for (int i = tid * ST128_FP16_COUNT; i < m; i += blockDim.x * ST128_FP16_COUNT) {
        int index = st * m + i;
        int startIdx = st * group + i / groupCnt;
        int endIdx = st * group + (i + ST128_FP16_COUNT - 1) / groupCnt;
        scalesBuffer.x = scalesBuffer.y = __ldg(scales + startIdx);
        minBuffer.x = minBuffer.y = __ldg(mins + startIdx);
        if (endIdx > startIdx) {
            threshold = (i + ST128_FP16_COUNT - 1) % groupCnt;
            scalesBuffer.y = __ldg(scales + endIdx);
            minBuffer.y = __ldg(mins + endIdx);
        }
        // 读取
        union_char4 aBuffer;
        union_half8 bBuffer;
        aBuffer.in = *reinterpret_cast<const uint32_t *>(a + index / 2);
        // 处理
        for (int j = 0; j < ST128_FP16_COUNT / 2; j++) {
            if (i + j * 2 + 1 < m) {
                float scale = __half2float(j * 2 < threshold ? scalesBuffer.x : scalesBuffer.y);
                float min = __half2float(j * 2 < threshold ? minBuffer.x : minBuffer.y);
                bBuffer.out[j * 2] = __float2half(scale * (aBuffer.out[j] >> 4) + min);
                bBuffer.out[j * 2 + 1] = __float2half(scale * (aBuffer.out[j] & 0xF) + min);
            }
        }
        reinterpret_cast<uint4 *>(b)[index / ST128_FP16_COUNT] = bBuffer.in;
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void
FastllmGemvInt4GroupKernel3(float *A, uint8_t *B, float *C, float *bias, half *scales, half *mins, int m, int k, int group, int groupCnt) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    // 1. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
#pragma unroll
    for (int p = 0; p < PART; p++) {
        sdata[p][tid] = 0;
    }

    for (int i = tid * 2; i < m / 2; i += THREAD_PER_BLOCK * 2) {
        float4 aBuffer = FETCH_FLOAT4(A[i * 2]);

        for (int p = st; p < end; p++) {
            uint16_t bBuffer = *reinterpret_cast<const uint16_t *>(B + p * m / 2 + i);
            int g = p * group + (i * 2 / groupCnt);
            float curmin = __half2float(__ldg(mins + g)), curscale = __half2float(__ldg(scales + g));
            sdata[p - st][tid] +=
                aBuffer.x * (curmin + curscale * (float)((bBuffer >> 4) & 15)) + aBuffer.y * (curmin + curscale * (float)(bBuffer & 15));
            sdata[p - st][tid] +=
                aBuffer.z * (curmin + curscale * (float)(bBuffer >> 12)) + aBuffer.w * (curmin + curscale * (float)((bBuffer >> 8) & 15));
        }
    }
    __syncthreads();
    for (int p = 0; p < PART; p++) {
        for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
            if ((tid & (2 * s - 1)) == 0) {
                sdata[p][tid] += sdata[p][tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            C[st + p] = sdata[p][0] + bias[st + p];
        }
        __syncthreads();
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void
FastllmGemvInt4GroupKernel2(float *A, uint8_t *B, float *C, float *bias, half *scales, half *mins, int m, int k, int group, int groupCnt) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    // 1. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
#pragma unroll
    for (int p = 0; p < PART; p++) {
        sdata[p][tid] = 0;
    }

    for (int i = tid; i < m / 8; i += THREAD_PER_BLOCK) {
        float4 aBuffer = FETCH_FLOAT4(A[i * 8]);
        float4 bBuffer = FETCH_FLOAT4(A[i * 8 + 4]);

        for (int p = st; p < end; p++) {
            uint8_t now0 = B[p * m / 2 + i * 4];
            uint8_t now1 = B[p * m / 2 + i * 4 + 1];
            uint8_t now2 = B[p * m / 2 + i * 4 + 2];
            uint8_t now3 = B[p * m / 2 + i * 4 + 3];
            int g = p * group + (i * 8 / groupCnt);
            float curmin = (float)mins[g], curscale = (float)scales[g];
            sdata[p - st][tid] += (aBuffer.x * (curmin + (float)curscale * (now0 >> 4)) + aBuffer.y * (curmin + (float)curscale * (now0 & 15)));
            sdata[p - st][tid] += (aBuffer.z * (curmin + (float)curscale * (now1 >> 4)) + aBuffer.w * (curmin + (float)curscale * (now1 & 15)));
            sdata[p - st][tid] += (bBuffer.x * (curmin + (float)curscale * (now2 >> 4)) + bBuffer.y * (curmin + (float)curscale * (now2 & 15)));
            sdata[p - st][tid] += (bBuffer.z * (curmin + (float)curscale * (now3 >> 4)) + bBuffer.w * (curmin + (float)curscale * (now3 & 15)));
        }
    }
    __syncthreads();
    for (int p = 0; p < PART; p++) {
        for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
            if ((tid & (2 * s - 1)) == 0) {
                sdata[p][tid] += sdata[p][tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            C[st + p] = sdata[p][0] + bias[st + p];
        }
        __syncthreads();
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFP8E4M3Kernel1MultiRow(float *A, uint8_t *B, float *C, float *bias, float *scales, int m, int k, int blockM, int blockK) {

    __shared__ float sdata[PART][THREAD_PER_BLOCK];

    unsigned int tid = threadIdx.x;
    int st = blockIdx.x;
    int ms = (m - 1) / blockM + 1;
    const float magicScaleConstant = exp2f(120.0f);

    uint8_t *baseB = B + st * m;
    float *baseScale = scales + (st / blockK) * ms;

#pragma unroll
    for (int x = 0; x < PART; x++) {
        sdata[x][tid] = 0.0;
    }

    union_char4 bBuffer;

    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
        float curScale = *(baseScale + (i / blockM));
        bBuffer.in = *reinterpret_cast<const uint32_t *>(baseB + i);

        for (int x = 0; x < PART; x++) {
            float4 aBuffer = FETCH_FLOAT4(A[x * m + i]);
            sdata[x][i] += aBuffer.x * (__uint_as_float(((bBuffer.out[0] & 0x80) << 24) | ((bBuffer.out[0] & 0x7F) << 20)));
            sdata[x][i] += aBuffer.y * (__uint_as_float(((bBuffer.out[1] & 0x80) << 24) | ((bBuffer.out[1] & 0x7F) << 20)));
            sdata[x][i] += aBuffer.z * (__uint_as_float(((bBuffer.out[2] & 0x80) << 24) | ((bBuffer.out[2] & 0x7F) << 20)));
            sdata[x][i] += aBuffer.w * (__uint_as_float(((bBuffer.out[3] & 0x80) << 24) | ((bBuffer.out[3] & 0x7F) << 20)));
        }
    }

    __syncthreads();

#pragma unroll
    for (int x = 0; x < PART; x++) {
        float val = sdata[x][tid];
        float c = 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            float y = __shfl_down_sync(0xffffffff, val, offset) - c;
            float t = val + y;
            c = (t - val) - y;
            val = t;
        }

        // 写回每个 warp 的归约结果
        if ((tid % warpSize) == 0)
            sdata[x][tid / warpSize] = val;
    }

    __syncthreads();

    // 再在第一个 warp 上归约 warp-level 部分
    if (tid < warpSize) {
#pragma unroll
        for (int x = 0; x < PART; x++) {
            float val = (tid < THREAD_PER_BLOCK / warpSize) ? sdata[x][tid] : 0.0f;
            float c = 0.0f;

            for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
                float y = __shfl_down_sync(0xffffffff, val, offset) - c;
                float t = val + y;
                c = (t - val) - y;
                val = t;
            }

            if (tid == 0)
                sdata[x][0] = val;
        }
    }

    __syncthreads();

    if (tid == 0) {
        if (bias == nullptr) {
            for (int x = 0; x < PART; x++)
                C[st + k * x] = sdata[x][0] * magicScaleConstant;
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++)
                C[st + k * x] = sdata[x][0] * magicScaleConstant + bias[st];
        }
    }
    __syncthreads();
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvHalfFP8E4M3Kernel1MultiRow(half *A, uint8_t *B, half *C, half *bias, float *scales, int m, int k, int blockM, int blockK) {

    int tid = threadIdx.x;
    __share__ float sdata[PART][THREAD_PER_BLOCK];

    int ms = (m - 1) / blockM + 1;
    int st = blockIdx.x;

    uint8_t *baseB = B + st * m;
    float *baseScale = (st / blockK) * ms;

#pragma unroll
    for (int x = 0; x < PART; x++) {
        sdatap[x][tid] = 0.0;
    }

    float curScale = 0.0;
    union_half4 aBuffer;
    union_char4 bBuffer;
    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
        curScale = baseScale[i / blockM];
        bBuffer.in = *reinterpret_cast<const uint32_t *>(baseB + i);

        half b0 = __short_as_half(((bBuffer.out[0] & 0x80) << 8) | ((bBuffer.out[0] & 0x7F) << 7));
        half b1 = __short_as_half(((bBuffer.out[1] & 0x80) << 8) | ((bBuffer.out[1] & 0x7F) << 7));
        half b2 = __short_as_half(((bBuffer.out[2] & 0x80) << 8) | ((bBuffer.out[2] & 0x7F) << 7));
        half b3 = __short_as_half(((bBuffer.out[3] & 0x80) << 8) | ((bBuffer.out[3] & 0x7F) << 7));

        half2 b01 = make_half2(b0, b1);
        half2 b23 = make_half2(b2, b3);

        for (int x = 0; x < PART; x++) {
            aBuffer.in = *reinterpret_cast<const uint2 *>(A + x * m + i);

#if (CUDART_VERSION < 12000) || defined(CUDA_NO_TENSOR_CORE)
            sdata[x][tid] += ((float)regA.out[0] * (float)B01.x + (float)regA.out[1] * (float)B01.y + (float)regA.out[2] * (float)B23.x +
                              (float)regA.out[3] * (float)B23.y) *
                             curScale;
#else
            half2 p01 = __hmul2(aBuffer.out2[0], b01);
            half2 p02 = __hmul2(aBuffer.out2[1], b23);
            half2 sum0102 = __hadd2(p01, p02);
            half sum = __hadd(sum0102.x, sum0102.y);
            sdatap[x][i] += __half2float(sum) * curScale;
#endif
        }
    }
    __syncthreads();

    float diff = 0.0f;
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                float other = sdata[x][tid + s] - diff;
                float sumTmp = sdata[x][tid] + other;
                diff = (sumTmp - sdata[x][tid]) - other;
                sdata[x][tid] = sumTmp;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (bias == nullptr) {
            for (int x = 0; x < PART; x++)
                C[st + k * x] = (half)(sdata[x][0] * magicScaleConstant);
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++)
                C[st + k * x] = (half)(sdata[x][0] * magicScaleConstant + (float)bias[st]);
        }
    }
    __syncthreads();
}

CudaInfos *cudaInfos = nullptr;

CudaInfos *getCudaInfos() {
    if (cudaInfos == nullptr) {
        cudaInfos = new CudaInfos();
    }
    return cudaInfos;
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

    state = cudaMalloc(&ret, size);
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

void FastllmCudaSetDevice(int gpu_id) { cudaSetDevice(gpu_id); }

int FastllmCudaGetDevice() {
    int id = -1;
    cudaGetDevice(&id);
    return id;
}

void DeviceSync() {
    // cudaDeviceSynchronize();
}

void FastllmCudaClearBigBuffer() {
    int id = -1;
    cudaGetDevice(&id);

    if (bigBuffersMap.empty()) {
        return;
    }

    for (auto &it : bigBuffersMap) {
        auto &bigBuffers = it.second;

        long long littleMemSum = 0;
        long long littleMemSumLimit = 300 * 1024 * 1024; // 留一小部分复用
        std::set<int> limitedID;
        std::vector<std::pair<size_t, int>> idle_size;
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (!bigBuffers[i].busy) {
                idle_size.push_back(std::make_pair(bigBuffers[i].size, i));
            }
        }

        std::sort(idle_size.begin(), idle_size.end());
        for (int i = 0; i < idle_size.size(); i++) {
            littleMemSum += idle_size[i].first;

            if (littleMemSum > littleMemSumLimit) {
                break;
            } else {
                limitedID.insert(idle_size[i].second);
            }
        }

        std::vector<CudaMemoryBuffer> temp;
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (!bigBuffers[i].busy && limitedID.find(i) == limitedID.end()) {
                cudaError state = cudaSuccess;
                cudaSetDevice(it.first);
                state = cudaFree(bigBuffers[i].data);
                if (cudaSuccess != state)
                    printf("Error: CUDA error when release memory on device %d!", it.first);
                checkCudaErrors("", state);
            } else {
                temp.push_back(bigBuffers[i]);
            }
        }

        bigBuffers.clear();
        bigBuffers = temp;
    }

    cudaSetDevice(id);
}

void FastllmCudaMemcpy2DDeviceToDevice(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height) {
    cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice);
    DeviceSync();
}

void *FastllmCudaDirectMalloc(size_t size) {
    void *ret;
    cudaError state = cudaMalloc(&ret, size);
    if (cudaSuccess != state) {
        printf("Error: CUDA error when allocating %lu kB memory! maybe there's no enough memory left on device.", size >> 10);
        checkCudaErrors("", state);
        return nullptr;
    }
    return ret;
}

void FastllmCudaMemset0(void *ret, size_t size) { cudaMemset(ret, 0, size); }

void *FastllmCudaPrepareInput(const Data &input) {
    void *ret = nullptr;
    if (input.dataDevice == DataDevice::CUDA) {
        // 已经在 CUDA 上
        ret = (void *)input.cudaData;
    } else {
        // 在 CPU 上，需要先在 GPU 上分配显存
        cudaError_t err = cudaMalloc(&ret, input.expansionBytes);
        if (err != cudaSuccess) {
            checkCudaErrors("Error: CUDA malloc failed!", err);
            return nullptr;
        }

        // 拷贝数据
        err = cudaMemcpy(ret, input.cpuData, input.expansionBytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            checkCudaErrors("Error: CUDA memcpy H2D failed!", err);
            cudaFree(ret);
            return nullptr;
        }
    }
    return ret;
}

void FastllmCudaFinishInput(const Data &input, void *data) {
    if (input.dataDevice != DataDevice::CUDA) {
        FastllmCudaFree(data);
    }
}

void *FastllmCudaPrepareOutput(Data &output) {
    void *ret;
    if (output.dataDevice == DataDevice::CUDA) {
        ret = (float *)output.cudaData;
    } else {
        ret = (float *)FastllmCudaMalloc(output.expansionBytes);
    }
    return ret;
}

void FastllmCudaFinishOutput(Data &output, void *data) {
    if (output.dataDevice != DataDevice::CUDA) {
        FastllmCudaCopyFromDeviceToHost(output.cpuData, data, output.expansionBytes);
        FastllmCudaFree(data);
    }
}

bool FastllmCudaGelu(const Data &input, Data &output) {
    int len = input.Count(0);
    float *cudaInput = (float *)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *)FastllmCudaPrepareOutput(output);
    int threadPerBlock = std::min(256, len);

    if (input.dataType == DataType::FLOAT16) {
        FastllmGeluKernel<<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>((half *)cudaInput, (half *)cudaOutput, len);
    } else if (input.dataType == DataType::FLOAT32) {
        FastllmGeluKernel<<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, len);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaGeluNew(const Data &input, Data &output) {
    int len = input.Count(0);
    float *cudaInput = (float *)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *)FastllmCudaPrepareOutput(output);
    int threadPerBlock = std::min(256, len);
    FastllmGeluNewKernel<<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, len);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmFloatToHalf(void *a, void *b, int len) {
    int threadPerBlock = std::min(256, len);
    FastllmCudaFloat2HalfKernel<<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>((float *)a, (half *)b, len);
    DeviceSync();
    return true;
}

bool FastllmHalfToFloat(void *a, void *b, int len) {
    int threadPerBlock = std::min(256, len);
    FastllmCudaHalf2FloatKernel<<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>((half *)a, (float *)b, len);
    DeviceSync();
    return true;
}

bool FastllmBF16ToFloat(void *a, void *b, int len) {
    int threadPerBlock = std::min(256, len);
    FastllmCudaBF162FloatKernel<<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>((uint16_t *)a, (float *)b, len);
    DeviceSync();
    return true;
}

bool FastllmCudaEmbedding(const Data &input, const Data &weight, Data &output) {
    int vocabSize = weight.dims[0], embSize = weight.dims[1];
    uint64_t inputLen = input.Count(0);

    float *inputData = (float *)input.cudaData;
    float *dstOutputData = (float *)output.cudaData;

    if (weight.dataType == DataType::FLOAT32) {
        float *outputData = (float *)dstOutputData;
        float *weightData = (float *)weight.cudaData;
        FastllmCudaFloatEmbeddingKernel<128><<<inputLen, 128>>>(inputData, weightData, outputData, embSize);
    } else if (weight.dataType == DataType::FLOAT16) {
        half *outputData = (half *)dstOutputData;
        half *weightData = (half *)weight.cudaData;
        FastllmCudaFloatEmbeddingKernel<128><<<inputLen, 128>>>(inputData, weightData, outputData, embSize);
    } else if (weight.dataType == DataType::BFLOAT16) {
        std::vector<float> cpuInputData = std::vector<float>(inputLen, 0.0f);
        FastllmCudaCopyFromDeviceToHost(cpuInputData.data(), inputData, inputLen * sizeof(float));
        float *outputData = (float *)dstOutputData;
        uint16_t *weightData = (uint16_t *)weight.cudaData;
        for (int i = 0; i < inputLen; i++) {
            int token = (int)(cpuInputData[i] + 1e-9);
            FastllmBF16ToFloat(weightData + token * embSize, outputData + i * embSize, embSize);
        }
    } else {
    }

    DeviceSync();
    return true;
}

bool FastllmCudaRMSNorm(const Data &input, Data &weight, Data &output, float eps) {
    float *cudaInput = (float *)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *)FastllmCudaPrepareInput(output);

    int dimsLen = input.dims.size();
    int axis = dimsLen - 1;
    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];

    if (input.dataType == DataType::FLOAT32) {
        if (channels < 64) {
            FastllmRMSNormKernelInner1<1><<<outer, 1>>>(cudaInput, (float *)weight.cudaData, cudaOutput, outer, channels, eps);
        } else if (channels < 512) {
            FastllmRMSNormKernelInner1<64><<<outer, 64>>>(cudaInput, (float *)weight.cudaData, cudaOutput, outer, channels, eps);
        } else {
            FastllmRMSNormKernelInner1<512><<<outer, 512>>>(cudaInput, (float *)weight.cudaData, cudaOutput, outer, channels, eps);
        }
    } else if (input.dataType == DataType::FLOAT16) {
        if (channels < 512) {
            FastllmRMSNormKernelInner1<64><<<outer, 64>>>((half *)cudaInput, (float *)weight.cudaData, (half *)cudaOutput, outer, channels, eps);
        } else {
            FastllmRMSNormKernelInner1<512><<<outer, 512>>>((half *)cudaInput, (float *)weight.cudaData, (half *)cudaOutput, outer, channels, eps);
        }
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaLayerNorm(const Data &input, Data &gamma, Data &beta, Data &output, int axis) {
    float *cudaInput = (float *)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *)FastllmCudaPrepareInput(output);

    int dimsLen = input.dims.size();
    axis = (axis % dimsLen + dimsLen) % dimsLen;
    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];
    int inner = input.strides[axis];

    if (inner == 1) {
        if (gamma.dataType != DataType::FLOAT32 || beta.dataType != DataType::FLOAT32) {
            printf("layernorm datatype error.\n");
            exit(0);
        } else if (input.dataType == DataType::FLOAT32) {
            if (channels < 64) {
                FastllmLayerNormKernelInner1<1>
                    <<<outer, 1>>>(cudaInput, (float *)gamma.cudaData, (float *)beta.cudaData, cudaOutput, outer, channels);
            } else if (channels < 512) {
                FastllmLayerNormKernelInner1<64>
                    <<<outer, 64>>>(cudaInput, (float *)gamma.cudaData, (float *)beta.cudaData, cudaOutput, outer, channels);
            } else {
                FastllmLayerNormKernelInner1<512>
                    <<<outer, 512>>>(cudaInput, (float *)gamma.cudaData, (float *)beta.cudaData, cudaOutput, outer, channels);
            }
        } else if (input.dataType == DataType::FLOAT16) {
            if (channels < 64) {
                FastllmLayerNormKernelInner1<1>
                    <<<outer, 1>>>((half *)cudaInput, (float *)gamma.cudaData, (float *)beta.cudaData, (half *)cudaOutput, outer, channels);
            } else if (channels < 512) {
                FastllmLayerNormKernelInner1<64>
                    <<<outer, 64>>>((half *)cudaInput, (float *)gamma.cudaData, (float *)beta.cudaData, (half *)cudaOutput, outer, channels);
            } else {
                FastllmLayerNormKernelInner1<512>
                    <<<outer, 512>>>((half *)cudaInput, (float *)gamma.cudaData, (float *)beta.cudaData, (half *)cudaOutput, outer, channels);
            }
        } else {
            printf("layernorm datatype error.\n");
            exit(0);
        }
    } else {
        printf("layernorm error.\n");
        exit(0);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaSoftmax(const Data &input, Data &output, int axis) {
    float *cudaInput = (float *)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *)FastllmCudaPrepareInput(output);

    int dimsLen = input.dims.size();
    axis = (axis % dimsLen + dimsLen) % dimsLen;
    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];
    int inner = input.Count(axis + 1);
    if (inner == 1) {
        if (input.dataType == DataType::FLOAT32) {
            if (channels < 8) {
                FastllmSoftmaxKernelInner1<1><<<outer, 1>>>(cudaInput, cudaOutput, outer, channels);
            } else if (channels < 64) {
                FastllmSoftmaxKernelInner1<8><<<outer, 8>>>(cudaInput, cudaOutput, outer, channels);
            } else if (channels < 512) {
                FastllmSoftmaxKernelInner1<64><<<outer, 64>>>(cudaInput, cudaOutput, outer, channels);
            } else {
                FastllmSoftmaxKernelInner1<256><<<outer, 256>>>(cudaInput, cudaOutput, outer, channels);
            }
        } else {
            if (channels < 8) {
                FastllmSoftmaxKernelInner1<1><<<outer, 1>>>((half *)cudaInput, (half *)cudaOutput, outer, channels);
            } else if (channels < 64) {
                FastllmSoftmaxKernelInner1<8><<<outer, 8>>>((half *)cudaInput, (half *)cudaOutput, outer, channels);
            } else if (channels < 512) {
                FastllmSoftmaxKernelInner1<64><<<outer, 64>>>((half *)cudaInput, (half *)cudaOutput, outer, channels);
            } else {
                FastllmSoftmaxKernelInner1<256><<<outer, 256>>>((half *)cudaInput, (half *)cudaOutput, outer, channels);
            }
        }
    } else {
        printf("softmax error.\n");
        exit(0);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaAddTo(Data &input0, const Data &input1, float alpha) {
    int len = input0.Count(0);
    float *cudaData = (float *)FastllmCudaPrepareInput(input0);
    float *input1Data = (float *)FastllmCudaPrepareInput(input1);

    int threadPerBlock = std::min(1024, len);
    if (input0.dataType == DataType::FLOAT32) {
        FastllmAddToKernel<<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaData, input1Data, alpha, len);
    } else if (input0.dataType == DataType::FLOAT16) {
        FastllmAddToKernel<<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>((half *)cudaData, (half *)input1Data, __float2half_rn(alpha), len);
    }

    FastllmCudaFinishInput(input1, input1Data);
    FastllmCudaFinishOutput(input0, cudaData);
    return true;
}

bool FastllmCudaMulTo(Data &input0, const Data &input1, float alpha) {
    int len = input0.Count(0);
    float *cudaData = (float *)FastllmCudaPrepareInput(input0);
    float *input1Data = (float *)FastllmCudaPrepareInput(input1);

    int threadPerBlock = std::min(256, len);
    if (input0.dataType == DataType::FLOAT32) {
        FastllmMulToKernel<<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaData, input1Data, alpha, len);
    } else {
        FastllmMulToKernel<<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>((half *)cudaData, (half *)input1Data, alpha, len);
    }
    FastllmCudaFinishInput(input1, input1Data);
    FastllmCudaFinishOutput(input0, cudaData);
    return true;
}

bool FastllmCudaMul(const Data &input, float v, Data &output) {
    int len = input.Count(0);
    float *cudaInput = (float *)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *)FastllmCudaPrepareOutput(output);
    int threadPerBlock = std::min(256, len);

    if (input.dataType == DataType::FLOAT32) {
        FastllmMulKernel<<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, v, len);
    } else {
        FastllmMulKernel<<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>((half *)cudaInput, (half *)cudaOutput, __float2half_rn(v), len);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaSoftmaxBatch(Data **inputs, Data **outputs, int axis, int batch) {
    int total = 0;
    for (int b = 0; b < batch; b++) {
        auto &input = *inputs[b];
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int outer = input.Count(0) / input.Count(axis);
        total += outer;
    }
    uint8_t **pointers = (uint8_t **)FastllmCudaMalloc(sizeof(uint8_t *) * total * 3);
    uint8_t **cpuPointers = new uint8_t *[total * 3];
    int cur = 0;

    for (int b = 0; b < batch; b++) {
        auto &input = *inputs[b];
        auto &output = *outputs[b];
        float *cudaInput = (float *)input.cudaData;
        float *cudaOutput = (float *)output.cudaData;

        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int outer = input.Count(0) / input.Count(axis);
        int channels = input.dims[axis];
        int inner = input.Count(axis + 1);

        if (inner == 1) {
            for (int o = 0; o < outer; o++) {
                cpuPointers[cur * 3 + 0] = (uint8_t *)(cudaInput + o * channels);
                cpuPointers[cur * 3 + 1] = (uint8_t *)(cudaOutput + o * channels);
                cpuPointers[cur * 3 + 2] = (uint8_t *)((size_t)channels);
                cur++;
            }
        } else {
            printf("softmax error.\n");
            exit(0);
        }
    }

    cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t *) * total * 3, cudaMemcpyHostToDevice);
    FastllmSoftmaxKernelBatchInner1<float, 256><<<total, 256>>>(pointers);

    FastllmCudaFree(pointers);
    delete[] cpuPointers;
    DeviceSync();
    return true;
}

bool FastllmCudaTopK(const Data &input, Data &output, int topk) {
    if (topk > 50) {
        printf("topk: unsupport topk > 50.");
        exit(0);
    }

    float *cudaInput = (float *)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *)FastllmCudaPrepareInput(output);

    int dimsLen = input.dims.size();
    int outer = input.Count(0) / input.Count(dimsLen - 1);
    int channels = input.dims[dimsLen - 1];

#ifdef USE_ROCM
    if (topk == 1) {
        FastllmLayerNormKernelTop1<256><<<outer, 256>>>(cudaInput, cudaOutput, channels);
    } else {
        FastllmLayerNormKernelTopK<64, 50><<<outer, 64>>>(cudaInput, cudaOutput, topk, channels);
    }
#else
    if (outer > 4 || topk == 1) {
        if (topk == 1) {
            FastllmLayerNormKernelTop1<256><<<outer, 256>>>(cudaInput, cudaOutput, channels);
        } else {
            FastllmLayerNormKernelTopK<64, 50><<<outer, 64>>>(cudaInput, cudaOutput, topk, channels);
        }
    } else {
        TopKFunctor functor(cudaInput, cudaOutput, channels, topk);
        for (int i = 0; i < outer; ++i) {
            functor(i);
        }
    }
#endif
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaPermute(Data &input, const std::vector<int> &axis) {
    if (input.dataDevice != DataDevice::CUDA) {
        printf("permute: data should in cuda.\n");
        exit(0);
    }

    int len = input.Count(0);
    uint8_t *tempData = (uint8_t *)FastllmCudaMalloc(len * input.unitSize);
    cudaMemcpy(tempData, input.cudaData, len * input.unitSize, cudaMemcpyKind::cudaMemcpyDeviceToDevice);

    std::vector<int> new_dims;
    for (int i = 0; i < axis.size(); i++) {
        new_dims.push_back(input.dims[axis[i]]);
    }

    if (axis == std::vector<int>{1, 0, 2}) {
        int n = input.dims[0];
        int m = input.dims[1];
        int k = input.dims[2];
        FastllmTransposeByRowKernel<256><<<n * m, 256>>>((uint8_t *)input.cudaData, tempData, n, m, k * input.unitSize);
        input.Resize(new_dims);
    } else if (axis == std::vector<int>{2, 0, 1, 3}) {
        int n = input.dims[0] * input.dims[1];
        int m = input.dims[2];
        int k = input.dims[3];
        FastllmTransposeByRowKernel<256><<<n * m, 256>>>((uint8_t *)input.cudaData, tempData, n, m, k * input.unitSize);
        input.Resize(new_dims);
    } else if (axis == std::vector<int>{1, 2, 0, 3}) {
        int n = input.dims[0];
        int m = input.dims[1] * input.dims[2];
        int k = input.dims[3];
        FastllmTransposeByRowKernel<256><<<n * m, 256>>>((uint8_t *)input.cudaData, tempData, n, m, k * input.unitSize);
        input.Resize(new_dims);
    } else if (axis == std::vector<int>{0, 2, 1, 3} && input.dims[0] == 1) {
        int n = input.dims[1];
        int m = input.dims[2];
        int k = input.dims[3];
        FastllmTransposeByRowKernel<256><<<n * m, 256>>>((uint8_t *)input.cudaData, tempData, n, m, k * input.unitSize);
        input.Resize(new_dims);
    } else {
        int threadPerBlock = std::min(256, len);
        uint8_t *stride_old = (uint8_t *)FastllmCudaMalloc(input.strides.size() * sizeof(uint64_t));
        cudaMemcpy(stride_old, input.strides.data(), input.strides.size() * sizeof(uint64_t), cudaMemcpyKind::cudaMemcpyHostToDevice);

        input.Resize(new_dims);
        uint8_t *stride_new = (uint8_t *)FastllmCudaMalloc(input.strides.size() * sizeof(uint64_t));
        cudaMemcpy(stride_new, input.strides.data(), input.strides.size() * sizeof(uint64_t), cudaMemcpyKind::cudaMemcpyHostToDevice);

        int axisLen = axis.size();
        int *cudaaxis = (int *)FastllmCudaMalloc(axis.size() * sizeof(int));
        cudaMemcpy(cudaaxis, axis.data(), axis.size() * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);

        if (input.unitSize == 4) {
            FastllmPermuteKernel<float><<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(
                (float *)input.cudaData, (float *)tempData, cudaaxis, (uint64_t *)stride_old, (uint64_t *)stride_new, axisLen, len);
        } else if (input.unitSize == 2) {
            FastllmPermuteKernel<uint16_t><<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(
                (uint16_t *)input.cudaData, (uint16_t *)tempData, cudaaxis, (uint64_t *)stride_old, (uint64_t *)stride_new, axisLen, len);
        } else if (input.unitSize == 1) {
            FastllmPermuteKernel<uint8_t><<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(
                (uint8_t *)input.cudaData, (uint8_t *)tempData, cudaaxis, (uint64_t *)stride_old, (uint64_t *)stride_new, axisLen, len);
        }

        FastllmCudaFree(stride_old);
        FastllmCudaFree(stride_new);
        FastllmCudaFree(cudaaxis);
    }

    FastllmCudaFree(tempData);
    DeviceSync();
    return true;
}

int GetPointerDeviceId(void *ptr) {
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

    if (err == cudaSuccess) {
#if (CUDART_VERSION < 10000) && !(defined(USE_ROCM))
        if (attributes.memoryType == cudaMemoryTypeDevice) {
#else
        if (attributes.type == cudaMemoryTypeDevice) {
#endif
            int device = attributes.device;
            printf("Pointer belongs to device %d\n", device);
            return device;
        } else {
            printf("Pointer is not device memory\n");
            return -1;
        }
    } else {
        printf("Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
}

int FastllmCudaGetDeviceCount() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return deviceCount;
}

bool FastllmCudaMLA(const Data &qNope, const Data &qPe, const Data &kvCache, const Data &peCache, Data &ss, Data &output, float softmaxScale) {
    int b = qPe.dims[0], s = qPe.dims[1], h = qPe.dims[2], c = qNope.dims.back(), t = kvCache.dims[1], r = qPe.dims[3];
    auto fastllmCublasHandle = getFastllmCublasHandle();
    cublasStatus_t status;

    if (qNope.dataType == DataType::FLOAT32) {
        float *score = (float *)FastllmCudaMalloc(b * s * h * t * sizeof(float));
        float alpha = softmaxScale, beta0 = 0.0f, beta1 = 1.0f;
        status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_T,
                                           CUBLAS_OP_N,
                                           t,
                                           h,
                                           c,
                                           &alpha,
                                           (float *)kvCache.cudaData,
                                           c,
                                           t * c,
                                           (float *)qNope.cudaData,
                                           c,
                                           h * c,
                                           &beta0,
                                           score,
                                           t,
                                           t * h,
                                           1);
        status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_T,
                                           CUBLAS_OP_N,
                                           t,
                                           h,
                                           r,
                                           &alpha,
                                           (float *)peCache.cudaData,
                                           r,
                                           t * r,
                                           (float *)qPe.cudaData,
                                           r,
                                           h * r,
                                           &beta1,
                                           score,
                                           t,
                                           t * h,
                                           1);
        int outer = b * s * h, channels = t;
        FastllmSoftmaxKernelInner1<64><<<outer, 64>>>(score, score, outer, channels);
        status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_N,
                                           CUBLAS_OP_N,
                                           c,
                                           b * s * h,
                                           t,
                                           &beta1,
                                           (float *)kvCache.cudaData,
                                           c,
                                           t * c,
                                           score,
                                           t,
                                           b * s * h * t,
                                           &beta0,
                                           (float *)output.cudaData,
                                           c,
                                           c * b * s * h,
                                           1);
        FastllmCudaFree(score);
    } else if (qNope.dataType == DataType::FLOAT16) {
        half *score = (half *)FastllmCudaMalloc(b * s * h * t * sizeof(half));
        half alpha = __float2half_rn(softmaxScale), beta0 = __float2half_rn(0.0f), beta1 = __float2half_rn(1.0f);
        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_T,
                                           CUBLAS_OP_N,
                                           t,
                                           h,
                                           c,
                                           &alpha,
                                           (half *)kvCache.cudaData,
                                           c,
                                           t * c,
                                           (half *)qNope.cudaData,
                                           c,
                                           h * c,
                                           &beta0,
                                           score,
                                           t,
                                           t * h,
                                           1);
        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_T,
                                           CUBLAS_OP_N,
                                           t,
                                           h,
                                           r,
                                           &alpha,
                                           (half *)peCache.cudaData,
                                           r,
                                           t * r,
                                           (half *)qPe.cudaData,
                                           r,
                                           h * r,
                                           &beta1,
                                           score,
                                           t,
                                           t * h,
                                           1);
        int outer = b * s * h, channels = t;
        FastllmSoftmaxKernelInner1<64><<<outer, 64>>>(score, score, outer, channels);
        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_N,
                                           CUBLAS_OP_N,
                                           c,
                                           b * s * h,
                                           t,
                                           &beta1,
                                           (half *)kvCache.cudaData,
                                           c,
                                           t * c,
                                           score,
                                           t,
                                           b * s * h * t,
                                           &beta0,
                                           (half *)output.cudaData,
                                           c,
                                           c * b * s * h,
                                           1);
        FastllmCudaFree(score);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("status = %d\n", (int)status);
        printf("Error: cublas error during MatMul in MLA operator.\n");
        throw("cublas error");
    }

    DeviceSync();
    return true;
}

bool FastllmCudaAttention(const Data &q, const Data &k, const Data &v, const Data &mask, const Data &output, int group, float scale, int maskType) {
    int q0 = q.dims[0], q1 = q.dims[1], q2 = q.dims[2], k0 = k.dims[0], k1 = k.dims[1], k2 = k.dims[2], v0 = v.dims[0], v1 = v.dims[1],
        v2 = v.dims[2];

    float *qd = (float *)q.cudaData;
    float *kd = (float *)k.cudaData;
    float *vd = (float *)v.cudaData;
    float *od = (float *)output.cudaData;

    float *maskd = mask.dims.size() > 0 ? (float *)mask.cudaData : nullptr;
    int batch = mask.dims.size() == 3 ? mask.dims[0] : 1;
    int maskStride = (mask.dims.size() == 3 ? mask.strides[0] : mask.Count(0));

    auto fastllmCublasHandle = getFastllmCublasHandle();
    cublasStatus_t status;

    float alpha = 1.0f, beta0 = 0.0f;
    if (q1 >= 1024 || (q1 > 1 && q1 != k1 && k1 >= 1024)) {
        float *qk = (float *)FastllmCudaMalloc(q1 * k1 * sizeof(float));
        for (int i = 0; i < q0; i++) {
            status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                               CUBLAS_OP_T,
                                               CUBLAS_OP_N,
                                               k1,
                                               q1,
                                               q2,
                                               &scale,
                                               kd + (i / group) * (k1 * q2),
                                               q2,
                                               k1 * q2,
                                               qd + i * (q1 * q2),
                                               q2,
                                               q1 * q2,
                                               &beta0,
                                               qk,
                                               k1,
                                               k1 * q1,
                                               1);

            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("status = %d\n", (int)status);
                printf("Error: cublas error during MatMulTransB in Attention operator.\n");
                throw("cublas error");
                exit(0);
            }

            if (batch == 1 && maskd == nullptr && maskType == 0) {
                CausalMask<256, float><<<q1, 256>>>(qk, 0, q1, k1, k1 - q1);
                FastllmSoftmaxKernelInner1WithCausalMask<128, float><<<q1, 128>>>(qk, qk, q1, k1, k1 - q1);
            } else {
                if (maskd) {
                    SimpleMask<256><<<q1 * k1 / 256 + 1, 256>>>(qk, maskd + i / (q0 / batch) * maskStride, -10000.0, q1 * k1);
                }

                int outer = q1;
                if (k1 < 8) {
                    FastllmSoftmaxKernelInner1<1><<<outer, 1>>>(qk, qk, outer, k1);
                } else if (k1 < 64) {
                    FastllmSoftmaxKernelInner1<8><<<outer, 8>>>(qk, qk, outer, k1);
                } else if (k1 < 512) {
                    FastllmSoftmaxKernelInner1<64><<<outer, 64>>>(qk, qk, outer, k1);
                } else {
                    FastllmSoftmaxKernelInner1<256><<<outer, 256>>>(qk, qk, outer, k1);
                }
            }

            status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                               CUBLAS_OP_N,
                                               CUBLAS_OP_N,
                                               v2,
                                               q1,
                                               k1,
                                               &alpha,
                                               vd + (i / group) * (k1 * v2),
                                               v2,
                                               k1 * v2,
                                               qk,
                                               k1,
                                               q1 * k1,
                                               &beta0,
                                               od + i * (v2 * q1),
                                               v2,
                                               v2 * q1,
                                               1);

            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("status = %d\n", (int)status);
                printf("Error: cublas error during MatMul in Attention operator.\n");
                throw("cublas error");
                exit(0);
            }
        }

        FastllmCudaFree(qk);

    } else {
        float *qk = (float *)FastllmCudaMalloc(q0 * q1 * k1 * sizeof(float));
        float *temp = (float *)FastllmCudaMalloc(q0 * q1 * k1 * sizeof(float));
        status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_T,
                                           CUBLAS_OP_N,
                                           k1,
                                           q1 * group,
                                           q2,
                                           &scale,
                                           kd,
                                           q2,
                                           k1 * q2,
                                           qd,
                                           q2,
                                           q1 * q2 * group,
                                           &beta0,
                                           qk,
                                           k1,
                                           k1 * q1 * group,
                                           q0 / group);

        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("status = %d\n", (int)status);
            printf("Error: cublas error during MatMulTransB in Attention operator.\n");
            throw("cublas error");
            exit(0);
        }

        if (maskd) {
            int spatial = q1 * k1, n = batch, m = q0 / batch;
            FastllmAttentionMaskKernel<256><<<n * m, 256>>>(qk, maskd, -10000, n, m, spatial);
        }

        int outer = q0 * q1;
        if (k1 < 8) {
            FastllmSoftmaxKernelInner1<1><<<outer, 1>>>(qk, temp, outer, k1);
        } else if (k1 < 64) {
            FastllmSoftmaxKernelInner1<8><<<outer, 8>>>(qk, temp, outer, k1);
        } else if (k1 < 512) {
            FastllmSoftmaxKernelInner1<64><<<outer, 64>>>(qk, temp, outer, k1);
        } else {
            FastllmSoftmaxKernelInner1<256><<<outer, 256>>>(qk, temp, outer, k1);
        }

        status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_N,
                                           CUBLAS_OP_N,
                                           v2,
                                           q1 * group,
                                           k1,
                                           &alpha,
                                           vd,
                                           v2,
                                           k1 * v2,
                                           temp,
                                           k1,
                                           q1 * k1 * group,
                                           &beta0,
                                           od,
                                           v2,
                                           v2 * q1 * group,
                                           q0 / group);

        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("status = %d\n", (int)status);
            printf("Error: cublas error during MatMul in Attention operator.\n");
            throw("cublas error");
            exit(0);
        }

        FastllmCudaFree(qk);
        FastllmCudaFree(temp);
    }

    DeviceSync();
    return true;
}

void GpuQK(half *q, half *k, half *qk, int qlen, int klen, int dim, float scale, int base) {
    const int BQ = 128, BK = 128, DIM = 128;
    dim3 blockDim(128);
    int BX = (qlen + BQ - 1) / BQ;
    int BY = (klen + BK - 1) / BK;
    dim3 gridDim(BX, BY);
    HalfFC<BQ, DIM, BK><<<gridDim, blockDim>>>(q, k, qk, qlen, dim, klen, (half)scale, base);
}

bool FastllmCudaHalfAttention(
    const Data &q, const Data &k, const Data &v, const Data &mask, const Data &output, int group, float scale, int maskType) {
    int q0 = q.dims[0], q1 = q.dims[1], q2 = q.dims[2], k0 = k.dims[0], k1 = k.dims[1], k2 = k.dims[2], v0 = v.dims[0], v1 = v.dims[1],
        v2 = v.dims[2];

    half *qd = (half *)q.cudaData;
    half *kd = (half *)k.cudaData;
    half *vd = (half *)v.cudaData;
    half *od = (half *)output.cudaData;

    half *maskd = mask.dims.size() > 0 ? (half *)mask.cudaData : nullptr;
    int batch = mask.dims.size() == 3 ? mask.dims[0] : 1;
    int maskStride = (mask.dims.size() == 3 ? mask.strides[0] : mask.Count(0));

    auto fastllmCublasHandle = getFastllmCublasHandle();
    cublasStatus_t status;

    half beta = __float2half_rn(0.0f), one = __float2half_rn(1.0f), hscale = __float2half_rn(scale);

    if (q1 >= 1024 || (q1 > 1 && q1 != k1 && k1 >= 1024)) {
        bool useFastAtten = getCudaInfos()->hasTensorCore && (q2 == 128 && v2 == 128) && (batch == 1) && maskType == 0;
        useFastAtten = useFastAtten && (q1 % 1024 == 0) && (k1 % 1024 == 0);

        int alignQ1 = q1;
        int alignK1 = k1;
        int part = alignK1;
        if (useFastAtten == true) {
            alignQ1 = ((alignQ1 - 1) / 128 + 1) * 128;
            alignK1 = ((alignK1 - 1) / 128 + 1) * 128;
            part = alignK1 > 8192 ? 8192 : alignK1;
        }

        half *qk = (half *)FastllmCudaMalloc(alignQ1 * part * sizeof(half));
        cudaMemset(qk, 0.0f, alignQ1 * alignK1 * sizeof(half));

        for (int i = 0; i < q0; i++) {
            if (useFastAtten) {
                if (alignK1 > 8192) {
                    float *lastMax = (float *)FastllmCudaMalloc(alignQ1 * sizeof(float));
                    float *lastSum = (float *)FastllmCudaMalloc(alignQ1 * sizeof(float));
                    float *currentMax = (float *)FastllmCudaMalloc(alignQ1 * sizeof(float));
                    float *currentSum = (float *)FastllmCudaMalloc(alignQ1 * sizeof(float));

                    int threadPerBlock = min(256, alignQ1);
                    InitBlockAtten<<<(alignQ1 - 1) / threadPerBlock + 1, threadPerBlock>>>(lastSum, lastMax, currentSum, currentMax, alignQ1);

                    part = 8192;
                    for (int st = 0; i < alignK1; st += part) {
                        int len = min(part, alignK1 - st);
                        status = cublasHgemm(fastllmCublasHandle,
                                             CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             len,
                                             alignQ1,
                                             q2,
                                             &hscale,
                                             kd + (i / group) * k.Count(1) + st * k.strides[1],
                                             k.strides[1],
                                             qd + i * q.Count(1),
                                             q.strides[1],
                                             &beta,
                                             qk,
                                             len);
                        if (status != CUBLAS_STATUS_SUCCESS) {
                            printf("status = %d\n", (int)status);
                            printf("Error: cublas error during MatMul in Attention operator.\n");
                            throw("cublas error");
                            exit(0);
                        }

                        CausalMask<256, half><<<q1, 256>>>(qk, __float2half_rn(0.0f), alignQ1, len, k1 - q1 - st);
                        FastllmSoftmaxKernelInner1WithCausalMask<256, half><<<q1, 256>>>(qk, qk, alignQ1, len, k1 - q1 - st, currentMax, currentSum);

                        if (st > 0) {
                            AttnBlockUpdate<128><<<alignQ1, 128>>>(od + i * q1 * v2, alignQ1, v2, lastMax, lastSum, currentMax, currentSum);
                        } else {
                            cudaMemcpy(lastMax, currentMax, alignQ1 * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
                            cudaMemcpy(lastSum, currentSum, alignQ1 * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
                        }

                        half currentScale = __float2half(st > 0 ? 1.0f : 0.0f);
                        status = cublasHgemm(fastllmCublasHandle,
                                             CUBLAS_OP_N,
                                             CUBLAS_OP_N,
                                             v2,
                                             alignQ1,
                                             len,
                                             &one,
                                             vd + (i / group) * v.Count(1) + st * v.strides[1],
                                             v.strides[1],
                                             qk,
                                             len,
                                             &currentScale,
                                             od + i * q1 * v2,
                                             v2);
                        if (status != CUBLAS_STATUS_SUCCESS) {
                            printf("status = %d\n", (int)status);
                            printf("Error: cublas error during MatMul in Attention operator.\n");
                            throw("cublas error");
                            exit(0);
                        }
                    }

                    FastllmCudaFree(lastMax);
                    FastllmCudaFree(lastSum);
                    FastllmCudaFree(currentMax);
                    FastllmCudaFree(currentSum);

                } else {
                    GpuQK(qd + i * q.Count(1), kd + (i / group) * k.Count(1), qk, alignQ1, alignK1, q2, scale, k1 - q1);
                    FastllmSoftmaxKernelInner1WithCausalMask<128, half><<<q1, 128>>>(qk, qk, q1, alignK1, k1 - q1);
                    status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                                       CUBLAS_OP_N,
                                                       CUBLAS_OP_N,
                                                       v2,
                                                       q1,
                                                       k1,
                                                       &one,
                                                       vd + (i / group) * (k1 * v2),
                                                       v2,
                                                       k1 * v2,
                                                       qk,
                                                       k1,
                                                       q1 * k1,
                                                       &beta,
                                                       od + i * (v2 * q1),
                                                       v2,
                                                       v2 * q1,
                                                       1);

                    if (status != CUBLAS_STATUS_SUCCESS) {
                        printf("status = %d\n", (int)status);
                        printf("Error: cublas error during MatMul in Attention operator.\n");
                        throw("cublas error");
                        exit(0);
                    }
                }
            } else {
                status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                                   CUBLAS_OP_T,
                                                   CUBLAS_OP_N,
                                                   k1,
                                                   q1,
                                                   q2,
                                                   &hscale,
                                                   kd + (i / group) * (k1 * q2),
                                                   q2,
                                                   k1 * q2,
                                                   qd + i * (q1 * q2),
                                                   q2,
                                                   q1 * q2,
                                                   &beta,
                                                   qk,
                                                   k1,
                                                   k1 * q1,
                                                   1);

                if (status != CUBLAS_STATUS_SUCCESS) {
                    printf("status = %d\n", (int)status);
                    printf("Error: cublas error during MatMulTransB in Attention operator.\n");
                    throw("cublas error");
                    exit(0);
                }

                if (batch == 1 && maskd == nullptr && maskType == 0) {
                    CausalMask<256, half><<<q1, 256>>>(qk, __float2half(0), q1, k1, k1 - q1);
                    FastllmSoftmaxKernelInner1WithCausalMask<128, half><<<q1, 128>>>(qk, qk, q1, k1, k1 - q1);
                } else {
                    if (maskd) {
                        SimpleMask<256><<<q1 * k1 / 256 + 1, 256>>>(qk, maskd + i / (q0 / batch) * maskStride, __float2half(-10000), q1 * k1);
                    }

                    int outer = q1;
                    if (k1 < 8) {
                        FastllmSoftmaxKernelInner1<1><<<outer, 1>>>(qk, qk, outer, k1);
                    } else if (k1 < 64) {
                        FastllmSoftmaxKernelInner1<8><<<outer, 8>>>(qk, qk, outer, k1);
                    } else if (k1 < 512) {
                        FastllmSoftmaxKernelInner1<64><<<outer, 64>>>(qk, qk, outer, k1);
                    } else {
                        FastllmSoftmaxKernelInner1<256><<<outer, 256>>>(qk, qk, outer, k1);
                    }
                }

                status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                                   CUBLAS_OP_N,
                                                   CUBLAS_OP_N,
                                                   v2,
                                                   q1,
                                                   k1,
                                                   &one,
                                                   vd + (i / group) * (k1 * v2),
                                                   v2,
                                                   k1 * v2,
                                                   qk,
                                                   k1,
                                                   q1 * k1,
                                                   &beta,
                                                   od + i * (v2 * q1),
                                                   v2,
                                                   v2 * q1,
                                                   1);

                if (status != CUBLAS_STATUS_SUCCESS) {
                    printf("status = %d\n", (int)status);
                    printf("Error: cublas error during MatMul in Attention operator.\n");
                    throw("cublas error");
                    exit(0);
                }
            }
        }
        FastllmCudaFree(qk);
    } else {

        half *qk = (half *)FastllmCudaMalloc(q0 * q1 * k1 * sizeof(half));
        half *temp = (half *)FastllmCudaMalloc(q0 * q1 * k1 * sizeof(half));
        auto fastllmCublasHandle = getFastllmCublasHandle();
        cublasStatus_t status;

        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_T,
                                           CUBLAS_OP_N,
                                           k1,
                                           q1 * group,
                                           q2,
                                           &hscale,
                                           kd,
                                           k.strides[1],
                                           k.Count(1),
                                           qd,
                                           q.strides[1],
                                           q.Count(1) * group,
                                           &beta,
                                           qk,
                                           k1,
                                           k1 * q1 * group,
                                           q0 / group);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("status = %d\n", (int)status);
            printf("Error: cublas error during MatMulTransB in Attention operator.\n");
            throw("cublas error");
            exit(0);
        }

        if (maskd) {
            int spatial = q1 * k1, n = batch, m = q0 / batch;
            FastllmAttentionMaskKernel<256><<<n * m, 256>>>(qk, maskd, __float2half_rn(-10000), n, m, spatial);
        }

        int outer = q0 * q1;
        if (k1 < 8) {
            FastllmSoftmaxKernelInner1<1><<<outer, 1>>>(qk, temp, outer, k1);
        } else if (k1 < 64) {
            FastllmSoftmaxKernelInner1<8><<<outer, 8>>>(qk, temp, outer, k1);
        } else if (k1 < 512) {
            FastllmSoftmaxKernelInner1<64><<<outer, 64>>>(qk, temp, outer, k1);
        } else {
            FastllmSoftmaxKernelInner1<256><<<outer, 256>>>(qk, temp, outer, k1);
        }

        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_N,
                                           CUBLAS_OP_N,
                                           v2,
                                           q1 * group,
                                           k1,
                                           &one,
                                           vd,
                                           v.strides[1],
                                           v.Count(1),
                                           temp,
                                           k1,
                                           k1 * q1 * group,
                                           &beta,
                                           od,
                                           v2,
                                           v2 * q1 * group,
                                           q0 / group);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("status = %d\n", (int)status);
            printf("Error: cublas error during MatMul in Attention operator.\n");
            throw("cublas error");
            exit(0);
        }
        FastllmCudaFree(qk);
        FastllmCudaFree(temp);
        DeviceSync();
    }

    return true;
}

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
                            float alpha) {

    float *cudaInput0 = (float *)FastllmCudaPrepareInput(input0);
    float *cudaInput1 = (float *)FastllmCudaPrepareInput(input1);
    float *cudaOutput = (float *)FastllmCudaPrepareOutput(output);
    float beta = 0;

    auto fastllmCublasHandle = getFastllmCublasHandle();
    cublasStatus_t status;

    if (input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) {
        status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                           cublasOperation_t::CUBLAS_OP_N,
                                           cublasOperation_t::CUBLAS_OP_N,
                                           k,
                                           n,
                                           m,
                                           &alpha,
                                           cudaInput1,
                                           input1Stride,
                                           input1Spatial,
                                           cudaInput0,
                                           input0Stride,
                                           input0Spatial,
                                           &beta,
                                           cudaOutput,
                                           k,
                                           k * n,
                                           batch);

    } else if (input0.dataType == DataType::FLOAT16 && input1.dataType == DataType::FLOAT16) {
        half h_alpha = __float2half(alpha), h_beta = __float2half(beta);
        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                           cublasOperation_t::CUBLAS_OP_N,
                                           cublasOperation_t::CUBLAS_OP_N,
                                           k,
                                           n,
                                           m,
                                           &h_alpha,
                                           (half *)cudaInput1,
                                           input1Stride,
                                           input1Spatial,
                                           (half *)cudaInput0,
                                           input0Stride,
                                           input0Spatial,
                                           &h_beta,
                                           (half *)cudaOutput,
                                           k,
                                           k * n,
                                           batch);

    } else if (input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT16) {
        half h_alpha = __float2half(alpha), h_beta = __float2half(beta);
        half *tempInput0 = (half *)FastllmCudaMalloc(input0.Count(0) * sizeof(half));
        half *tempOutput = (half *)FastllmCudaMalloc(output.Count(0) * sizeof(half));
        FastllmFloatToHalf(cudaInput0, tempInput0, input0.Count(0));

        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                           cublasOperation_t::CUBLAS_OP_N,
                                           cublasOperation_t::CUBLAS_OP_N,
                                           k,
                                           n,
                                           m,
                                           &h_alpha,
                                           (half *)cudaInput1,
                                           input1Stride,
                                           input1Spatial,
                                           tempInput0,
                                           input0Stride,
                                           input0Spatial,
                                           &h_beta,
                                           tempOutput,
                                           k,
                                           k * n,
                                           batch);

        FastllmHalfToFloat(tempOutput, cudaOutput, output.Count(0));
        FastllmCudaFree(tempInput0);
        FastllmCudaFree(tempOutput);
    }

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("status = %d\n", (int)status);
        printf("%d %d %d\n", k, n, m);
        printf("Error: cublas error in batch MatMul.\n");
        throw("cublas error");
        exit(0);
    }

    FastllmCudaFinishInput(input0, cudaInput0);
    FastllmCudaFinishInput(input1, cudaInput1);
    FastllmCudaFinishOutput(output, cudaOutput);

    return true;
}

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
                                  float alpha) {
    float *cudaInput0 = (float *)FastllmCudaPrepareInput(input0);
    float *cudaInput1 = (float *)FastllmCudaPrepareInput(input1);
    float *cudaOutput = (float *)FastllmCudaPrepareOutput(output);
    float beta = 0;
    auto fastllmCublasHandle = getFastllmCublasHandle();
    cublasStatus_t status;

    if (input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) {
        status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_T,
                                           CUBLAS_OP_N,
                                           k,
                                           n,
                                           m,
                                           &alpha,
                                           cudaInput1,
                                           input1Stride,
                                           input1Spatial,
                                           cudaInput0,
                                           input0Stride,
                                           input0Spatial,
                                           &beta,
                                           cudaOutput,
                                           k,
                                           k * n,
                                           batch);
    } else if (input0.dataType == DataType::FLOAT16 && input1.dataType == DataType::FLOAT16) {
        half h_alpha = __float2half(alpha), h_beta = __float2half(beta);
        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_T,
                                           CUBLAS_OP_N,
                                           k,
                                           n,
                                           m,
                                           &h_alpha,
                                           (half *)cudaInput1,
                                           input1Stride,
                                           input1Spatial,
                                           (half *)cudaInput0,
                                           input0Stride,
                                           input0Spatial,
                                           &h_beta,
                                           (half *)cudaOutput,
                                           k,
                                           k * n,
                                           batch);
    } else if (input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT16) {
        half *tempInput0 = (half *)FastllmCudaMalloc(input0.Count(0) * sizeof(half));
        half *tempOutput = (half *)FastllmCudaMalloc(output.Count(0) * sizeof(half));
        FastllmFloatToHalf(cudaInput0, tempInput0, input0.Count(0));

        half h_alpha = __float2half(alpha), h_beta = __float2half(beta);
        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_T,
                                           CUBLAS_OP_N,
                                           k,
                                           n,
                                           m,
                                           &h_alpha,
                                           (half *)cudaInput1,
                                           input1Stride,
                                           input1Spatial,
                                           (half *)tempInput0,
                                           input0Stride,
                                           input0Spatial,
                                           &h_beta,
                                           (half *)tempOutput,
                                           k,
                                           k * n,
                                           batch);
        FastllmHalfToFloat(tempOutput, cudaOutput, output.Count(0));
        FastllmCudaFree(tempInput0);
        FastllmCudaFree(tempOutput);
    }

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("status = %d\n", (int)status);
        printf("%d %d %d\n", k, n, m);
        printf("Error: cublas error in batch MatMulTransB.\n");
        throw("cublas error");
        exit(0);
    }

    FastllmCudaFinishInput(input0, cudaInput0);
    FastllmCudaFinishInput(input1, cudaInput1);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaRotatePosition2D(Data &data, const Data &positionIds, const Data &sinData, const Data &cosData, int rotaryDim) {
    float *cudaData = (float *)FastllmCudaPrepareInput(data);
    float *cudaPositionIds = (float *)FastllmCudaPrepareInput(positionIds);
    float *cudaSin = (float *)FastllmCudaPrepareInput(sinData);
    float *cudaCos = (float *)FastllmCudaPrepareInput(cosData);

    int outer = data.dims[0] * data.dims[1];
    int spatial = data.Count(2);
    int len = data.dims[0], bs = data.dims[1];
    int n = data.dims[2], m = data.dims[3];
    FastllmRotatePosition2DKernel<<<outer * 2 * n, std::min(rotaryDim, m / 4)>>>(
        cudaData, cudaPositionIds, cudaSin, cudaCos, len, bs, spatial, n, m, (int)positionIds.dims.back(), (int)sinData.dims[1], rotaryDim);

    FastllmCudaFinishInput(positionIds, cudaPositionIds);
    FastllmCudaFinishInput(sinData, cudaSin);
    FastllmCudaFinishInput(cosData, cudaCos);
    FastllmCudaFinishOutput(data, cudaData);

    return true;
}

bool FastllmCudaNearlyRotatePosition2D(Data &data, const Data &positionIds, const Data &sinData, const Data &cosData, int rotaryDim) {
    float *cudaData = (float *)FastllmCudaPrepareInput(data);
    float *cudaPositionIds = (float *)FastllmCudaPrepareInput(positionIds);
    float *cudaSin = (float *)FastllmCudaPrepareInput(sinData);
    float *cudaCos = (float *)FastllmCudaPrepareInput(cosData);

    int outer = data.dims[0] * data.dims[1];
    int spatial = data.Count(2);
    int len = data.dims[0], bs = data.dims[1];
    int n = data.dims[2], m = data.dims[3];

    if (data.dataType == DataType::FLOAT32) {
        FastllmNearlyRotatePosition2DKernel<<<outer * n, std::min(rotaryDim, m / 2)>>>(
            cudaData, cudaPositionIds, cudaSin, cudaCos, len, bs, spatial, n, m, (int)positionIds.dims.back(), (int)sinData.dims[1], rotaryDim);
    } else if (data.dataType == DataType::FLOAT16) {
        FastllmNearlyRotatePosition2DKernel<<<outer * n, std::min(rotaryDim, m / 2)>>>((half *)cudaData,
                                                                                       cudaPositionIds,
                                                                                       cudaSin,
                                                                                       cudaCos,
                                                                                       len,
                                                                                       bs,
                                                                                       spatial,
                                                                                       n,
                                                                                       m,
                                                                                       (int)positionIds.dims.back(),
                                                                                       (int)sinData.dims[1],
                                                                                       rotaryDim);
    }

    FastllmCudaFinishInput(positionIds, cudaPositionIds);
    FastllmCudaFinishInput(sinData, cudaSin);
    FastllmCudaFinishInput(cosData, cudaCos);
    FastllmCudaFinishOutput(data, cudaData);
    return true;
}

bool FastllmCudaLlamaRotatePosition2D(Data &data, const Data &positionIds, const Data &sinData, const Data &cosData, int rotaryDim) {
    float *cudaData = (float *)FastllmCudaPrepareInput(data);
    float *cudaPositionIds = (float *)FastllmCudaPrepareInput(positionIds);
    float *cudaSin = (float *)FastllmCudaPrepareInput(sinData);
    float *cudaCos = (float *)FastllmCudaPrepareInput(cosData);

    int outer = data.dims[0] * data.dims[1];
    int spatial = data.Count(2);
    int bs = data.dims[0], len = data.dims[1];
    int n = data.dims[2], m = data.dims[3];

    if (data.dataType == DataType::FLOAT32) {
        FastllmLlamaRotatePosition2DKernel<<<outer * n, std::min(rotaryDim, m / 2)>>>(
            cudaData, cudaPositionIds, cudaSin, cudaCos, len, bs, spatial, n, m, (int)positionIds.dims.back(), (int)sinData.dims[1], rotaryDim);
    } else if (data.dataType == DataType::FLOAT16) {
        FastllmLlamaRotatePosition2DKernel<<<outer * n, std::min(rotaryDim, m / 2)>>>((half *)cudaData,
                                                                                      cudaPositionIds,
                                                                                      cudaSin,
                                                                                      cudaCos,
                                                                                      len,
                                                                                      bs,
                                                                                      spatial,
                                                                                      n,
                                                                                      m,
                                                                                      (int)positionIds.dims.back(),
                                                                                      (int)sinData.dims[1],
                                                                                      rotaryDim);
    }

    FastllmCudaFinishInput(positionIds, cudaPositionIds);
    FastllmCudaFinishInput(sinData, cudaSin);
    FastllmCudaFinishInput(cosData, cudaCos);
    FastllmCudaFinishOutput(data, cudaData);
    return true;
}

bool FastllmCudaApplyLognAttn(Data &input, Data &lognAttn, Data &positionIds) {
    float *inputData = (float *)input.cudaData;
    float *lognData = (float *)lognAttn.cudaData;
    float *posData = (float *)positionIds.cudaData;
    int batch = input.dims[0];
    int seqLen = input.dims[1];
    int spatial = input.Count(2);

    FastllmApplyLognAttnKernel<256><<<batch * seqLen, 256>>>(inputData, lognData, posData, batch, seqLen, spatial);
    return true;
}

bool FastllmCudaRepeatPenalty(Data &input, Data &penalty, Data &penaltyScale) {
    float *inputData = (float *)input.cudaData;
    float *penaltyData = (float *)penalty.cudaData;
    float *penaltyScaleData = (float *)penaltyScale.cudaData;
    int batch = penalty.dims[0], tokens = penalty.dims[1];
    int vocabs = input.dims.back();

    FastllmRepeatPenaltyKernel<64><<<batch, 64>>>(inputData, penaltyData, penaltyScaleData, tokens, vocabs);
    return true;
}

bool FastllmCudaBatchMatMulBatch(
    void **i0s, void **i1s, void **os, int *ns, int *ms, int *ks, int *i0Strides, int *i1Strides, float alpha, int batch) {
    uint8_t **pointers = (uint8_t **)FastllmCudaMalloc(sizeof(uint8_t *) * batch * 8);
    uint8_t **cpuPointers = new uint8_t *[batch * 8];
    for (int i = 0; i < batch; i++) {
        cpuPointers[i * 8 + 0] = (uint8_t *)i0s[i];
        cpuPointers[i * 8 + 1] = (uint8_t *)i1s[i];
        cpuPointers[i * 8 + 2] = (uint8_t *)os[i];
        cpuPointers[i * 8 + 3] = (uint8_t *)(size_t)ns[i];
        cpuPointers[i * 8 + 4] = (uint8_t *)(size_t)ms[i];
        cpuPointers[i * 8 + 5] = (uint8_t *)(size_t)ks[i];
        cpuPointers[i * 8 + 6] = (uint8_t *)(size_t)i0Strides[i];
        cpuPointers[i * 8 + 7] = (uint8_t *)(size_t)i1Strides[i];
    }
    cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t *) * batch * 8, cudaMemcpyHostToDevice);
    FastllmMatMulKernel<128><<<batch, 128>>>(pointers, alpha);
    FastllmCudaFree(pointers);
    delete[] cpuPointers;
    DeviceSync();
    return true;
}

bool FastllmCudaAttentionBatch(Data **q, Data **k, Data **v, Data **mask, Data **output, int group, float scale, int batch) {
    if (q[0]->dataType == DataType::FLOAT32) {
        return DoFastllmCudaAttentionBatch<float>(q, k, v, mask, output, group, scale, batch);
    } else if (q[0]->dataType == DataType::FLOAT16) {
        return DoFastllmCudaAttentionBatch<half>(q, k, v, mask, output, group, scale, batch);
    } else {
        printf("Error: attention datatype error.\n");
        throw("Error: attention datatype error.");
        exit(0);
    }
}

bool FastllmCudaSplitBatch(Data &input, Data **outputs, int axis) {
    int part = input.dims[axis];
    int outer = input.Count(0) / input.Count(axis);
    int inner = input.strides[axis];

    uint8_t **pointers = (uint8_t **)FastllmCudaMalloc(part * sizeof(uint8_t *));
    uint8_t **cpuPointers = new uint8_t *[part];
    for (int i = 0; i < part; i++) {
        cpuPointers[i] = (uint8_t *)outputs[i]->cudaData;
    }
    cudaMemcpy(pointers, cpuPointers, part * sizeof(uint8_t *), cudaMemcpyKind::cudaMemcpyHostToDevice);
    FastllmSplitBatchKernel<256><<<outer * part, 256>>>((uint8_t *)input.cudaData, pointers, outer, part, inner);

    FastllmCudaFree(pointers);
    delete[] cpuPointers;
    DeviceSync();
    return true;
}

bool FastllmCudaCatBatch(Data **inputs, Data &output, int axis) {
    int part = output.dims[axis];
    int outer = output.Count(0) / output.Count(axis);
    int inputStride = inputs[0]->Count(axis);
    int outputStride = output.Count(axis);
    int inner = output.strides[axis];
    int unitSize = output.unitSize;

    uint8_t **pointers = (uint8_t **)FastllmCudaMalloc(sizeof(uint8_t *) * part);
    uint8_t **cpuPointers = new uint8_t *[part];
    for (int i = 0; i < part; i++) {
        cpuPointers[i] = (uint8_t *)inputs[i]->cudaData;
    }
    cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t *) * part, cudaMemcpyHostToDevice);
    FastllmCatBatchKernel<256><<<part * outer, 256>>>(pointers, (uint8_t *)output.cudaData, outer, part, inner * unitSize);

    FastllmCudaFree(pointers);
    delete[] cpuPointers;

    DeviceSync();
    return true;
}

bool FastllmCudaMulBatch(Data **inputs, float v, int batch, Data **outputs) {
    float **pointers = (float **)FastllmCudaMalloc(sizeof(float *) * batch * 3);
    float **cpuPointers = new float *[batch * 3];
    for (int i = 0; i < batch; i++) {
        cpuPointers[i] = (float *)inputs[i]->cudaData;
        cpuPointers[i + batch] = (float *)outputs[i]->cudaData;
        cpuPointers[i + batch * 2] = (float *)(inputs[i]->Count(0));
    }
    cudaMemcpy(pointers, cpuPointers, sizeof(float *) * batch * 3, cudaMemcpyHostToDevice);
    FastllmMulBatchKernel<256><<<batch, 256>>>(pointers, batch, v);

    FastllmCudaFree(pointers);
    delete[] cpuPointers;

    DeviceSync();
    return true;
}

bool FastllmCudaBatchMatMulTransBBatch(
    void **i0s, void **i1s, void **os, int *ns, int *ms, int *ks, int *i0Strides, int *i1Strides, float alpha, int batch) {
    uint8_t **pointers = (uint8_t **)FastllmCudaMalloc(sizeof(uint8_t *) * batch * 8);
    uint8_t **cpuPointers = new uint8_t *[batch * 8];
    for (int i = 0; i < batch; i++) {
        cpuPointers[i * 8 + 0] = (uint8_t *)i0s[i];
        cpuPointers[i * 8 + 1] = (uint8_t *)i1s[i];
        cpuPointers[i * 8 + 2] = (uint8_t *)os[i];
        cpuPointers[i * 8 + 3] = (uint8_t *)(size_t)ns[i];
        cpuPointers[i * 8 + 4] = (uint8_t *)(size_t)ms[i];
        cpuPointers[i * 8 + 5] = (uint8_t *)(size_t)ks[i];
        cpuPointers[i * 8 + 6] = (uint8_t *)(size_t)i0Strides[i];
        cpuPointers[i * 8 + 7] = (uint8_t *)(size_t)i1Strides[i];
    }
    cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t *) * batch * 8, cudaMemcpyHostToDevice);
    FastllmMatMulTransBBatchKernel<128><<<batch, 128>>>(pointers, alpha);
    FastllmCudaFree(pointers);
    delete[] cpuPointers;
    DeviceSync();
    return true;
}

void LaunchFastllmGemmFp16Fp16(half *input, half *weight, half *output, half *bias, int n, int m, int k) {
    if (n == 1) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 1><<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 2) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 2><<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 3) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 3><<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 4) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 4><<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 5) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 5><<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 6) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 6><<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 7) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 7><<<k, 256>>>(input, weight, output, bias, m, k);
    } else {
        printf("Error: LaunchFastllmGemmFp16Fp16: n > 7.\n");
        exit(0);
    }
}

bool FastllmCudaHalfMatMulFloat16(const Data &input, Data &weight, const Data &bias, Data &output, int n, int m, int k) {
    if (weight.cudaData == nullptr || (weight.extraCudaHalfData.size() == 0 && bias.dims.size() > 0)) {
        half *cudaBiasData;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaBiasData, k * sizeof(half));
        if (bias.dims.size() > 0) {
            float *tempBiasData;
            state = cudaMalloc(&tempBiasData, k * sizeof(float));
            state = cudaMemcpy(tempBiasData, (uint8_t *)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
            int threadPerBlock = std::min(256, k);
            FastllmCudaFloat2HalfKernel<<<(k - 1) / threadPerBlock + 1, threadPerBlock>>>(tempBiasData, cudaBiasData, k);
            state = cudaFree(tempBiasData);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(half));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaHalfData.push_back((void *)cudaBiasData);
    }

    half *cudaInput = (half *)FastllmCudaPrepareInput(input);
    half *cudaOutput = (half *)FastllmCudaPrepareOutput(output);
    half *cudaBiasData = bias.cudaData == nullptr ? nullptr : static_cast<half *>(weight.extraCudaHalfData[0]);
    if (n < 8) {
        LaunchFastllmGemmFp16Fp16(cudaInput, (half *)weight.cudaData, cudaOutput, cudaBiasData, n, m, k);
    } else {

        auto fastllmCublasHandle = getFastllmCublasHandle();
        cublasStatus_t status;

#ifdef CUDA_NO_TENSOR_CORE
        float *cudaFp32Output = (float *)FastllmCudaMalloc(n * k * sizeof(float));
        cudaMemset(cudaFp32Output, 0.0f, n * k * sizeof(float));

        float alpha = 1.0f;
        float beta = 0.0f;

        cudaDataType A_dataType = cudaDataType::CUDA_R_16F;
        cudaDataType B_dataType = cudaDataType::CUDA_R_16F;
        cudaDataType C_dataType = cudaDataType::CUDA_R_32F;
        cudaDataType Compute_dataType = cudaDataType::CUDA_R_32F;

        status = cublasGemmEx(fastllmCublasHandle,
                              cublasOperation_t::CUBLAS_OP_T,
                              cublasOperation_t::CUBLAS_OP_N,
                              k,
                              n,
                              m,
                              &alpha,
                              (half *)weight.cudaData,
                              A_dataType,
                              m,
                              cudaInputdata,
                              B_dataType,
                              m,
                              &beta,
                              cudaFp32Output,
                              C_dataType,
                              k,
                              Compute_dataType,
                              static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#else
        half h_alpha = __float2half(1.0f);
        half h_beta = __float2half(0.0f);
        cudaDataType A_dataType = cudaDataType::CUDA_R_16F;
        cudaDataType B_dataType = cudaDataType::CUDA_R_16F;
        cudaDataType C_dataType = cudaDataType::CUDA_R_16F;
        cudaDataType Compute_dataType = cudaDataType::CUDA_R_16F;

        status = cublasGemmEx(fastllmCublasHandle,
                              cublasOperation_t::CUBLAS_OP_T,
                              cublasOperation_t::CUBLAS_OP_N,
                              k,
                              n,
                              m,
                              &h_alpha,
                              (half *)weight.cudaData,
                              A_dataType,
                              m,
                              cudaInput,
                              B_dataType,
                              m,
                              &h_beta,
                              cudaOutput,
                              C_dataType,
                              k,
                              Compute_dataType,
                              static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));

#endif

        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            exit(0);
        }

#ifdef CUDA_NO_TENSOR_CORE
        int len = n * k;
        int threadPerBlock = std::min(256, len);
        FastllmCudaFloat2HalfKernel<<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaFp32Output, cudaOutput, len);
        FastllmCudaFree(cudaFp32Output);
#endif
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel<<<n, 256>>>(cudaOutput, (half *)weight.extraCudaHalfData[0], k);
        }
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishInput(output, cudaOutput);
    return true;
}

void LaunchFastllmGemmFp16Int8(half *input, uint8_t *weight, half *output, half *bias, float *scales, uint8_t *zeros, int n, int m, int k) {
    for (int i = 0; i < n; i++) {
        FastllmGemvFp16Int8Kernel2<256, 1><<<k, 256>>>(input + i * m, weight, output + i * k, bias, scales, zeros, m, k);
    }
}

bool FastllmCudaHalfMatMulFloatInt8(const Data &input, Data &weight, const Data &bias, Data &output, int n, int m, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaHalfData.size() == 0) {
        weight.extraCudaHalfData.push_back((void *)weight.extraCudaData[0]);
        weight.extraCudaHalfData.push_back((void *)weight.extraCudaData[1]);

        half *cudaBiasData;
        cudaError_t state = cudaMalloc(&cudaBiasData, k * sizeof(half));
        if (bias.dims.size() > 0) {
            float *tempBiasData;
            state = cudaMalloc(&tempBiasData, k * sizeof(float));
            state = cudaMemcpy(tempBiasData, (uint8_t *)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
            int threadPerBlock = std::min(256, k);
            FastllmCudaFloat2HalfKernel<<<(k - 1) / threadPerBlock + 1, threadPerBlock>>>(tempBiasData, cudaBiasData, k);
            state = cudaFree(tempBiasData);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(half));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaHalfData.push_back((void *)cudaBiasData);
    }

    half *cudaInput = (half *)FastllmCudaPrepareInput(input);
    uint8_t *cudaWeightInput = (uint8_t *)weight.cudaData;
    half *cudaOutput = (half *)FastllmCudaPrepareOutput(output);
    half *cudaBiasData = bias.dims.size() > 0 ? (half *)weight.extraCudaHalfData[2] : nullptr;
    float *cudaScales = (float *)weight.extraCudaHalfData[0];
    uint8_t *cudaZeropoints = (uint8_t *)weight.extraCudaHalfData[1];

    if (n < 8) {
        LaunchFastllmGemmFp16Int8(cudaInput, cudaWeightInput, cudaOutput, cudaBiasData, cudaScales, cudaZeropoints, n, m, k);
    } else {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        cublasStatus_t status;

#ifdef CUDA_NO_TENSOR_CORE
        float alpha = 1.0, beta = 0.0;
        cudaDataType A_dataType = cudaDataType::CUDA_R_16F;
        cudaDataType B_dataType = cudaDataType::CUDA_R_16F;
        cudaDataType C_dataType = cudaDataType::CUDA_R_32F;
        cudaDataType Compute_dataType = cudaDataType::CUDA_R_32F;

        float *cudaFp32Output = (float *)FastllmCudaMalloc(n * m * sizeof(float));
        cudaMemset(cudaFp32Output, 0.0, n * m * sizeof(float));
#else
        half alpha = __float2half(1.0), beta = __float2half(0.0);
        cudaDataType A_dataType = cudaDataType::CUDA_R_16F;
        cudaDataType B_dataType = cudaDataType::CUDA_R_16F;
        cudaDataType C_dataType = cudaDataType::CUDA_R_16F;
        cudaDataType Compute_dataType = cudaDataType::CUDA_R_16F;
#endif
        int len = n * m;
        int threadPerBlock = std::min(256, len);
        len = k * m;

        half *dqWeight = (half *)FastllmCudaMalloc(k * m * sizeof(half));
        cudaMemset(dqWeight, 0, k * m * sizeof(half));

        FastllmCudaInt82HalfKernel<<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaWeightInput, cudaScales, cudaZeropoints, dqWeight, len, m);

#ifdef CUDA_NO_TENSOR_CORE
        status = cublasGemmEx(fastllmCublasHandle,
                              cublasOperation_t::CUBLAS_OP_T,
                              cublasOperation_t::CUBLAS_OP_N,
                              k,
                              n,
                              m,
                              &alpha,
                              dqWeight,
                              A_dataType,
                              m,
                              cudaInput,
                              B_dataType,
                              m,
                              &beta,
                              cudaFp32Output,
                              C_dataType,
                              k,
                              Compute_dataType,
                              static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#else
        status = cublasGemmEx(fastllmCublasHandle,
                              cublasOperation_t::CUBLAS_OP_T,
                              cublasOperation_t::CUBLAS_OP_N,
                              k,
                              n,
                              m,
                              &alpha,
                              dqWeight,
                              A_dataType,
                              m,
                              cudaInput,
                              B_dataType,
                              m,
                              &beta,
                              cudaOutput,
                              C_dataType,
                              k,
                              Compute_dataType,
                              static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#endif
        if (status != CUBLAS_STATUS_SUCCESS) {
            FastllmCudaFree(dqWeight);
            printf("Error: cublas error.\n");
            throw("cublas error");
            exit(0);
        }

#ifdef CUDA_NO_TENSOR_CORE
        int len = n * k;
        int threadPerBlock = std::min(256, len);
        FastllmCudaFloat2HalfKernel<<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaFp32Output, cudaOutput, len);
        FastllmCudaFree(cudaFp32Output);
#endif
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel<<<n, 256>>>(cudaOutput, cudaBiasData, k);
        }

        FastllmCudaFree(dqWeight);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishInput(output, cudaOutput);
    return true;
}

bool FastllmCudaMatMulFloat32(const Data &input, Data &weight, const Data &bias, Data &output, int n, int m, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        float *cudaBiasData;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaBiasData, k * sizeof(float));
        if (bias.dims.size() > 0) {
            state = cudaMemcpy(cudaBiasData, (uint8_t *)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(float));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaData.push_back((void *)cudaBiasData);
    }

    float *cudaBiasData = (float *)weight.extraCudaData[0];
    float *cudaInput = (float *)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *)FastllmCudaPrepareOutput(output);

    if (n > 1) {
        float h_alpha = 1.0, h_beta = 0.0;
        auto fastllmCublasHandle = getFastllmCublasHandle();
        // cudaDeviceSynchronize();
        cudaDataType_t AType = CUDA_R_32F, BType = CUDA_R_32F, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;
        cublasStatus_t status;

        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              k,
                              n,
                              m,
                              &h_alpha,
                              weight.cudaData,
                              AType,
                              m,
                              cudaInput,
                              BType,
                              m,
                              &h_beta,
                              cudaOutput,
                              CType,
                              k,
                              ComputeType,
                              static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            FastllmCudaFinishInput(input, cudaInput);
            FastllmCudaFinishOutput(output, cudaOutput);
            exit(0);
        }

        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel<<<n, 256>>>(cudaOutput, (float *)weight.extraCudaData[0], k);
        }
    } else {
        FastllmGemvFp32Fp32Kernel2<256, 1><<<k, 256>>>(cudaInput, (float *)weight.cudaData, cudaOutput, cudaBiasData, m, k);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaHalfMatMulFloat32(const Data &input, Data &weight, const Data &bias, Data &output, int n, int m, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        float *cudaBiasData;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaBiasData, k * sizeof(float));
        if (bias.dims.size() > 0) {
            state = cudaMemcpy(cudaBiasData, (uint8_t *)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(float));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaData.push_back((void *)cudaBiasData);
    }

    float *cudaBiasData = (float *)weight.extraCudaData[0];
    float *cudaInput = (float *)FastllmCudaMalloc(input.Count(0) * sizeof(float));
    float *cudaOutput = (float *)FastllmCudaMalloc(output.Count(0) * sizeof(float));
    int inputLen = input.Count(0);
    FastllmCudaHalf2FloatKernel<<<(inputLen - 1) / 256 + 1, 256>>>((half *)input.cudaData, cudaInput, inputLen);

    if (n > 1) {
        float h_alpha = 1.0, h_beta = 0.0;
        auto fastllmCublasHandle = getFastllmCublasHandle();
        // cudaDeviceSynchronize();
        cudaDataType_t AType = CUDA_R_32F, BType = CUDA_R_32F, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;
        cublasStatus_t status;

        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              k,
                              n,
                              m,
                              &h_alpha,
                              weight.cudaData,
                              AType,
                              m,
                              cudaInput,
                              BType,
                              m,
                              &h_beta,
                              cudaOutput,
                              CType,
                              k,
                              ComputeType,
                              static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            FastllmCudaFinishInput(input, cudaInput);
            FastllmCudaFinishOutput(output, cudaOutput);
            exit(0);
        }

        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel<<<n, 256>>>(cudaOutput, (float *)weight.extraCudaData[0], k);
        }
    } else {
        FastllmGemvFp32Fp32Kernel2<256, 1><<<k, 256>>>(cudaInput, (float *)weight.cudaData, cudaOutput, cudaBiasData, m, k);
    }

    int outputLen = output.Count(0);
    FastllmCudaFloat2HalfKernel<<<(outputLen - 1) / 256 + 1, 256>>>(cudaOutput, (half *)output.cudaData, outputLen);
    DeviceSync();
    return true;
}

void LaunchFastllmGemmFp32Fp16(float *input, half *weight, float *output, float *bias, int n, int m, int k) {
    if (n == 1) {
        FastllmGemvFp32Fp16Kernel2MultiRow<256, 1><<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 2) {
        FastllmGemvFp32Fp16Kernel2MultiRow<256, 2><<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 3) {
        FastllmGemvFp32Fp16Kernel2MultiRow<256, 3><<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 4) {
        FastllmGemvFp32Fp16Kernel2MultiRow<256, 4><<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 5) {
        FastllmGemvFp32Fp16Kernel2MultiRow<256, 5><<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 6) {
        FastllmGemvFp32Fp16Kernel2MultiRow<256, 6><<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 7) {
        FastllmGemvFp32Fp16Kernel2MultiRow<256, 7><<<k, 256>>>(input, weight, output, bias, m, k);
    } else {
        for (int i = 0; i < n; i++) {
            FastllmGemvFp32Fp16Kernel2MultiRow<256, 1><<<k, 256>>>(input + i * m, weight, output + i * k, bias, m, k);
        }
        return;

        printf("Error: LaunchFastllmGemmFp32Fp16: n > 7.\n");
        exit(0);
    }
}

void LaunchFastllmGemmFp32Int8(float *input, uint8_t *weight, float *output, float *bias, float *scales, uint8_t *zeros, int n, int m, int k) {
    for (int i = 0; i < n; i++) {
        FastllmGemvInt8Kernel2<256, 1><<<k, 256>>>(input + i * m, weight, output + i * k, bias, scales, zeros, m, k);
    }
}

bool FastllmCudaMatMulFloatInt8(const Data &input, Data &weight, const Data &bias, Data &output, int n, int m, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        float *cudaScales;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaScales, k * sizeof(float));
        state = cudaMemcpy(cudaScales, weight.scales.data(), k * sizeof(float), cudaMemcpyHostToDevice);
        weight.extraCudaData.push_back((void *)cudaScales);

        uint8_t *cudaZeropoints;
        state = cudaMalloc(&cudaZeropoints, k);
        uint8_t *zeropoints = new uint8_t[k];
        for (int i = 0; i < k; i++) {
            zeropoints[i] = weight.zeros[i];
        }
        state = cudaMemcpy(cudaZeropoints, zeropoints, k, cudaMemcpyHostToDevice);
        delete[] zeropoints;
        weight.extraCudaData.push_back((void *)cudaZeropoints);

        float *cudaBiasData;
        state = cudaMalloc(&cudaBiasData, k * sizeof(float));
        if (bias.dims.size() > 0) {
            state = cudaMemcpy(cudaBiasData, (uint8_t *)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(float));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaData.push_back((void *)cudaBiasData);
    }
    float *cudaScales = (float *)weight.extraCudaData[0];
    uint8_t *cudaZeropoints = (uint8_t *)weight.extraCudaData[1];
    float *cudaBiasData = (float *)weight.extraCudaData[2];

    float *cudaInput = (float *)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *)FastllmCudaPrepareOutput(output);

    if (n >= 8) {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        half *cudaFp16Input, *cudaFp16Output, *cudaFp16Weight;
#ifdef CUDA_NO_TENSOR_CORE
        cudaFp16Input = (half *)FastllmCudaMalloc(n * m * sizeof(half));
        cudaFp16Weight = (half *)FastllmCudaMalloc(k * m * sizeof(half));
        float h_alpha = 1.0, h_beta = 0.0;
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;
#else
        cudaFp16Input = (half *)FastllmCudaMalloc(n * m * sizeof(half));
        cudaFp16Output = (half *)FastllmCudaMalloc(n * k * sizeof(half));
        cudaFp16Weight = (half *)FastllmCudaMalloc(k * m * sizeof(half));

        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
#endif
        cublasStatus_t status;

        int len = n * m;
        int threadPerBlock = std::min(256, len);
        FastllmCudaFloat2HalfKernel<<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaFp16Input, len);

        len = k * m;
#ifdef CUDA_NO_TENSOR_CORE
        int gridSize = (len - 1) / (threadPerBlock * ST128_FP16_COUNT) + 1;
        FastllmCudaInt82HalfKernel<<<gridSize, threadPerBlock>>>((uint8_t *)weight.cudaData, cudaScales, cudaZeropoints, cudaFp16Weight, len, m);

        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              k,
                              n,
                              m,
                              &h_alpha,
                              cudaFp16Weight,
                              AType,
                              m,
                              cudaFp16Input,
                              BType,
                              m,
                              &h_beta,
                              cudaOutput,
                              CType,
                              k,
                              ComputeType,
                              static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#else
        FastllmCudaInt82HalfKernel<<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(
            (uint8_t *)weight.cudaData, cudaScales, cudaZeropoints, cudaFp16Weight, len, m);

        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              k,
                              n,
                              m,
                              &h_alpha,
                              cudaFp16Weight,
                              AType,
                              m,
                              cudaFp16Input,
                              BType,
                              m,
                              &h_beta,
                              cudaFp16Output,
                              CType,
                              k,
                              ComputeType,
                              static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#endif
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            exit(0);
        }

        len = n * k;
#ifdef CUDA_NO_TENSOR_CORE
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel<<<n, 256>>>(cudaOutput, cudaBiasData, k);
        }
        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Weight);
#else
        FastllmCudaHalf2FloatKernel<<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaFp16Output, cudaOutput, len);
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel<<<n, 256>>>(cudaOutput, cudaBiasData, k);
        }

        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Output);
        FastllmCudaFree(cudaFp16Weight);
#endif
    } else {
        LaunchFastllmGemmFp32Int8(cudaInput, (uint8_t *)weight.cudaData, cudaOutput, cudaBiasData, cudaScales, cudaZeropoints, n, m, k);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

void LaunchFastllmGemvInt4Kernel2(float *input, uint8_t *weight, float *output, float *bias, float *scales, uint8_t *zeros, int n, int m, int k) {
    for (int i = 0; i < n; i++) {
        FastllmGemvInt4Kernel2<256, 1><<<k, 256>>>(input + i * m, weight, output + i * k, bias, scales, zeros, m, k);
    }
}

bool FastllmCudaMatMulFloatInt4(const Data &input, Data &weight, const Data &bias, Data &output, int n, int m, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        float *cudaScales;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaScales, k * sizeof(float));
        state = cudaMemcpy(cudaScales, weight.scales.data(), k * sizeof(float), cudaMemcpyHostToDevice);
        weight.extraCudaData.push_back((void *)cudaScales);

        uint8_t *cudaZeropoints;
        state = cudaMalloc(&cudaZeropoints, k);
        uint8_t *zeropoints = new uint8_t[k];
        for (int i = 0; i < k; i++) {
            zeropoints[i] = weight.perChannelsConfigs[i].zeroPoint;
        }
        state = cudaMemcpy(cudaZeropoints, zeropoints, k, cudaMemcpyHostToDevice);
        delete[] zeropoints;
        weight.extraCudaData.push_back((void *)cudaZeropoints);

        float *cudaBiasData;
        state = cudaMalloc(&cudaBiasData, k * sizeof(float));
        if (bias.dims.size() > 0) {
            state = cudaMemcpy(cudaBiasData, (uint8_t *)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(float));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaData.push_back((void *)cudaBiasData);
    }

    float *cudaScales = (float *)weight.extraCudaData[0];
    uint8_t *cudaZeropoints = (uint8_t *)weight.extraCudaData[1];
    float *cudaBiasData = (float *)weight.extraCudaData[2];

    float *cudaInput = (float *)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *)FastllmCudaPrepareOutput(output);
    LaunchFastllmGemvInt4Kernel2(cudaInput, (uint8_t *)weight.cudaData, cudaOutput, cudaBiasData, cudaScales, cudaZeropoints, n, m, k);

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

void LaunchFastllmGemmFp32Int4NoZero(float *input, uint8_t *weight, float *output, float *bias, float *scales, float *mins, int n, int m, int k) {
    /* for (int i = 0; i < n; i++) {
         FastllmGemvInt4NoZeroKernel1<64, 1> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, scales, mins, m, k);
     }
     return;*/
    if (n == 1) {
        FastllmGemvInt4NoZeroKernel1MultiRow<64, 1><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k);
    } else if (n == 2) {
        FastllmGemvInt4NoZeroKernel1MultiRow<64, 2><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k);
    } else if (n == 3) {
        FastllmGemvInt4NoZeroKernel1MultiRow<64, 3><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k);
    } else if (n == 4) {
        FastllmGemvInt4NoZeroKernel1MultiRow<64, 4><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k);
    } else if (n == 5) {
        FastllmGemvInt4NoZeroKernel1MultiRow<64, 5><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k);
    } else if (n == 6) {
        FastllmGemvInt4NoZeroKernel1MultiRow<64, 6><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k);
    } else if (n == 7) {
        FastllmGemvInt4NoZeroKernel1MultiRow<64, 7><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k);
    } else {
        for (int i = 0; i < n; i++) {
            FastllmGemvInt4NoZeroKernel1<64, 1><<<k, 64>>>(input + i * m, weight, output + i * k, bias, scales, mins, m, k);
        }
        return;
    }
}

bool FastllmCudaMatMulFloatInt4NoZero(const Data &input, Data &weight, const Data &bias, Data &output, int n, int m, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        float *cudaScales;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaScales, k * sizeof(float));
        state = cudaMemcpy(cudaScales, weight.scales.data(), k * sizeof(float), cudaMemcpyHostToDevice);
        weight.extraCudaData.push_back((void *)cudaScales);

        float *cudaMins;
        state = cudaMalloc(&cudaMins, k * sizeof(float));
        float *mins = new float[k];
        for (int i = 0; i < k; i++) {
            mins[i] = weight.mins[i];
        }
        state = cudaMemcpy(cudaMins, mins, k * sizeof(float), cudaMemcpyHostToDevice);
        delete[] mins;
        weight.extraCudaData.push_back((void *)cudaMins);

        float *cudaBiasData;
        state = cudaMalloc(&cudaBiasData, k * sizeof(float));
        if (bias.dims.size() > 0) {
            state = cudaMemcpy(cudaBiasData, (uint8_t *)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(float));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaData.push_back((void *)cudaBiasData);
    }

    float *cudaScales = (float *)weight.extraCudaData[0];
    float *cudaMins = (float *)weight.extraCudaData[1];
    float *cudaBiasData = (float *)weight.extraCudaData[2];

    float *cudaInput = (float *)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *)FastllmCudaPrepareOutput(output);

    if (n >= 16) {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        half *cudaFp16Input, *cudaFp16Output, *cudaFp16Weight;
#ifdef CUDA_NO_TENSOR_CORE
        cudaFp16Input = (half *)FastllmCudaMalloc(n * m * sizeof(half));
        cudaFp16Weight = (half *)FastllmCudaMalloc(k * m * sizeof(half));

        float h_alpha = 1.0, h_beta = 0.0;
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;
#else
        cudaFp16Input = (half *)FastllmCudaMalloc(n * m * sizeof(half));
        cudaFp16Output = (half *)FastllmCudaMalloc(n * k * sizeof(half));
        cudaFp16Weight = (half *)FastllmCudaMalloc(k * m * sizeof(half));

        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
#endif
        cublasStatus_t status;

        int len = n * m;
        int threadPerBlock = std::min(256, len);
        FastllmCudaFloat2HalfKernel<<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaFp16Input, len);

        len = k * m;
        int gridSize = (len - 1) / (threadPerBlock * 4) + 1;
        FastllmCudaInt42HalfKernel<<<gridSize, threadPerBlock>>>((uint8_t *)weight.cudaData, cudaScales, cudaMins, cudaFp16Weight, len, m);

#ifdef CUDA_NO_TENSOR_CORE
        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              k,
                              n,
                              m,
                              &h_alpha,
                              cudaFp16Weight,
                              AType,
                              m,
                              cudaFp16Input,
                              BType,
                              m,
                              &h_beta,
                              cudaOutput,
                              CType,
                              k,
                              ComputeType,
                              static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#else
        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              k,
                              n,
                              m,
                              &h_alpha,
                              cudaFp16Weight,
                              AType,
                              m,
                              cudaFp16Input,
                              BType,
                              m,
                              &h_beta,
                              cudaFp16Output,
                              CType,
                              k,
                              ComputeType,
                              static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#endif
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            exit(0);
        }

        len = n * k;
#ifdef CUDA_NO_TENSOR_CORE
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel<<<n, 256>>>(cudaOutput, cudaBiasData, k);
        }
        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Weight);
#else
        FastllmCudaHalf2FloatKernel<<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaFp16Output, cudaOutput, len);
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel<<<n, 256>>>(cudaOutput, cudaBiasData, k);
        }

        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Output);
        FastllmCudaFree(cudaFp16Weight);
#endif
    } else {
        LaunchFastllmGemmFp32Int4NoZero(cudaInput, (uint8_t *)weight.cudaData, cudaOutput, cudaBiasData, cudaScales, cudaMins, n, m, k);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

void LaunchFastllmGemmFp16Int4NoZero(half *input, uint8_t *weight, half *output, half *bias, float *scales, float *mins, int n, int m, int k) {
    if (n == 1) {
        FastllmGemvFp16Int4NoZeroKernel1MultiRow<64, 1><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k);
    } else if (n == 2) {
        FastllmGemvFp16Int4NoZeroKernel1MultiRow<64, 2><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k);
    } else if (n == 3) {
        FastllmGemvFp16Int4NoZeroKernel1MultiRow<64, 3><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k);
    } else if (n == 4) {
        FastllmGemvFp16Int4NoZeroKernel1MultiRow<64, 4><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k);
    } else if (n == 5) {
        FastllmGemvFp16Int4NoZeroKernel1MultiRow<64, 5><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k);
    } else if (n == 6) {
        FastllmGemvFp16Int4NoZeroKernel1MultiRow<64, 6><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k);
    } else if (n == 7) {
        FastllmGemvFp16Int4NoZeroKernel1MultiRow<64, 7><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k);
    } else {
        for (int i = 0; i < n; i++) {
            FastllmGemvFp16Int4NoZeroKernel2<64, 1><<<k / 1, 64>>>(input + i * m, weight, output + i * k, bias, scales, mins, m, k);
        }
    }
}

bool FastllmCudaHalfMatMulFloatInt4NoZero(const Data &input, Data &weight, const Data &bias, Data &output, int n, int m, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaHalfData.size() == 0) {
        weight.extraCudaHalfData.push_back((void *)weight.extraCudaData[0]);
        weight.extraCudaHalfData.push_back((void *)weight.extraCudaData[1]);

        half *cudaBiasData;
        cudaError_t state = cudaMalloc(&cudaBiasData, k * sizeof(half));
        if (bias.dims.size() > 0) {
            float *tempBiasData;
            state = cudaMalloc(&tempBiasData, k * sizeof(float));
            state = cudaMemcpy(tempBiasData, (uint8_t *)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
            int threadPerBlock = std::min(256, k);
            FastllmCudaFloat2HalfKernel<<<(k - 1) / threadPerBlock + 1, threadPerBlock>>>(tempBiasData, cudaBiasData, k);
            state = cudaFree(tempBiasData);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(half));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaHalfData.push_back((void *)cudaBiasData);
    }
    float *cudaScales = (float *)weight.extraCudaHalfData[0];
    float *cudaMins = (float *)weight.extraCudaHalfData[1];

    half *cudaInput = (half *)FastllmCudaPrepareInput(input);
    half *cudaOutput = (half *)FastllmCudaPrepareOutput(output);

    if (n >= 8) {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        half *cudaFp16Weight;

        cudaFp16Weight = (half *)FastllmCudaMalloc(k * m * sizeof(half));

#ifdef CUDA_NO_TENSOR_CORE
        float *cudaFp32Output = (float *)FastllmCudaMalloc(n * k * sizeof(float));
        float h_alpha = 1.0, h_beta = 0.0;
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;
#else
        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
#endif
        cublasStatus_t status;

        int len = n * m;
        int threadPerBlock = std::min(256, len);

        len = k * m;
        int gridSize = (len - 1) / (threadPerBlock * 4) + 1;
        FastllmCudaInt42HalfKernel<<<gridSize, threadPerBlock>>>((uint8_t *)weight.cudaData, cudaScales, cudaMins, cudaFp16Weight, len, m);

#ifdef CUDA_NO_TENSOR_CORE
        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              k,
                              n,
                              m,
                              &h_alpha,
                              cudaFp16Weight,
                              AType,
                              m,
                              cudaInput,
                              BType,
                              m,
                              &h_beta,
                              cudaFp32Output,
                              CType,
                              k,
                              ComputeType,
                              static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#else
        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              k,
                              n,
                              m,
                              &h_alpha,
                              cudaFp16Weight,
                              AType,
                              m,
                              cudaInput,
                              BType,
                              m,
                              &h_beta,
                              cudaOutput,
                              CType,
                              k,
                              ComputeType,
                              static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#endif

        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            exit(0);
        }

#ifdef CUDA_NO_TENSOR_CORE
        len = n * k;
        FastllmCudaFloat2HalfKernel<<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaFp32Output, cudaOutput, len);
        FastllmCudaFree(cudaFp32Output);
#endif
        if (bias.dims.size() > 0) {
            half *cudaBiasData = (half *)weight.extraCudaHalfData[2];
            FastllmCudaBiasKernel<<<n, 256>>>(cudaOutput, cudaBiasData, k);
        }

        FastllmCudaFree(cudaFp16Weight);
    } else {
        half *cudaBiasData = (half *)weight.extraCudaHalfData[2];
        LaunchFastllmGemmFp16Int4NoZero(cudaInput, (uint8_t *)weight.cudaData, cudaOutput, cudaBiasData, cudaScales, cudaMins, n, m, k);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

void LaunchFastllmGemmFp16Int4Group(
    half *input, uint8_t *weight, half *output, half *bias, half *scales, half *mins, int n, int m, int k, int group, int groupCnt) {
    if (n == 1) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 1><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 2) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 2><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 3) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 3><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 4) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 4><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 5) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 5><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 6) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 6><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 7) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 7><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 8) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 8><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 9) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 9><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 10) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 10><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 11) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 11><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 12) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 12><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 13) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 13><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 14) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 14><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 15) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 15><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 16) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 16><<<k, 64>>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else {
        for (int i = 0; i < n; i++) {
            FastllmGemvHalfInt4GroupKernelMultiRow<64, 1>
                <<<k, 64>>>(input + i * m, weight, output + i * k, bias, scales, mins, m, k, group, groupCnt);
        }
        return;
    }
}

void LaunchFastllmGemmFp32Int4Group(
    float *input, uint8_t *weight, float *output, float *bias, half *scales, half *mins, int n, int m, int k, int group, int groupCnt) {
    for (int i = 0; i < n; i++) {
#ifdef CUDA_NO_TENSOR_CORE
        FastllmGemvInt4GroupKernel3<64, 4><<<k / 4, 64>>>(input + i * m, weight, output + i * k, bias, scales, mins, m, k, group, groupCnt);
#else
        FastllmGemvInt4GroupKernel2<64, 4><<<k / 4, 64>>>(input + i * m, weight, output + i * k, bias, scales, mins, m, k, group, groupCnt);
#endif
    }
}

bool FastllmCudaMatMulFloatInt4Group(const Data &input, Data &weight, const Data &bias, Data &output, int n, int m, int k) {
    int group = weight.group, groupCnt = weight.groupCnt;
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        half *cudaScales;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaScales, k * group * sizeof(half));
        half *scales = new half[k * group];
        for (int i = 0; i < k * group; i++) {
            scales[i] = (half)weight.scales[i];
        }
        state = cudaMemcpy(cudaScales, scales, k * group * sizeof(half), cudaMemcpyHostToDevice);
        weight.extraCudaData.push_back((void *)cudaScales);
        delete[] scales;

        half *cudaMins;
        state = cudaMalloc(&cudaMins, k * group * sizeof(half));
        half *mins = new half[k * group];
        for (int i = 0; i < k * group; i++) {
            mins[i] = (half)weight.mins[i];
        }
        state = cudaMemcpy(cudaMins, mins, k * group * sizeof(half), cudaMemcpyHostToDevice);
        delete[] mins;
        weight.extraCudaData.push_back((void *)cudaMins);

        float *cudaBiasData;
        state = cudaMalloc(&cudaBiasData, k * sizeof(float));
        if (bias.dims.size() > 0) {
            state = cudaMemcpy(cudaBiasData, (uint8_t *)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(float));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaData.push_back((void *)cudaBiasData);
    }

    half *cudaScales = (half *)weight.extraCudaData[0];
    half *cudaMins = (half *)weight.extraCudaData[1];
    float *cudaBiasData = (float *)weight.extraCudaData[2];

    float *cudaInput = (float *)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *)FastllmCudaPrepareOutput(output);
    if (n >= 8) {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        half *cudaFp16Input, *cudaFp16Output, *cudaFp16Weight;
#ifdef CUDA_NO_TENSOR_CORE
        cudaFp16Input = (half *)FastllmCudaMalloc(n * m * sizeof(half));
        cudaFp16Weight = (half *)FastllmCudaMalloc(k * m * sizeof(half));

        float h_alpha = 1.0, h_beta = 0.0;
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;
#else
        cudaFp16Input = (half *)FastllmCudaMalloc(n * m * sizeof(half));
        cudaFp16Output = (half *)FastllmCudaMalloc(n * k * sizeof(half));
        cudaFp16Weight = (half *)FastllmCudaMalloc(k * m * sizeof(half));

        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
#endif
        cublasStatus_t status;

        int len = n * m;
        int threadPerBlock = std::min(256, len);
        FastllmCudaFloat2HalfKernel<<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaFp16Input, len);

        len = k * m;
        FastllmCudaInt4Group2HalfKernel<<<k, 64>>>((uint8_t *)weight.cudaData, cudaScales, cudaMins, cudaFp16Weight, k, m, group, groupCnt);

#ifdef CUDA_NO_TENSOR_CORE
        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              k,
                              n,
                              m,
                              &h_alpha,
                              cudaFp16Weight,
                              AType,
                              m,
                              cudaFp16Input,
                              BType,
                              m,
                              &h_beta,
                              cudaOutput,
                              CType,
                              k,
                              ComputeType,
                              static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#else
        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              k,
                              n,
                              m,
                              &h_alpha,
                              cudaFp16Weight,
                              AType,
                              m,
                              cudaFp16Input,
                              BType,
                              m,
                              &h_beta,
                              cudaFp16Output,
                              CType,
                              k,
                              ComputeType,
                              static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#endif
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error. status = %d\n", status);
            throw("cublas error");
            exit(0);
        }

        len = n * k;
#ifdef CUDA_NO_TENSOR_CORE
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel<<<n, 256>>>(cudaOutput, cudaBiasData, k);
        }
        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Weight);
#else
        FastllmCudaHalf2FloatKernel<<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaFp16Output, cudaOutput, len);
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel<<<n, 256>>>(cudaOutput, cudaBiasData, k);
        }

        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Output);
        FastllmCudaFree(cudaFp16Weight);
#endif
    } else {
        LaunchFastllmGemmFp32Int4Group(
            cudaInput, (uint8_t *)weight.cudaData, cudaOutput, cudaBiasData, cudaScales, cudaMins, n, m, k, group, groupCnt);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaHalfMatMulFloatInt4Group(const Data &input, Data &weight, const Data &bias, Data &output, int n, int m, int k) {
    int group = weight.group, groupCnt = weight.groupCnt;
    if (weight.cudaData == nullptr || weight.extraCudaHalfData.size() == 0) {
        weight.extraCudaHalfData.push_back((void *)weight.extraCudaData[0]);
        weight.extraCudaHalfData.push_back((void *)weight.extraCudaData[1]);

        half *cudaBiasData;
        cudaError_t state = cudaMalloc(&cudaBiasData, k * sizeof(half));
        if (bias.dims.size() > 0) {
            float *tempBiasData;
            state = cudaMalloc(&tempBiasData, k * sizeof(float));
            state = cudaMemcpy(tempBiasData, (uint8_t *)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
            int threadPerBlock = std::min(256, k);
            FastllmCudaFloat2HalfKernel<<<(k - 1) / threadPerBlock + 1, threadPerBlock>>>(tempBiasData, cudaBiasData, k);
            state = cudaFree(tempBiasData);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(half));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaHalfData.push_back((void *)cudaBiasData);
    }
    half *cudaScales = (half *)weight.extraCudaHalfData[0];
    half *cudaMins = (half *)weight.extraCudaHalfData[1];

    half *cudaInput = (half *)FastllmCudaPrepareInput(input);
    half *cudaOutput = (half *)FastllmCudaPrepareOutput(output);

    if (n > 16) {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        half *cudaFp16Weight;
        cudaFp16Weight = (half *)FastllmCudaMalloc(k * m * sizeof(half));

#ifdef CUDA_NO_TENSOR_CORE
        float *cudaFp32Output = (float *)FastllmCudaMalloc(n * k * sizeof(float));
        float h_alpha = 1.0, h_beta = 0.0;
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;
#else
        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
#endif
        cublasStatus_t status;

        int len = n * m;
        int threadPerBlock = std::min(256, len);

        len = k * m;

        FastllmCudaInt4Group2HalfKernel<<<k, 256>>>((uint8_t *)weight.cudaData, cudaScales, cudaMins, cudaFp16Weight, k, m, group, groupCnt);

#ifdef CUDA_NO_TENSOR_CORE
        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              k,
                              n,
                              m,
                              &h_alpha,
                              cudaFp16Weight,
                              AType,
                              m,
                              cudaInput,
                              BType,
                              m,
                              &h_beta,
                              cudaFp32Output,
                              CType,
                              k,
                              ComputeType,
                              static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#else
        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              k,
                              n,
                              m,
                              &h_alpha,
                              cudaFp16Weight,
                              AType,
                              m,
                              cudaInput,
                              BType,
                              m,
                              &h_beta,
                              cudaOutput,
                              CType,
                              k,
                              ComputeType,
                              static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));

#endif
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error. status = %d\n", status);
            throw("cublas error");
            exit(0);
        }

#ifdef CUDA_NO_TENSOR_CORE
        len = n * k;
        FastllmCudaFloat2HalfKernel<<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaFp32Output, cudaOutput, len);
        FastllmCudaFree(cudaFp32Output);
#endif
        if (bias.dims.size() > 0) {
            half *cudaBiasData = (half *)weight.extraCudaHalfData[2];
            FastllmCudaBiasKernel<<<n, 256>>>(cudaOutput, cudaBiasData, k);
        }

        FastllmCudaFree(cudaFp16Weight);
    } else {
        half *cudaBiasData = (half *)weight.extraCudaHalfData[2];
        LaunchFastllmGemmFp16Int4Group(
            cudaInput, (uint8_t *)weight.cudaData, cudaOutput, cudaBiasData, cudaScales, cudaMins, n, m, k, group, groupCnt);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

void LaunchFastllmGemmFp32FP8E4M3(
    float *input, uint8_t *weight, float *output, float *bias, float *scales, int n, int m, int k, int blockM, int blockK) {
    if (n == 1) {
        FastllmGemvFP8E4M3Kernel1MultiRow<64, 1><<<k, 64>>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 2) {
        FastllmGemvFP8E4M3Kernel1MultiRow<64, 2><<<k, 64>>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 3) {
        FastllmGemvFP8E4M3Kernel1MultiRow<64, 3><<<k, 64>>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 4) {
        FastllmGemvFP8E4M3Kernel1MultiRow<64, 4><<<k, 64>>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 5) {
        FastllmGemvFP8E4M3Kernel1MultiRow<64, 5><<<k, 64>>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 6) {
        FastllmGemvFP8E4M3Kernel1MultiRow<64, 6><<<k, 64>>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 7) {
        FastllmGemvFP8E4M3Kernel1MultiRow<64, 7><<<k, 64>>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else {
        int i = 0;
        for (; i + 7 < n; i += 8) {
            FastllmGemvFP8E4M3Kernel1MultiRow<64, 8><<<k, 64>>>(input + i * m, weight, output + i * k, bias, scales, m, k, blockM, blockK);
        }
        for (; i < n; i++) {
            FastllmGemvFP8E4M3Kernel1MultiRow<64, 1><<<k, 64>>>(input + i * m, weight, output + i * k, bias, scales, m, k, blockM, blockK);
        }
        return;
    }
}

bool FastllmCudaMatMulFloatFP8E4M3(const Data &input, Data &weight, const Data &bias, Data &output, int n, int m, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        float *cudaScales;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaScales, weight.scales.size() * sizeof(float));
        state = cudaMemcpy(cudaScales, weight.scales.data(), weight.scales.size() * sizeof(float), cudaMemcpyHostToDevice);
        weight.extraCudaData.push_back((void *)cudaScales);

        float *cudaBiasData;
        state = cudaMalloc(&cudaBiasData, k * sizeof(float));
        if (bias.dims.size() > 0) {
            state = cudaMemcpy(cudaBiasData, (uint8_t *)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(float));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaData.push_back((void *)cudaBiasData);
    }

    float *cudaScales = (float *)weight.extraCudaData[0];
    float *cudaBiasData = (float *)weight.extraCudaData[1];

    float *cudaInput = (float *)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *)FastllmCudaPrepareOutput(output);

    if (n >= 1e9) {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        half *cudaFp16Input, *cudaFp16Output, *cudaFp16Weight;

        cudaFp16Input = (half *)FastllmCudaMalloc(n * m * sizeof(half));
        cudaFp16Output = (half *)FastllmCudaMalloc(n * k * sizeof(half));
        cudaFp16Weight = (half *)FastllmCudaMalloc(k * m * sizeof(half));

        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
        cublasStatus_t status;

        int len = n * m;
        int threadPerBlock = std::min(256, len);
        FastllmCudaFloat2HalfKernel<<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaFp16Input, len);

        len = k * m;
        /* FastllmCudaInt42HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((uint8_t *) weight.cudaData,
                                                                                         cudaScales,
                                                                                         cudaMins,
                                                                                         cudaFp16Weight, len, m);*/
        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              k,
                              n,
                              m,
                              &h_alpha,
                              cudaFp16Weight,
                              AType,
                              m,
                              cudaFp16Input,
                              BType,
                              m,
                              &h_beta,
                              cudaFp16Output,
                              CType,
                              k,
                              ComputeType,
                              static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            exit(0);
        }

        len = n * k;
        FastllmCudaHalf2FloatKernel<<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaFp16Output, cudaOutput, len);
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel<<<n, 256>>>(cudaOutput, cudaBiasData, k);
        }

        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Output);
        FastllmCudaFree(cudaFp16Weight);
    } else {
        LaunchFastllmGemmFp32FP8E4M3(
            cudaInput, (uint8_t *)weight.cudaData, cudaOutput, cudaBiasData, cudaScales, n, m, k, weight.blockM, weight.blockK);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

static std::map<int, cublasHandle_t> s_fastllmCublasHandleMap;
cublasHandle_t getFastllmCublasHandle() {
    int id = -1;
    cudaGetDevice(&id);
    if (s_fastllmCublasHandleMap.find(id) != s_fastllmCublasHandleMap.end()) {
        return s_fastllmCublasHandleMap[id];
    }

    cublasHandle_t handler = nullptr;
    auto status = cublasCreate(&handler);
    if (status != cublasStatus_t::CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS initialization failed\n");
        return nullptr;
    }

    s_fastllmCublasHandleMap[id] = handler;

    return handler;
}