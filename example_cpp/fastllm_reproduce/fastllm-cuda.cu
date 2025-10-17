#include "fastllm-cuda.cuh"
#include <mma.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

using namespace nvcuda;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
#define CUDA_NO_TENSOR_CORE
#endif

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
