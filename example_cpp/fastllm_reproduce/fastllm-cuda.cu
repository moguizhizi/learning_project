#include "fastllm-cuda.cuh"

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
    FastllmSoftmaxKernelInner1Func<THREAD_PER_BLOCK>(input + o * channels, output + o * channels, o + base + 1, nullptr, nullptr);
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
