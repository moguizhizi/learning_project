#include <curand.h>

#include <cub/cub.cuh>

#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }
#define CURAND_CHECK(callstr)                                                                    \
    {                                                                                            \
        curandStatus_t error_code = callstr;                                                     \
        if (error_code != CURAND_STATUS_SUCCESS) {                                               \
            std::cerr << "cuRAND error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                           \
        }                                                                                        \
    }

enum SOFTMAX_TYPE {
    SOFTMAX_TYPE_NAIVE,
    SOFTMAX_TYPE_SAFE,
    SOFTMAX_TYPE_ONLINE,
};

struct __align__(8) MD {
    float m;
    float d;
};

__device__ __forceinline__ float max_op(float a, float b) {
    return fmaxf(a, b);
}

__device__ __forceinline__ MD reduce_max_op(MD a, MD b) {
    bool bigger = (a.m > b.m);
    MD bigger_md = bigger ? a : b;
    MD smaller_md = bigger ? b : a;
    MD res;

    res.d = bigger_md.d + smaller_md.d * __expf(smaller_md.m - bigger_md.m);
    res.m = bigger_md.m;

    return res;
}

void fil_random_values(float *x, int count) {
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, curandRngType_t::CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    CURAND_CHECK(curandGenerateUniform(gen, x, count));
    CURAND_CHECK(curandDestroyGenerator(gen));
}

std::string getSoftmaxTypeName(SOFTMAX_TYPE t) {
    switch (t) {
        case SOFTMAX_TYPE_NAIVE:
            return "Naive Softmax";
        case SOFTMAX_TYPE_SAFE:
            return "Safe Softmax";
        case SOFTMAX_TYPE_ONLINE:
            return "Online Softmax";
        default:
            assert(0);
            break;
    }
    return "";
}

template <int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void naive_softmax(const float *__restrict x, float *__restrict y, int V) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    x = x + bid * V;
    y = y + bid * V;

    float d_part = 0.0f;
    for (int elem_id = tid; elem_id < V; elem_id += THREADBLOCK_SIZE) {
        d_part += __expf(x[elem_id]);
    }

    typedef cub::BlockReduce<float, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    __shared__ float d_total_divide;

    float d = BlockReduce(tempStorage).Sum(d_part);
    if (tid == 0) {
        d_total_divide = __fdividef(1.0F, d);
    }
    __syncthreads();

    for (int elem_id = tid; elem_id < V; elem_id += THREADBLOCK_SIZE) {
        y[elem_id] = __expf(x[elem_id]) * d_total_divide;
    }
}

template <int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void safe_softmax(const float *__restrict x, float *__restrict y, int V) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    x = x + bid * V;
    y = y + bid * V;

    float max_part = -FLT_MAX;
    for (int elem_id = tid; elem_id < V; elem_id += THREADBLOCK_SIZE) {
        max_part = max_op(max_part, x[elem_id]);
    }

    typedef cub::BlockReduce<float, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;

    float max_value = BlockReduce(tempStorage).Reduce(max_part, max_op);
    __shared__ float shared_max;
    if (tid == 0) {
        shared_max = max_value;
    }
    __syncthreads();

    float sum_part = 0.0F;
    for (int elem_id = tid; elem_id < V; elem_id += THREADBLOCK_SIZE) {
        sum_part += __expf(x[elem_id] - shared_max);
    }

    float sum_value = BlockReduce(tempStorage).Sum(sum_part);
    __shared__ float shared_divide;
    if (tid == 0) {
        shared_divide = __fdividef(1.0F, sum_value);
    }
    __syncthreads();

    for (int elem_id = tid; elem_id < V; elem_id += THREADBLOCK_SIZE) {
        y[elem_id] = __expf(x[elem_id] - shared_max) * shared_divide;
    }
}

template <int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void online_softmax(const float *__restrict x, float *__restrict y, int V) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    x += bid * V;
    y += bid * V;

    MD md_part;
    md_part.m = -FLT_MAX;
    md_part.d = 0.0F;
    for (int elem_id = tid; elem_id < V; elem_id += THREADBLOCK_SIZE) {
        MD new_md;
        new_md.m = x[elem_id];
        new_md.d = 1.0F;
        md_part = reduce_max_op(md_part, new_md);
    }

    typedef cub::BlockReduce<MD, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    __shared__ float divide_value;

    MD md_value = BlockReduce(tempStorage).Reduce(md_part, reduce_max_op);
    if (tid == 0) {
        divide_value = __fdividef(1.0F, md_value.d);
    }
    __syncthreads();

    for (int elem_id = tid; elem_id < V; elem_id += THREADBLOCK_SIZE) {
        y[elem_id] = __expf(x[elem_id] - md_value.m) * divide_value;
    }
}

template <int MAX_K>
struct TopK {
    int p[MAX_K];
    float u[MAX_K];

    __device__ __forceinline__ void insert(float elem, int elem_id) {
        if (elem > u[MAX_K - 1]) {
            u[MAX_K - 1] = elem;
            p[MAX_K - 1] = elem_id;
        }
        for (int k = MAX_K - 2; k >= 0; --k) {
            if (u[k + 1] > u[k]) {
                float u2 = u[k];
                int p2 = p[k];
                u[k] = u[k + 1];
                p[k] = p[k + 1];
                u[k + 1] = u2;
                p[k + 1] = p2;
            }
        }
    }
};

template <int MAX_K>
__device__ __forceinline__ TopK<MAX_K> reduce_topk_op(const TopK<MAX_K> &a, const TopK<MAX_K> &b) {
    TopK<MAX_K> res = a;
    for (int i = 0; i < MAX_K; i++) {
        res.insert(b.u[i], b.p[i]);
    }
    return res;
}

std::vector<float> run_softmax(int V, int batchSize, SOFTMAX_TYPE type) {
    float *x;
    float *y;

    CUDA_CHECK(cudaMalloc(&x, batchSize * V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y, batchSize * V * sizeof(float)));

    switch (type) {
        case SOFTMAX_TYPE_NAIVE:
            naive_softmax<256><<<batchSize, 256>>>(x, y, V);
            break;

        case SOFTMAX_TYPE_SAFE:
            safe_softmax<256><<<batchSize, 256>>>(x, y, V);
            break;
        case SOFTMAX_TYPE_ONLINE:
            online_softmax<256><<<batchSize, 256>>>(x, y, V);
            break;

        default:
            assert(0);
    }

    std::vector<float> res(batchSize * V);
    CUDA_CHECK(cudaMemcpy(res.data(), y, batchSize * V * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    return res;
}

void compare_softmax_results(int V, int batch_size, SOFTMAX_TYPE t1, SOFTMAX_TYPE t2) {
    std::vector<float> res1 = run_softmax(V, batch_size, t1);
    std::vector<float> res2 = run_softmax(V, batch_size, t2);

    float max_diff = 0.0F;
    double total_diff = 0.0F;
    for (int i = 0; i < res1.size(); ++i) {
        float diff = fabs(res1[i] - res2[i]);
        max_diff = std::max(max_diff, diff);
        total_diff += diff;
    }
    std::cout << "Comparing " << getSoftmaxTypeName(t1) << " and " << getSoftmaxTypeName(t2) << ": Max diff = " << max_diff
              << ", Avg diff = " << (float)(total_diff / res1.size()) << std::endl;
}

int main(int argc, char *argv[]) {
    std::cout << "Softmax correctness check:" << std::endl;
    compare_softmax_results(300, 100, SOFTMAX_TYPE_NAIVE, SOFTMAX_TYPE_SAFE);
    compare_softmax_results(300, 100, SOFTMAX_TYPE_NAIVE, SOFTMAX_TYPE_ONLINE);
    return 0;
}