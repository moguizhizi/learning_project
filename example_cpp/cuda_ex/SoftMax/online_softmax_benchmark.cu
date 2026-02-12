#include <cub/cub.cuh>

enum SOFTMAX_TYPE {
    SOFTMAX_TYPE_NAIVE,
    SOFTMAX_TYPE_SAFE,
    SOFTMAX_TYPE_ONLINE,
};

__device__ __forceinline__ float max_op(float a, float b) {
    return fmaxf(a, b);
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
__launch_bounds__(THREADBLOCK_SIZE) __global__ void safe_softmax(const float *__restrict x, float *y, int V) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    x += bid * V;
    y += bid * V;

    float max_part = -FLT_MAX;
    for (int elem_id = tid; elem_id < V; elem_id += THREADBLOCK_SIZE) {
        max_part = max_op(max_part, x[elem_id]);
    }

    typedef cub::BlockReduce<float, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    float max_value = BlockReduce(tempStorage).Reduce(max_part, max_op);

    float sum_part = 0.0F;
    for (int elem_id = tid; elem_id < V; elem_id += THREADBLOCK_SIZE) {
        sum_part += __expf(x[elem_id] - max_value);
    }

    __syncthreads();
    __shared__ float shared_divide;
    float sum_value = BlockReduce(tempStorage).Sum(sum_part);
    if (tid == 0) {
        shared_divide = __fdividef(1.0F, sum_value);
    }

    __syncthreads();
    for (int elem_id = tid; elem_id < V; elem_id += THREADBLOCK_SIZE) {
        y[elem_id] = __expf(x[elem_id] - max_value) * shared_divide;
    }
}